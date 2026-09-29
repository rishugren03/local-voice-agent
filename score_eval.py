"""Scores the SQLite trace against the eval suite's expectations.

Scoring is per SESSION, never per global call order. run_eval.py can put several
simulated users in the room at once, and the agent interleaves their turns on one
CPU, so call_ids from different conversations land in the trace in whatever order
the LLM happened to answer. A scorer that walked call_ids positionally would hand
user A's math case user B's weather answer.

Within a session, calls are matched to cases by WHAT WAS HEARD — the transcript
the STT stage logged — rather than by position, because a turn that was dropped,
split by a barge-in, or answered late would otherwise shift every later case onto
the wrong call. Order is only the fallback when a transcript is unavailable or too
garbled to identify the utterance.

run_eval.py writes .run/eval_manifest.json (which identity ran which cases, and
the eval window); that file is what tells the scorer how to split the trace into
sessions and which cases each one owes. Without it the scorer still runs, but it
falls back to the default case order for every session it finds.

Usage:
    python3 score_eval.py                              # uses .run/eval_manifest.json
    python3 score_eval.py --manifest .run/eval_manifest.json
    python3 score_eval.py --start <epoch> --end <epoch>
    python3 score_eval.py --json                       # machine-readable report
    python3 score_eval.py --import-jsonl old.jsonl     # one-off: score a pre-SQLite trace
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

from agent_platform.trace_store import DEFAULT_DB, TraceStore, load_events

MANIFEST_FILE = os.path.join(".run", "eval_manifest.json")

# The suite in its default order, used when a run's manifest does not say which
# cases each session ran (a hand-typed window, an older trace, a session that was
# not a simulated eval user at all).
DEFAULT_CASES = ["greeting", "math_simple", "math_complex", "calendar_check",
                 "weather_check", "handoff_trigger", "off_topic"]

# Define what "correct" looks like for each test case, so scoring is automatic.
# expect_tool: the exact tool call we expect to see (None if no tool should be used)
# expect_in_response: a substring that should appear in the final spoken response
# expect_heard: words from the utterance this case plays (see generate_test_audio.py).
#   This is the identity of the turn — it is how a call is matched to a case, so
#   scoring survives interleaved sessions, a dropped turn, or a misheard word.
#   It is a matching hint, never an assertion: whisper.cpp base.en mishears often
#   enough that insisting on an exact transcript would fail cases the agent handled.
EXPECTATIONS = {
    "greeting": {"expect_tool": None, "expect_in_response": None,
                 "expect_heard": "hello how are you"},
    "math_simple": {"expect_tool": "calculate", "expect_in_response": "27",
                    "expect_heard": "12 plus 15"},
    "math_complex": {"expect_tool": "calculate", "expect_in_response": "564",
                     "expect_heard": "47 by 12"},
    "calendar_check": {"expect_tool": "check_calendar", "expect_in_response": None,
                       "expect_heard": "my calendar for august"},
    # Added with the schema-driven tool registry: this tool exists only in
    # tools.json + the agent's tools list, so this case is what proves a
    # config-only tool survives the whole pipeline.
    "weather_check": {"expect_tool": "get_weather", "expect_in_response": None,
                      "expect_heard": "weather like in seattle"},
    "handoff_trigger": {"expect_tool": "handoff_to_scheduler", "expect_in_response": None,
                        "expect_heard": "free slot next week"},
    "off_topic": {"expect_tool": None, "expect_in_response": None,
                  "expect_heard": "fun fact about space"},
}

# whisper.cpp writes the same number as digits in one clip and as words in the
# next, so both spellings are folded to digits before a transcript is compared
# with an expectation.
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}

# How much of a case's expect_heard must survive in a call's transcript before the
# two are considered the same utterance. Loose enough for "multiply 47 by 12" to
# survive one dropped or mangled word, tight enough that "12 plus 15" (3 tokens)
# cannot match a "47 by 12" call (best overlap 1 of 3).
HEARD_COVERAGE = 0.7

# Events that carry no call_id (session lifecycle) never join a call, but they are
# the only record of how a session ended, so they are kept aside for the report.
LIFECYCLE_EVENTS = ("session_end",)

# A turn the agent's VAD opened and then refused because the audio was silent. It
# has a call_id but it is not a call: no STT, no LLM, no answer. Counting it as
# one would invent a phantom turn in the score, so it is reported on its own line.
REJECTED_EVENTS = ("vad_rejected",)


def load_events_in_window(start_ts, end_ts, db_path):
    """Every event whose timestamp falls inside the window, in time order.

    The window is a range predicate on the events table's epoch column, so the
    scorer reads the eval run rather than the whole trace and filtering it here.
    """
    return load_events(start_ts, end_ts, db_path)


def group_by_session(events):
    """session_id -> call_id -> [events], plus the session's lifecycle events.

    session_id is the only stable link between a call and the conversation it
    belongs to; call_id is fresh per turn and means nothing on its own.
    """
    sessions = defaultdict(lambda: defaultdict(list))
    lifecycle = defaultdict(list)
    rejected = defaultdict(list)

    for e in events:
        session_id = e.get("session_id")
        if session_id is None:
            continue
        if e.get("call_id") is None or e["event"] in LIFECYCLE_EVENTS:
            lifecycle[session_id].append(e)
            continue
        if e["event"] in REJECTED_EVENTS:
            rejected[session_id].append(e)
            continue
        sessions[session_id][e["call_id"]].append(e)

    for calls in sessions.values():
        for call_id in calls:
            calls[call_id].sort(key=lambda e: e["timestamp"])

    return dict(sessions), dict(lifecycle), dict(rejected)


def ordered_calls(calls):
    """A session's calls, oldest first.

    Sorted by the FIRST event of each call rather than by call_id (which is random)
    or by the stt event, because a call that never reached STT still happened and
    still belongs in the sequence.
    """
    return [calls[cid] for cid in sorted(calls, key=lambda c: calls[c][0]["timestamp"])]


def event_of(call_events, event_type):
    return next((e for e in call_events if e["event"] == event_type), None)


def transcript_of(call_events):
    stt = event_of(call_events, "stt")
    return stt.get("text", "") if stt else ""


def tokens(text):
    """Comparable word list for a transcript or an expectation.

    Lowercase and punctuation-free, because whisper.cpp emits "August 15th." where
    the expectation says "august". Number words become digits, because it
    transcribes the same utterance as "12 plus 15" or "twelve plus fifteen"
    depending on the clip, and that difference must not decide whether a turn is
    recognised as the math case.
    """
    words = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split()
    return [NUMBER_WORDS.get(w, w) for w in words]


def heard_coverage(expected_tokens, actual_text):
    """Fraction of a case's expect_heard words present in a call's transcript.

    Set-based, so a word said twice still counts once: coverage answers "were
    these words said", not "how many times".
    """
    if not expected_tokens:
        return None
    actual = set(tokens(actual_text))
    return sum(1 for t in expected_tokens if t in actual) / len(expected_tokens)


def match_calls(session_calls, case_ids):
    """Pair each expected case with the call that actually served it.

    Two passes, because doing it case-by-case in one pass silently corrupts the
    rest of the session when one turn is missing: the case with no matching
    transcript would consume the call belonging to the NEXT case, and every case
    after it would shift down by one.

    Pass 1 anchors every case it can on the transcript — the unclaimed call that
    covers most of the case's expect_heard wins, if it clears HEARD_COVERAGE.
    This is what makes scoring immune to interleaved sessions, a rotated script,
    and calls arriving out of order.

    Pass 2 fills the gaps from the calls nothing claimed. A leftover call is only
    attributed to a case that sounds at least partly like it; a case whose words
    appear in none of the leftovers was genuinely never heard, and is reported
    MISSING rather than scored against a stranger's answer. With no transcripts at
    all in the session there is nothing to match on, so position is used and
    every such result says so.

    Returns (matches, unmatched_calls) where each match is
    {case, call, call_id, mode, coverage} and mode is "transcript" (all the
    expected words heard), "fuzzy" (enough of them to anchor out of order),
    "order" (fallback, flagged as unconfirmed), or "missing".
    """
    transcripts = [transcript_of(c) for c in session_calls]
    has_transcripts = any(t.strip() for t in transcripts)

    picked = [None] * len(session_calls)
    assigned = {}
    unanchored = []

    for case_id in case_ids:
        expect_tokens = tokens(EXPECTATIONS.get(case_id, {}).get("expect_heard") or "")
        if not expect_tokens:
            # No anchor declared for this case, so position is all there is.
            unanchored.append((case_id, expect_tokens))
            continue

        best_idx, best_cov = None, 0.0
        for i in range(len(session_calls)):
            if picked[i] is not None:
                continue
            cov = heard_coverage(expect_tokens, transcripts[i]) or 0.0
            if cov > best_cov:
                best_idx, best_cov = i, cov

        if best_idx is not None and best_cov >= HEARD_COVERAGE:
            picked[best_idx] = case_id
            assigned[case_id] = {
                "case": case_id,
                "call": session_calls[best_idx],
                "call_id": session_calls[best_idx][0]["call_id"],
                "mode": "transcript" if best_cov >= 1.0 else "fuzzy",
                "coverage": round(best_cov, 2),
            }
        else:
            unanchored.append((case_id, expect_tokens))

    leftovers = [i for i in range(len(session_calls)) if picked[i] is None]
    for case_id, expect_tokens in unanchored:
        if not leftovers:
            assigned[case_id] = {"case": case_id, "call": None, "call_id": None,
                                 "mode": "missing", "coverage": None}
            continue

        coverages = ([heard_coverage(expect_tokens, transcripts[i]) or 0.0
                      for i in leftovers] if expect_tokens else [])
        # Best coverage, earliest on a tie — which is exactly "the next call in
        # order" when nothing matches.
        best_pos = max(range(len(leftovers)), key=lambda k: coverages[k]) if coverages else 0
        best_cov = coverages[best_pos] if coverages else 0.0

        if has_transcripts and expect_tokens and best_cov <= 0.0:
            # Every remaining call is about something else, so this case was never
            # heard. Scoring it here would blame the agent for an utterance that
            # never reached it, and would consume a call another case needs.
            assigned[case_id] = {"case": case_id, "call": None, "call_id": None,
                                 "mode": "missing", "coverage": 0.0}
            continue

        idx = leftovers.pop(best_pos)
        picked[idx] = case_id
        assigned[case_id] = {
            "case": case_id,
            "call": session_calls[idx],
            "call_id": session_calls[idx][0]["call_id"],
            "mode": "order",
            "coverage": round(best_cov, 2) if has_transcripts else None,
        }

    matches = [assigned[case_id] for case_id in case_ids]
    unmatched = [session_calls[i] for i in range(len(session_calls)) if picked[i] is None]
    return matches, unmatched


def score_case(case_id, match):
    """Judge one case against the call that served it."""
    expect = EXPECTATIONS.get(case_id, {})
    result = {
        "case": case_id,
        "session_id": match["call"][0]["session_id"] if match["call"] else None,
        "call_id": match["call_id"],
        "match_mode": match["mode"],
        "heard_coverage": match["coverage"],
    }

    if match["call"] is None:
        detail = ("no call in this session heard this case's utterance"
                  if match["coverage"] is not None
                  else "no call in this session served this case")
        return {**result, "status": "MISSING", "detail": detail,
                "response": None, "tool_name": None, "tool_args": None}

    call_events = match["call"]
    llm_event = event_of(call_events, "llm")

    result["heard"] = transcript_of(call_events)
    result["tool_name"] = llm_event.get("tool_name") if llm_event else None
    result["tool_args"] = llm_event.get("tool_args") if llm_event else None
    result["response"] = llm_event.get("response", "") if llm_event else None
    result["agent"] = llm_event.get("agent") if llm_event else None

    if not llm_event:
        return {**result, "status": "FAIL", "detail": "No LLM event found"}

    tool_used = llm_event.get("tool", False)
    tool_name = result["tool_name"]
    response = result["response"] or ""

    passed = True
    details = []

    if expect.get("expect_tool") and not tool_used:
        passed = False
        details.append("expected tool call, none used")
    elif expect.get("expect_tool") and tool_name != expect["expect_tool"]:
        # The trace records which tool fired, so a wrong tool is a real failure
        # rather than "some tool was used" — that is what would catch a
        # config-only tool being offered but never actually invoked.
        passed = False
        details.append(f"expected tool '{expect['expect_tool']}', got '{tool_name}'")

    if expect.get("expect_in_response") and expect["expect_in_response"] not in response:
        passed = False
        details.append(f"expected '{expect['expect_in_response']}' in response, got: \"{response}\"")

    if match["mode"] == "order" and expect.get("expect_heard"):
        # This case is not confirmed by its transcript — the match is positional,
        # so the report has to say the attribution is a guess even when the tool
        # and the answer both line up.
        if match["coverage"] is None:
            details.append("no transcript in this session to match on; "
                           "attributed by position only")
        else:
            details.append(f"case utterance only partly heard (best match "
                           f"{match['coverage']}); attributed by position")

    return {**result, "status": "PASS" if passed else "FAIL",
            "detail": "; ".join(details) if details else "ok"}


def score_session(session_id, calls, case_ids):
    matches, unmatched = match_calls(ordered_calls(calls), case_ids)
    results = [score_case(m["case"], m) for m in matches]

    return {
        "session_id": session_id,
        "n_calls": len(calls),
        "results": results,
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "unmatched_calls": [c[0]["call_id"] for c in unmatched],
        "unmatched_detail": [
            {"call_id": c[0]["call_id"], "heard": transcript_of(c)} for c in unmatched
        ],
    }


def identity_for(session_id, identities):
    """Which simulated user a session belongs to.

    The agent names sessions f"{identity}-{uuid6}", so the longest identity that
    prefixes the session id is the owner. Longest wins because one identity can be
    a prefix of another ('eval-runner-a' vs 'eval-runner-ab').
    """
    matches = [i for i in identities
               if session_id == i or session_id.startswith(i + "-")]
    return max(matches, key=len) if matches else None


def resolve_case_lists(session_ids, manifest):
    """session_id -> (identity, case_ids), using the manifest when it has an answer."""
    users = (manifest or {}).get("users") or []
    identities = [u["identity"] for u in users if u.get("identity")]
    by_identity = {u["identity"]: u.get("cases") or DEFAULT_CASES for u in users}

    plan = {}
    for session_id in session_ids:
        identity = identity_for(session_id, identities) if identities else None
        case_ids = by_identity.get(identity) if identity else None
        if case_ids is None:
            # Either there is no manifest at all, or this session is not a
            # simulated eval user (a real browser caller, say). Score it against
            # the default suite so nothing in the window goes unreported.
            case_ids = list(DEFAULT_CASES)
        plan[session_id] = (identity, case_ids)
    return plan


def compute_latency_stats(calls):
    """p50/p90 per stage, plus end-to-end, across a set of calls."""
    stt_times, llm_times, tts_times, total_times = [], [], [], []

    for events in calls:
        stt = event_of(events, "stt")
        llm = event_of(events, "llm")
        tts = event_of(events, "tts")

        stt_s = stt.get("duration_s") if stt else None
        llm_s = llm.get("duration_s") if llm else None
        tts_s = tts.get("duration_s") if tts else None

        if stt_s is not None:
            stt_times.append(stt_s)
        if llm_s is not None:
            llm_times.append(llm_s)
        if tts_s is not None:
            tts_times.append(tts_s)
        if stt_s is not None and llm_s is not None and tts_s is not None:
            total_times.append(stt_s + llm_s + tts_s)

    def pct(data, p):
        if not data:
            return None
        return round(sorted(data)[min(int(len(data) * p), len(data) - 1)], 2)

    return {
        "stt_p50": pct(stt_times, 0.5), "stt_p90": pct(stt_times, 0.9),
        "llm_p50": pct(llm_times, 0.5), "llm_p90": pct(llm_times, 0.9),
        "tts_p50": pct(tts_times, 0.5), "tts_p90": pct(tts_times, 0.9),
        "total_p50": pct(total_times, 0.5), "total_p90": pct(total_times, 0.9),
        "n_calls": len(calls),
    }


def first_seen(sessions, session_id):
    """When a session's first event landed — the order sessions are reported in."""
    return min(e["timestamp"]
               for call_events in sessions[session_id].values()
               for e in call_events)


def build_report(events, manifest):
    """Score every session in the window, independently of the others."""
    sessions, lifecycle, rejected = group_by_session(events)
    plan = resolve_case_lists(sorted(sessions, key=lambda s: first_seen(sessions, s)),
                              manifest)

    session_reports = []
    for session_id in plan:
        identity, case_ids = plan[session_id]
        report = score_session(session_id, sessions[session_id], case_ids)
        report["identity"] = identity
        report["cases_expected"] = len(case_ids)
        report["known_cases"] = identity is not None
        report["latency"] = compute_latency_stats(list(sessions[session_id].values()))
        report["rejected_silent_turns"] = [
            {"audio_s": e.get("audio_s"), "peak": e.get("peak")} for e in rejected.get(session_id, [])
        ]
        end = next((e for e in lifecycle.get(session_id, [])
                    if e["event"] == "session_end"), None)
        if end:
            report["session_end"] = {
                "final_agent": end.get("final_agent"),
                "turns_served": end.get("turns_served"),
                "barge_ins": end.get("barge_ins"),
            }
        session_reports.append(report)

    all_calls = [c for s in sessions.values() for c in s.values()]
    total = sum(len(r["results"]) for r in session_reports)
    passed = sum(r["passed"] for r in session_reports)

    return {
        "window": (manifest or {}).get("window"),
        "n_sessions": len(session_reports),
        "n_calls": len(all_calls),
        "passed": passed,
        "total": total,
        "sessions": session_reports,
        "latency_overall": compute_latency_stats(all_calls),
    }


def print_report(report):
    if report["window"]:
        w = report["window"]
        print(f"Window: {w.get('start')} .. {w.get('end')}")

    print(f"\nFound {report['n_calls']} calls across {report['n_sessions']} session(s).\n")

    for session in report["sessions"]:
        label = session["identity"] or "unknown caller"
        print("=" * 78)
        print(f"SESSION {session['session_id']}  (identity: {label}, "
              f"{session['n_calls']} calls, {session['cases_expected']} cases expected)")
        if not session["known_cases"]:
            print("  note: this session is not in the run manifest, so the default case "
                  "order was assumed")
        print("=" * 78)
        for r in session["results"]:
            print(f"[{r['status']:7}] {r['case']:16} call={str(r['call_id']):10} "
                  f"match={r['match_mode']:9} {r['detail']}")
        print(f"  session score: {session['passed']}/{len(session['results'])}")

        if session["unmatched_calls"]:
            print(f"  {len(session['unmatched_calls'])} call(s) matched no expected case:")
            for extra in session["unmatched_detail"]:
                print(f"    {extra['call_id']}: heard {extra['heard']!r}")

        if session["rejected_silent_turns"]:
            n = len(session["rejected_silent_turns"])
            total_s = sum(t.get("audio_s") or 0 for t in session["rejected_silent_turns"])
            print(f"  {n} silent turn(s) dropped by the VAD before transcription "
                  f"({total_s:.1f}s of audio, not scored as calls)")

        if session.get("session_end"):
            end = session["session_end"]
            print(f"  session_end: {end['turns_served']} turn(s), "
                  f"{end['barge_ins']} barge-in(s), final agent '{end['final_agent']}'")

        stats = session["latency"]
        print(f"  latency: stt p50={stats['stt_p50']} llm p50={stats['llm_p50']} "
              f"tts p50={stats['tts_p50']} total p50={stats['total_p50']} "
              f"(p90 total={stats['total_p90']})")

    print("\n" + "=" * 78)
    print(f"TOTAL: {report['passed']}/{report['total']} passed")
    print("=" * 78)
    for session in report["sessions"]:
        print(f"  {str(session['identity'] or '?'):16} {session['session_id']:22} "
              f"{session['passed']}/{len(session['results'])}")
    print()

    print("Per-case detail:")
    for session in report["sessions"]:
        print(f"  [{session['identity'] or '?'}] {session['session_id']}")
        for r in session["results"]:
            print(f"    {r['case']:18} tool={str(r.get('tool_name')):22} "
                  f"args={r.get('tool_args')}")
            print(f"    {'':18} heard: {r.get('heard')!r}")
            print(f"    {'':18} said:  {r.get('response')!r}")

    print("=" * 78)
    print("LATENCY (seconds, all sessions pooled)")
    print("=" * 78)
    stats = report["latency_overall"]
    print(f"STT   p50={stats['stt_p50']}  p90={stats['stt_p90']}")
    print(f"LLM   p50={stats['llm_p50']}  p90={stats['llm_p90']}")
    print(f"TTS   p50={stats['tts_p50']}  p90={stats['tts_p90']}")
    print(f"TOTAL p50={stats['total_p50']}  p90={stats['total_p90']}")
    print(f"\nBased on {stats['n_calls']} calls.")


def load_manifest(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def sys_stdin_is_a_tty():
    return sys.stdin.isatty()


def resolve_window(args, manifest):
    """The eval window: from the manifest, from flags, or typed in.

    The window is widened by --pad because the tail of the last turn's TTS can
    land just after the runner's window closes. The pad is kept small on purpose:
    run_eval.py already stays connected for a drain after the last case, and a
    wide pad would reach back into a previous run's session and score it as a
    caller nobody declared.
    """
    pad = args.pad
    if args.start is not None and args.end is not None:
        return args.start - pad, args.end + pad

    window = (manifest or {}).get("window") or {}
    if "start" in window and "end" in window:
        return window["start"] - pad, window["end"] + pad

    if not sys_stdin_is_a_tty():
        raise SystemExit(
            "No window given and no manifest found. Pass --start/--end, or "
            "--manifest with the run_eval.py output."
        )

    print("Paste the eval window start and end timestamps from run_eval.py's output.")
    start_ts = float(input("Window start: ").strip())
    end_ts = float(input("Window end: ").strip())
    return start_ts - pad, end_ts + pad


def main():
    ap = argparse.ArgumentParser(description="Score the eval suite from the SQLite trace")
    ap.add_argument("--manifest", default=MANIFEST_FILE,
                    help=f"run_eval.py manifest (default: {MANIFEST_FILE})")
    ap.add_argument("--trace", default=DEFAULT_DB,
                    help=f"trace database (default: {DEFAULT_DB})")
    ap.add_argument("--import-jsonl", metavar="PATH",
                    help="load a legacy call_trace.jsonl into the database, then exit")
    ap.add_argument("--start", type=float, help="eval window start (epoch seconds)")
    ap.add_argument("--end", type=float, help="eval window end (epoch seconds)")
    ap.add_argument("--pad", type=float, default=5.0,
                    help="seconds to widen the window by, to catch the last turn's TTS")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args()

    if args.import_jsonl:
        with TraceStore(args.trace) as store:
            imported, skipped = store.import_jsonl(args.import_jsonl)
        print(f"Imported {imported} events from {args.import_jsonl} into {args.trace}"
              + (f" ({skipped} already present)" if skipped else ""))
        return

    manifest = load_manifest(args.manifest)
    start_ts, end_ts = resolve_window(args, manifest)

    events = load_events_in_window(start_ts, end_ts, args.trace)
    report = build_report(events, manifest)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
