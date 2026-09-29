"""Deterministic checks for the eval scorer — no LiveKit, no Ollama, no Piper.

The live two-user eval (run_eval.py --users 2) is what proves the scorer against
real interleaved traces, but it takes minutes and needs every local service up.
These checks pin the same attribution logic on a synthetic trace in under a second:
two conversations whose turns interleave in the database, one of which drops a
turn, mishears another, and picks up a stray call.

Run: python3 test_scoring.py
"""

import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta

import score_eval as scorer
from agent_platform.trace_store import TraceStore

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


BASE = datetime(2026, 1, 1, 12, 0, 0)


class TraceBuilder:
    """Writes a SQLite trace and a run_eval manifest for a fake run.

    Turns are appended in whatever order the caller asks for, so a scenario can
    interleave two conversations the way the real agent does when both callers are
    mid-turn on the same CPU. Each event is written through the real TraceStore
    with an explicit timestamp, so these tests exercise the same insert path the
    live agent uses rather than a stand-in for it.
    """

    def __init__(self, directory):
        # Each builder gets its own file. The JSONL version reopened the same path
        # in "w" mode, so every builder started from an empty trace; a private
        # subdirectory gives the same isolation now that the database is opened
        # for append rather than truncated on write.
        self.path = os.path.join(tempfile.mkdtemp(dir=directory), "call_trace.db")
        self.store = TraceStore(self.path)
        self.clock = 0.0

    def _emit(self, session_id, call_id, event, **data):
        self.clock += 0.4
        self.store.log_event(session_id, call_id, event, data,
                             at=BASE + timedelta(seconds=self.clock))

    def turn(self, session_id, call_id, heard, tool=None, response="ok",
             tool_args=None, duration_s=1.0):
        self._emit(session_id, call_id, "stt", duration_s=duration_s, text=heard)
        self._emit(session_id, call_id, "llm", duration_s=duration_s * 2,
                   response=response, tool=bool(tool), tool_name=tool,
                   tool_args=tool_args, agent="primary", next_agent="primary",
                   llm_model="phi4-mini", tts_voice="lessac")
        self._emit(session_id, call_id, "tts", duration_s=duration_s,
                   returncode=0, agent="primary", next_agent="primary",
                   played=True)

    def bare_call(self, session_id, call_id, heard=None):
        """A call with an stt event and no answer — the agent never got to it."""
        self._emit(session_id, call_id, "stt", duration_s=0.9, text=heard or "")

    def session_end(self, session_id, identity, turns, barge_ins=0, final_agent="primary"):
        self._emit(session_id, None, "session_end", identity=identity,
                   final_agent=final_agent, turns_served=turns, barge_ins=barge_ins)

    def write(self, window_users):
        self.store.close()
        return {
            "window": {"start": BASE.timestamp() - 5, "end": BASE.timestamp() + 3600},
            "cases": [c["id"] for _, cases in window_users for c in cases],
            "users": [
                {"identity": identity, "cases": [c["id"] for c in cases],
                 "started_at": BASE.timestamp(), "ended_at": BASE.timestamp() + 60}
                for identity, cases in window_users
            ],
        }

    def report(self, manifest, trace=None):
        events = scorer.load_events_in_window(BASE.timestamp() - 5,
                                              BASE.timestamp() + 3600,
                                              trace or self.path)
        return scorer.build_report(events, manifest)


def session_by_identity(report, identity):
    return next(s for s in report["sessions"] if s["identity"] == identity)


def case_result(report, identity, case_id):
    session = session_by_identity(report, identity)
    return next(r for r in session["results"] if r["case"] == case_id)


def scenario_two_interleaved_users(directory):
    print("\n1. Two concurrent conversations interleave; each case lands on its own session")
    b = TraceBuilder(directory)
    a, c = "eval-runner-a-111111", "eval-runner-b-222222"

    # Emitted strictly round-robin: in the trace file, a's greeting sits next to b's
    # greeting. Global call order would hand a's math case b's answer.
    b.turn(a, "a1", "Hello, how are you?", response="Hello!")
    b.turn(c, "b1", "Hello, how are you?", response="Hello!")
    b.turn(a, "a2", "What is 12 plus 15?", tool="calculate",
           tool_args={"expression": "12+15"}, response="The sum is 27.")
    b.turn(c, "b2", "What is 12 plus 15?", tool="calculate",
           tool_args={"expression": "12+15"}, response="27.")
    b.turn(a, "a3", "Multiply 47 by 12", tool="calculate",
           tool_args={"expression": "47*12"}, response="The product is 564.")
    b.turn(c, "b3", "Multiply 47 by 12", tool="get_weather",
           tool_args={"location": "Seattle"}, response="It is raining.")

    manifest = b.write([
        ("eval-runner-a", [{"id": "greeting"}, {"id": "math_simple"}, {"id": "math_complex"}]),
        ("eval-runner-b", [{"id": "greeting"}, {"id": "math_simple"}, {"id": "math_complex"}]),
    ])
    report = b.report(manifest)

    check("both sessions scored separately", report["n_sessions"] == 2,
          f"{report['n_sessions']} session(s)")

    check("a's math_simple used calculate",
          case_result(report, "eval-runner-a", "math_simple")["tool_name"] == "calculate")
    check("b's math_complex used its own (wrong) tool",
          case_result(report, "eval-runner-b", "math_complex")["tool_name"] == "get_weather",
          "attribution holds even when the answer differs")
    check("a's math_complex saw 564",
          case_result(report, "eval-runner-a", "math_complex")["status"] == "PASS")
    check("b's math_complex failed on the tool it called",
          case_result(report, "eval-runner-b", "math_complex")["status"] == "FAIL")
    check("every call id stayed inside its own session",
          all(r["session_id"] == a for r in session_by_identity(report, "eval-runner-a")["results"])
          and all(r["session_id"] == c
                  for r in session_by_identity(report, "eval-runner-b")["results"]))
    check("every case matched on its transcript, not its position",
          all(r["match_mode"] == "transcript"
              for s in report["sessions"] for r in s["results"]))


def scenario_rotated_order(directory):
    print("\n2. A user whose script is rotated still scores against the manifest order")
    b = TraceBuilder(directory)
    identity, session = "eval-runner-b", "eval-runner-b-222222"

    # The manifest lists the suite in its canonical order; the caller actually ran
    # it starting at case 3, so trace order and case order disagree completely.
    b.turn(session, "c3", "What's the weather like in Seattle right now?",
           tool="get_weather", tool_args={"location": "Seattle"}, response="48 degrees.")
    b.turn(session, "c1", "Hello, how are you?", response="Hello!")
    b.turn(session, "c2", "What is 12 plus 15?", tool="calculate",
           tool_args={"expression": "12+15"}, response="The sum is 27.")

    manifest = b.write([(identity, [{"id": "greeting"}, {"id": "math_simple"},
                                    {"id": "weather_check"}])])
    report = b.report(manifest)

    check("greeting matched the greeting call, not the first one",
          case_result(report, identity, "greeting")["call_id"] == "c1",
          f"got {case_result(report, identity, 'greeting')['call_id']}")
    check("math_simple matched the math call",
          case_result(report, identity, "math_simple")["call_id"] == "c2",
          f"got {case_result(report, identity, 'math_simple')['call_id']}")
    check("weather_check matched the weather call",
          case_result(report, identity, "weather_check")["call_id"] == "c3",
          f"got {case_result(report, identity, 'weather_check')['call_id']}")
    check("all three cases passed", report["passed"] == 3, f"{report['passed']}/3")


def scenario_dropped_and_extra_calls(directory):
    print("\n3. A dropped turn does not shift every later case onto the wrong call")
    b = TraceBuilder(directory)
    identity, session = "eval-runner-a", "eval-runner-a-111111"

    b.turn(session, "a1", "Hello, how are you?", response="Hello!")
    # math_simple's audio was never transcribed: no call for it at all.
    b.turn(session, "a3", "Multiply 47 by 12", tool="calculate",
           tool_args={"expression": "47*12"}, response="The product is 564.")
    # A stray call the suite never asked for (an echo picked up as a new turn).
    b.bare_call(session, "aX", "what")
    b.turn(session, "a4", "What's the weather like in Seattle right now?",
           tool="get_weather", tool_args={"location": "Seattle"}, response="48 degrees.")

    manifest = b.write([(identity, [{"id": "greeting"}, {"id": "math_simple"},
                                    {"id": "math_complex"}, {"id": "weather_check"}])])
    report = b.report(manifest)

    check("the missing turn is MISSING, not silently re-scored",
          case_result(report, identity, "math_simple")["status"] == "MISSING")
    check("math_complex still got the 47x12 call",
          case_result(report, identity, "math_complex")["call_id"] == "a3",
          f"got {case_result(report, identity, 'math_complex')['call_id']}")
    check("weather_check still got its own call",
          case_result(report, identity, "weather_check")["call_id"] == "a4",
          f"got {case_result(report, identity, 'weather_check')['call_id']}")
    check("the stray call is reported, not consumed by a case",
          session_by_identity(report, identity)["unmatched_calls"] == ["aX"],
          str(session_by_identity(report, identity)["unmatched_calls"]))
    check("3 of 4 passed", report["passed"] == 3, f"{report['passed']}/4")


def scenario_misheard_transcript(directory):
    print("\n4. A misheard utterance still finds its own turn")
    b = TraceBuilder(directory)
    identity, session = "eval-runner-a", "eval-runner-a-111111"

    b.turn(session, "a1", "Hello, how are you?", response="Hello!")
    # whisper spells the same number as a word in one clip and as digits in the next.
    b.turn(session, "a2", "what is twelve plus fifteen", tool="calculate",
           tool_args={"expression": "12+15"}, response="The sum is 27.")
    b.turn(session, "a3", "Multiply 47 by 12", tool="calculate",
           tool_args={"expression": "47*12"}, response="The product is 564.")
    # One word short of the full utterance, but enough to anchor it.
    b.turn(session, "a4", "what's the weather in Seattle right now",
           tool="get_weather", tool_args={"location": "Seattle"}, response="48 degrees.")

    manifest = b.write([(identity, [{"id": "greeting"}, {"id": "math_simple"},
                                    {"id": "math_complex"}, {"id": "weather_check"}])])
    report = b.report(manifest)

    check("numbers spoken as words still match math_simple",
          case_result(report, identity, "math_simple")["call_id"] == "a2",
          f"call={case_result(report, identity, 'math_simple')['call_id']} "
          f"mode={case_result(report, identity, 'math_simple')['match_mode']}")
    check("that match is reported as exact, not positional",
          case_result(report, identity, "math_simple")["match_mode"] == "transcript",
          case_result(report, identity, "math_simple")["match_mode"])
    check("a partly-heard utterance is still anchored to its own turn",
          case_result(report, identity, "weather_check")["call_id"] == "a4",
          f"call={case_result(report, identity, 'weather_check')['call_id']}")
    check("and is reported as a fuzzy match",
          case_result(report, identity, "weather_check")["match_mode"] == "fuzzy",
          case_result(report, identity, "weather_check")["match_mode"])
    check("all four cases passed", report["passed"] == 4, f"{report['passed']}/4")


def scenario_positional_fallback(directory):
    print("\n5. With no transcripts at all, scoring falls back to order and says so")
    b = TraceBuilder(directory)
    identity, session = "eval-runner-a", "eval-runner-a-111111"

    b.turn(session, "a1", "", tool="calculate",
           tool_args={"expression": "12+15"}, response="The sum is 27.")
    b.turn(session, "a2", "", tool="calculate",
           tool_args={"expression": "47*12"}, response="The product is 564.")

    manifest = b.write([(identity, [{"id": "math_simple"}, {"id": "math_complex"}])])
    report = b.report(manifest)

    check("both cases are still scored when nothing can anchor them",
          report["passed"] == 2, f"{report['passed']}/2")
    check("the report says the attribution is positional only",
          all(r["match_mode"] == "order"
              and "attributed by position only" in r["detail"]
              for r in session_by_identity(report, identity)["results"]),
          session_by_identity(report, identity)["results"][0]["detail"])


def scenario_window_boundaries(directory):
    print("\n6. The window filter and latency stats behave per session")
    b = TraceBuilder(directory)
    a, c = "eval-runner-a-111111", "eval-runner-b-222222"
    b.turn(a, "a1", "Hello, how are you?", response="Hello!", duration_s=0.5)
    b.turn(c, "b1", "What is 12 plus 15?", tool="calculate",
           tool_args={"expression": "12+15"}, response="27.", duration_s=1.5)
    b.session_end(a, "eval-runner-a", turns=1, barge_ins=2, final_agent="trivia")
    b.session_end(c, "eval-runner-b", turns=1)

    manifest = b.write([
        ("eval-runner-a", [{"id": "greeting"}]),
        ("eval-runner-b", [{"id": "math_simple"}]),
    ])
    report = b.report(manifest)

    check("session_end is not counted as a call",
          session_by_identity(report, "eval-runner-a")["n_calls"] == 1,
          str(session_by_identity(report, "eval-runner-a")["n_calls"]))
    latency_a = session_by_identity(report, "eval-runner-a")["latency"]["total_p50"]
    latency_b = session_by_identity(report, "eval-runner-b")["latency"]["total_p50"]
    check("per-session latency only covers that session's own call",
          latency_a == 2.0 and latency_b == 6.0,
          f"a={latency_a} b={latency_b}")
    check("pooled latency spans both sessions",
          report["latency_overall"]["total_p50"] in (2.0, 6.0),
          str(report["latency_overall"]["total_p50"]))
    check("the session's teardown is reported",
          session_by_identity(report, "eval-runner-a")["session_end"]["barge_ins"] == 2)

    # A window that excludes everything must score nothing rather than crash.
    events = scorer.load_events_in_window(BASE.timestamp() + 10000,
                                          BASE.timestamp() + 20000, b.path)
    empty = scorer.build_report(events, manifest)
    check("an empty window scores 0 calls", empty["n_calls"] == 0 and empty["total"] == 0)


def scenario_session_without_manifest_entry(directory):
    print("\n7. A session the manifest does not know about is still reported")
    b = TraceBuilder(directory)
    identity, session = "eval-runner-a", "eval-runner-a-111111"
    b.turn(session, "a1", "Hello, how are you?", response="Hello!")

    manifest = b.write([("eval-runner-a", [{"id": "greeting"}])])
    # Reassign the trace to a caller nobody declared. This rewrote the whole
    # JSONL file under the old loader; it is one UPDATE now.
    with sqlite3.connect(b.path) as conn:
        conn.execute("UPDATE events SET session_id = 'browser-caller-999999' "
                     "WHERE session_id = ?", (session,))
    report = scorer.build_report(
        scorer.load_events_in_window(BASE.timestamp() - 5, BASE.timestamp() + 3600, b.path),
        manifest)

    check("the unknown session is scored, not dropped",
          report["n_sessions"] == 1 and report["total"] == 7,
          f"{report['n_sessions']} session(s), {report['total']} case(s)")
    check("it is flagged as not-in-manifest",
          report["sessions"][0]["identity"] is None
          and report["sessions"][0]["known_cases"] is False,
          f"identity={report['sessions'][0]['identity']}")


def scenario_rejected_silent_turns(directory):
    print("\n8. A turn the VAD dropped for being silent is not scored as a call")
    b = TraceBuilder(directory)
    identity, session = "eval-runner-a", "eval-runner-a-222222"
    b.turn(session, "a1", "Hello, how are you?", response="Hello!")
    # What the agent logs when the VAD gate opens a turn on digital silence and
    # the peak floor refuses it: a call_id, but no stt / llm / tts behind it.
    b._emit(session, "a0", "vad_rejected", reason="audio below peak floor",
            peak=0, peak_floor=500, audio_s=0.67)
    b.turn(session, "a2", "What is 12 plus 15?", tool="calculate",
           tool_args={"expression": "12 + 15"}, response="27")

    manifest = b.write([(identity, [{"id": "greeting"}, {"id": "math_simple"}])])
    report = b.report(manifest)
    session_report = session_by_identity(report, identity)

    check("the rejected turn is not counted as a call",
          report["n_calls"] == 2, f"n_calls={report['n_calls']}")
    check("the rejected turn is not matched to a case",
          session_report["unmatched_calls"] == []
          and len(session_report["results"]) == 2,
          f"unmatched={session_report['unmatched_calls']}")
    check("both real cases still score",
          session_report["passed"] == 2
          and [r["case"] for r in session_report["results"]] == ["greeting", "math_simple"])
    check("it is reported as a dropped silent turn",
          len(session_report["rejected_silent_turns"]) == 1
          and session_report["rejected_silent_turns"][0]["audio_s"] == 0.67,
          f"{session_report['rejected_silent_turns']}")

    # A session whose every turn was rejected never had a conversation, so it
    # must not appear as seven MISSING cases.
    b2 = TraceBuilder(directory)
    b2._emit(session, "z0", "vad_rejected", reason="audio below peak floor",
             peak=0, peak_floor=500, audio_s=0.67)
    manifest2 = b2.write([(identity, [{"id": "greeting"}])])
    report2 = b2.report(manifest2)
    check("a session with only rejected turns is not reported as missing cases",
          report2["n_sessions"] == 0 and report2["total"] == 0,
          f"{report2['n_sessions']} session(s), {report2['total']} case(s)")


def main():
    print("Eval scorer checks")
    with tempfile.TemporaryDirectory() as d:
        scenario_two_interleaved_users(d)
        scenario_rotated_order(d)
        scenario_dropped_and_extra_calls(d)
        scenario_misheard_transcript(d)
        scenario_positional_fallback(d)
        scenario_window_boundaries(d)
        scenario_session_without_manifest_entry(d)
        scenario_rejected_silent_turns(d)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
