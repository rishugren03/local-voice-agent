import json
import statistics
from datetime import datetime

TRACE_FILE = "call_trace.jsonl"

# Define what "correct" looks like for each test case, so scoring is automatic.
# expect_tool: the tool call we expect to see (None if no tool should be used)
# expect_in_response: a substring that should appear in the final spoken response if correct
EXPECTATIONS = {
    "greeting": {"expect_tool": None, "expect_in_response": None},
    "math_simple": {"expect_tool": "calculate", "expect_in_response": "27"},
    "math_complex": {"expect_tool": "calculate", "expect_in_response": "564"},
    "calendar_check": {"expect_tool": "check_calendar", "expect_in_response": None},
    "handoff_trigger": {"expect_tool": "handoff_to_scheduler", "expect_in_response": None},
    "off_topic": {"expect_tool": None, "expect_in_response": None},
}


def load_events_in_window(start_ts, end_ts):
    events = []
    with open(TRACE_FILE) as f:
        for line in f:
            entry = json.loads(line)
            entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
            if start_ts <= entry_time <= end_ts:
                events.append(entry)
    return events


def group_by_call(events):
    calls = {}
    for e in events:
        calls.setdefault(e["call_id"], []).append(e)
    for cid in calls:
        calls[cid].sort(key=lambda e: e["timestamp"])
    return calls


def score_run(calls, expected_order):
    """
    Matches calls to expected test cases IN ORDER (since call_id doesn't carry
    the test case name). This assumes the eval run's calls appear in the trace
    in the same order the test suite fired them.
    """
    call_ids = list(calls.keys())
    results = []

    for i, case_id in enumerate(expected_order):
        if i >= len(call_ids):
            results.append({"case": case_id, "status": "MISSING", "detail": "No matching call found in trace"})
            continue

        cid = call_ids[i]
        call_events = calls[cid]
        llm_event = next((e for e in call_events if e["event"] == "llm"), None)
        expect = EXPECTATIONS[case_id]

        if not llm_event:
            results.append({"case": case_id, "status": "FAIL", "detail": "No LLM event found"})
            continue

        response = llm_event.get("response", "")
        tool_used = llm_event.get("tool", False)

        passed = True
        details = []

        if expect["expect_tool"] and not tool_used:
            passed = False
            details.append(f"expected tool call, none used")

        if expect["expect_in_response"] and expect["expect_in_response"] not in response:
            passed = False
            details.append(f"expected '{expect['expect_in_response']}' in response, got: \"{response}\"")

        results.append({
            "case": case_id,
            "status": "PASS" if passed else "FAIL",
            "detail": "; ".join(details) if details else "ok",
            "response": response
        })

    return results


def compute_latency_stats(calls):
    stt_times, llm_times, tts_times, total_times = [], [], [], []

    for cid, events in calls.items():
        stt = next((e["duration_s"] for e in events if e["event"] == "stt"), None)
        llm = next((e["duration_s"] for e in events if e["event"] == "llm"), None)
        tts = next((e["duration_s"] for e in events if e["event"] == "tts"), None)

        if stt: stt_times.append(stt)
        if llm: llm_times.append(llm)
        if tts: tts_times.append(tts)
        if stt and llm and tts:
            total_times.append(stt + llm + tts)

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


if __name__ == "__main__":
    print("Paste the eval window start and end timestamps from run_eval.py's output.")
    start_ts = float(input("Window start: ").strip())
    end_ts = float(input("Window end: ").strip())

    expected_order = ["greeting", "math_simple", "math_complex", "calendar_check", "handoff_trigger", "off_topic"]

    events = load_events_in_window(start_ts, end_ts)
    calls = group_by_call(events)

    print(f"\nFound {len(calls)} calls in this window.\n")

    print("=" * 60)
    print("TASK RESULTS")
    print("=" * 60)
    results = score_run(calls, expected_order)
    passed = 0
    for r in results:
        print(f"[{r['status']:7}] {r['case']:20} {r['detail']}")
        if r["status"] == "PASS":
            passed += 1
    print(f"\nScore: {passed}/{len(expected_order)} passed\n")

    print("=" * 60)
    print("LATENCY (seconds)")
    print("=" * 60)
    stats = compute_latency_stats(calls)
    print(f"STT   p50={stats['stt_p50']}  p90={stats['stt_p90']}")
    print(f"LLM   p50={stats['llm_p50']}  p90={stats['llm_p90']}")
    print(f"TTS   p50={stats['tts_p50']}  p90={stats['tts_p90']}")
    print(f"TOTAL p50={stats['total_p50']}  p90={stats['total_p90']}")
    print(f"\nBased on {stats['n_calls']} calls.")