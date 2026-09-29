# build_dashboard_data.py
"""Builds dashboard_data.json from the SQLite trace.

The JSONL version read every line into a dict keyed by call_id and re-grouped it
in Python. The same answer here is two queries: one per call, one for that call's
events, with SQLite doing the grouping and the summing.

The output shape is unchanged, so dashboard.html keeps working.
"""

import json
import os
import sys

from agent_platform.trace_store import DEFAULT_DB, connect_readonly, row_to_event

OUT_FILE = "dashboard_data.json"


def build(db_path):
    conn = connect_readonly(db_path)
    try:
        calls = conn.execute(
            "SELECT call_id, MIN(started_at) AS started_at, "
            "       SUM(COALESCE(duration_s, 0)) AS total_duration_s "
            "FROM events "
            "WHERE call_id IS NOT NULL "
        # The old code grouped by first event; MIN(started_at) is that, and it is
        # the call's own row rather than something inferred from its events.
        "GROUP BY call_id "
        # SUM ignores NULL, so COALESCE only matters for a call whose events all
        # lack a duration (a barge-in with no STT yet) — it totals 0, not NULL.
        "ORDER BY started_at DESC"
        ).fetchall()

        output = []
        for row in calls:
            events = [
                row_to_event(e)
                for e in conn.execute(
                    "SELECT * FROM events WHERE call_id = ? ORDER BY ts", (row["call_id"],)
                )
            ]
            output.append({
                "call_id": row["call_id"],
                "timestamp": row["started_at"],
                "total_duration_s": round(row["total_duration_s"] or 0, 2),
                "events": events,
            })
        return output
    finally:
        conn.close()


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TRACE_DB", DEFAULT_DB)
    output = build(db_path)

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Processed {len(output)} calls from {db_path} into {OUT_FILE}")


if __name__ == "__main__":
    main()
