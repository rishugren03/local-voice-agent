# build_dashboard_data.py
import json
from collections import defaultdict

calls = defaultdict(list)

with open("call_trace.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        calls[entry["call_id"]].append(entry)

output = []
for call_id, events in calls.items():
    events.sort(key=lambda e: e["timestamp"])
    total_duration = sum(e.get("duration_s", 0) for e in events)
    output.append({
        "call_id": call_id,
        "timestamp": events[0]["timestamp"],
        "total_duration_s": round(total_duration, 2),
        "events": events
    })

output.sort(key=lambda c: c["timestamp"], reverse=True)

with open("dashboard_data.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Processed {len(output)} calls into dashboard_data.json")