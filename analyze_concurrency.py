"""Turns a concurrency_probe.py report + the SQLite trace into a leak verdict.

Two things are checked per spoken turn:

  1. LEAK      — a participant other than the turn's owner has signal energy on
                  the tracks it is subscribed to, while that turn is playing.
  2. SILENCE   — the owner has NO energy during its own turn. This is the control:
                  without it, "zero leaks" would also be satisfied by an agent that
                  publishes silence to everyone, which is a worse bug than a leak.

Usage:
    python3 analyze_concurrency.py .run/concurrency_before.json
    python3 analyze_concurrency.py .run/concurrency_after.json
"""

import json
import sys
from collections import defaultdict
from datetime import datetime

from agent_platform.trace_store import DEFAULT_DB, load_events

TRACE_FILE = DEFAULT_DB
RMS_THRESHOLD = 1200.0   # speech frames on a 22050Hz int16 stream; silence is <300
WINDOW_S = 0.25


def load_trace(path=TRACE_FILE):
    return load_events(path=path)


def epoch(ts):
    return datetime.fromisoformat(ts).timestamp()


def main():
    probe_path = sys.argv[1] if len(sys.argv) > 1 else "concurrency_probe.json"
    with open(probe_path) as f:
        probe = json.load(f)

    trace = load_trace(sys.argv[2] if len(sys.argv) > 2 else TRACE_FILE)
    has_windows = any("publish_start" in e for e in trace)

    def to_wall(mono_ts):
        return mono_ts - probe["start_mono"] + probe["start_wall"]

    def from_wall(wall_ts):
        return wall_ts - probe["start_wall"] + probe["start_mono"]

    def rel(wall_ts):
        """Seconds since the probe started, for readable window labels."""
        return wall_ts - probe["start_wall"]

    # identity -> {track_sid: [(wall_ts, rms)]}
    energy = defaultdict(lambda: defaultdict(list))
    meta = defaultdict(dict)
    for p in probe["participants"]:
        for t in p["tracks"]:
            meta[p["identity"]][t["sid"]] = t
            for ts, rms in t["windows"]:
                energy[p["identity"]][t["sid"]].append((to_wall(ts), rms))

    def peak_in(identity, start, end):
        best = 0.0
        for windows in energy[identity].values():
            for ts, rms in windows:
                if start <= ts <= end and rms > best:
                    best = rms
        return best

    def owner_of(session_id):
        for p in probe["participants"]:
            if session_id.startswith(p["identity"]):
                return p["identity"]
        return None

    print("=" * 78)
    print(f"PROBE {probe_path}  ({probe['wall_clock']}s, "
          f"{len(probe['participants'])} participants)")
    print(f"rms threshold={RMS_THRESHOLD:.0f}  "
          f"playback windows in trace={'yes' if has_windows else 'NO (unmeasurable)'}")
    print("=" * 78)

    # --- [1] routing ---------------------------------------------------------
    print("\n[1] TRACK SUBSCRIPTION / ROUTING")
    all_sids = defaultdict(list)
    for p in probe["participants"]:
        sids = sorted(t["sid"] for t in p["tracks"])
        agent = [meta[p["identity"]][s] for s in sids if meta[p["identity"]][s]["name"].startswith("agent-voice")]
        for sid in sids:
            all_sids[sid].append(p["identity"])
        for m in agent:
            heard = sum(1 for ts, rms in energy[p["identity"]][m["sid"]] if rms > RMS_THRESHOLD) * WINDOW_S
            print(f"    {p['identity']:<8} agent track {m['sid']} "
                  f"('{m['name']}'): {heard:.2f}s of audio above threshold, "
                  f"peak={max((r for _, r in energy[p['identity']][m['sid']]), default=0):.0f}")
    shared = {sid: ids for sid, ids in all_sids.items() if len(ids) > 1}
    if shared:
        print(f"    LEAK: {len(shared)} agent track(s) reachable by multiple participants:")
        for sid, ids in shared.items():
            print(f"      {sid} -> {ids}: every session's audio reaches every caller")
    else:
        print("    OK: no agent track is reachable by more than one participant")

    # --- [2] per-turn --------------------------------------------------------
    print("\n[2] PER-TURN CHECK")
    tts = [e for e in trace if e.get("event") == "tts" and e.get("played", True)]
    if not has_windows:
        print(f"    {len(tts)} spoken turn(s), but the trace carries no publish window,")
        print("    so per-turn audio attribution is impossible for this run.")
        print("    Only the structural check above is conclusive here.")
    else:
        # A listener's own turn windows. Audio on a listener's track that falls
        # outside all of them cannot belong to that listener, so it is a leak.
        own_windows = defaultdict(list)
        for e in tts:
            owner = owner_of(e["session_id"])
            if owner:
                own_windows[owner].append((epoch(e["publish_start"]), epoch(e["publish_end"]),
                                           e["call_id"]))

        def stray_bursts(identity, start, end):
            out = []
            for sid, windows in energy[identity].items():
                name = meta[identity][sid]["name"]
                if not name.startswith("agent-voice"):
                    continue
                for ts, rms in windows:
                    if not (start <= ts <= end) or rms <= RMS_THRESHOLD:
                        continue
                    if not any(s - 0.3 <= ts <= e + 0.3 for s, e, _ in own_windows[identity]):
                        out.append((round(rel(ts), 1), int(rms)))
            return out

        print(f"    {'session':<22} {'agent':<10} {'playback':<16} {'owner heard':<20} "
              f"{'ended':<10} other-listener stray audio")
        leaks = silences = 0
        for e in sorted(tts, key=lambda x: x["timestamp"]):
            start, end = epoch(e["publish_start"]), epoch(e["publish_end"])
            owner = owner_of(e["session_id"])
            reason = e.get("playback_end_reason", "?")
            own_peak = peak_in(owner, start, end) if owner else 0.0
            if own_peak <= RMS_THRESHOLD and reason == "completed":
                silences += 1
            others = []
            for identity in energy:
                if identity == owner:
                    continue
                stray = stray_bursts(identity, start, end)
                if stray:
                    others.append(f"{identity} LEAK{stray}")
                    leaks += len(stray)
                else:
                    others.append(f"{identity} clean")
            win = f"t+{rel(start):.1f}..{rel(end):.1f}"
            print(f"    {e['session_id']:<22} {e.get('agent', '?'):<10} {win:<16} "
                  f"peak={own_peak:.0f} ({reason})".ljust(90) + f"{reason:<10} "
                  + ", ".join(others))
        print(f"\n    {len(tts)} turn(s): {leaks} burst(s) of audio on a track that was not "
              f"the listener's own turn, {silences} turn(s) the owner never heard")
        print("    (concurrent sessions legitimately overlap in time; a burst only")
        print("     counts as a leak if it is outside the listener's own turns)")

    # --- [3] agent state -----------------------------------------------------
    print("\n[3] PER-SESSION AGENT STATE")
    by_session = defaultdict(list)
    for e in trace:
        if e.get("event") == "llm":
            by_session[e["session_id"]].append(e)
    for sid, evs in sorted(by_session.items()):
        evs.sort(key=lambda x: x["timestamp"])
        seq = " -> ".join(f"{ev.get('agent')}(->{ev.get('next_agent')})" for ev in evs)
        voices = sorted({ev.get("tts_voice") for ev in evs if ev.get("tts_voice")})
        print(f"    {sid:<22} {seq}")
        print(f"    {'':<22} voices: {', '.join(voices)}")
    barge = [e for e in trace if e.get("event") == "barge_in"]
    if barge:
        print("\n    barge-ins:")
        for e in barge:
            print(f"      {e['session_id']} interrupted call {e['call_id']} "
                  f"(agent={e.get('active_agent')}, p={e.get('speech_prob')})")
    ends = [e for e in trace if e.get("event") == "session_end"]
    if ends:
        print("\n    session teardown:")
        for e in ends:
            print(f"      {e['session_id']} ({e['identity']}): {e['turns_served']} turn(s), "
                  f"{e['barge_ins']} barge-in(s), final agent '{e['final_agent']}'")
    print()


if __name__ == "__main__":
    main()
