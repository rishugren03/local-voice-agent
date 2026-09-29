"""Control-plane API tests.

Written as plain asserts run by `python test_control_plane.py`, matching the
style of the other test files in this repo — no pytest dependency, and the
output is the same PASS/FAIL list you can read top to bottom.

Every test runs against a temporary database, pointed at by monkeypatching the
module-level paths before the app is imported. The real call_trace.db and the
real agent JSON are never touched: these tests delete agents and rewrite tools,
and that has no business happening to live data.
"""

import json
import os
import sys
import tempfile
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TMP = tempfile.mkdtemp(prefix="control_plane_test_")
CONFIG_DB = os.path.join(TMP, "config.db")
TRACE_DB = os.path.join(TMP, "trace.db")
AGENTS_DIR = os.path.join(TMP, "agents")
TOOLS_FILE = os.path.join(TMP, "tools.json")

from fastapi.testclient import TestClient  # noqa: E402

import control_plane.app as app_module  # noqa: E402
from agent_platform.config_store import ConfigStore  # noqa: E402
from agent_platform.trace_store import TraceStore  # noqa: E402

# Repoint the app at the temp database before any request is made.
app_module.CONFIG_DB = CONFIG_DB
app_module.TRACE_DB = TRACE_DB
app_module.AGENTS_DIR = AGENTS_DIR
app_module.TOOLS_FILE = TOOLS_FILE

client = TestClient(app_module.app)

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append((name, detail))
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def section(title):
    print(f"\n=== {title} ===")


def seed():
    """A minimal valid config in the temp database.

    Two agents and one tool, not the real three-agent set, so a test can assert
    on exact counts without every other test's writes changing them.
    """
    store = ConfigStore(CONFIG_DB)
    store.save_tool("echo", {
        "description": "Echo the given text back.",
        "args_schema": {"properties": {"text": {
            "type": "string", "description": "What to echo", "example": "hi"}}},
    })
    store.save_agent({
        "id": "primary", "system_prompt": "You are the primary agent.",
        "tools": ["echo"], "handoffs": ["helper"], "rules": ["Be brief."],
    })
    store.save_agent({
        "id": "helper", "system_prompt": "You help.",
        "tools": ["echo"], "handoffs": [], "rules": [],
    })
    store.close()


seed()

# -- version and empty-state reads ------------------------------------------

section("version and reads")

r = client.get("/api/config/version")
check("version endpoint answers", r.status_code == 200, r.text)
body = r.json()
check("version reports the agent count", body.get("agents") == 2, str(body))
# Two, not one: saving `primary` with a handoff to `helper` also creates the
# handoff_to_helper tool row the worker's build_prompt() looks up.
check("version reports the tool count", body.get("tools") == 2, str(body))
check("version is an integer", isinstance(body.get("config_version"), int), str(body))

r = client.get("/api/agents")
check("agents list is 200", r.status_code == 200, r.text)
check("agents list has both agents", len(r.json()["agents"]) == 2, r.text)

r = client.get("/api/tools")
check("tools list is 200", r.status_code == 200, r.text)
check("tools list has the seeded tool and the auto-created handoff",
      {t["name"] for t in r.json()["tools"]} == {"echo", "handoff_to_helper"}, r.text)

# -- agent create / update / delete -----------------------------------------

section("agent CRUD")

r = client.put("/api/agents/primary", json={
    "id": "primary", "system_prompt": "Updated prompt.",
    "tools": ["echo"], "handoffs": ["helper"], "rules": [],
})
check("agent update is 200", r.status_code == 200, r.text)
check("agent update changed the prompt",
      r.json()["agent"]["system_prompt"] == "Updated prompt.", r.text)
v_after_update = r.json()["config_version"]

r = client.get("/api/agents/primary")
check("agent read back matches", r.json()["system_prompt"] == "Updated prompt.", r.text)

r = client.put("/api/agents/newcomer", json={
    "id": "newcomer", "system_prompt": "Brand new.",
    "tools": ["echo"], "handoffs": [], "rules": [],
})
check("agent create is 200", r.status_code == 200, r.text)
check("create bumped the version", r.json()["config_version"] > v_after_update, r.text)

r = client.patch("/api/agents/newcomer", json={
    "id": "newcomer", "system_prompt": "Patched.", "rules": ["Only say this."],
})
check("agent patch is 200", r.status_code == 200, r.text)
patched = r.json()["agent"]
check("patch changed the prompt", patched["system_prompt"] == "Patched.", r.text)
check("patch kept the tools it did not mention",
      patched["tools"] == ["echo"], r.text)
check("patch set the one field it did mention",
      patched["rules"] == ["Only say this."], r.text)

# A patch that mentions only one field must not blank out the others. The
# system_prompt default is "" rather than None, so this is the case that
# distinguishes exclude_unset from exclude_none.
r = client.patch("/api/agents/newcomer", json={"id": "newcomer", "tools": ["echo"]})
check("patch that omits the prompt is accepted", r.status_code == 200, r.text)
check("patch that omits the prompt leaves it alone",
      r.json()["agent"]["system_prompt"] == "Patched.", r.text)

r = client.delete("/api/agents/newcomer")
check("agent delete is 200", r.status_code == 200, r.text)
check("deleted agent is gone", client.get("/api/agents/newcomer").status_code == 404)

r = client.get("/api/agents/does-not-exist")
check("missing agent is 404", r.status_code == 404, r.text)

# -- validation: the reasons the API must refuse a write --------------------

section("validation")

cases = [
    ("unknown tool is rejected",
     "/api/agents/bad", {"id": "bad", "system_prompt": "x", "tools": ["nope"]},
     422, "nope"),
    ("self-handoff is rejected",
     "/api/agents/selfish", {"id": "selfish", "system_prompt": "x",
                             "tools": [], "handoffs": ["selfish"]},
     422, "itself"),
    ("handoff to a missing agent is rejected",
     "/api/agents/hopeful", {"id": "hopeful", "system_prompt": "x",
                             "tools": [], "handoffs": ["ghost"]},
     422, "ghost"),
    ("empty system prompt is rejected",
     "/api/agents/quiet", {"id": "quiet", "system_prompt": "   ",
                           "tools": [], "handoffs": []},
     422, "system prompt"),
    ("id with a slash is rejected by the schema",
     "/api/agents/bad", {"id": "bad/id", "system_prompt": "x"},
     422, "letters, digits"),
    ("body/path id mismatch is refused",
     "/api/agents/mismatch", {"id": "different", "system_prompt": "x"},
     400, "does not match"),
]
for name, path, payload, expect_status, expect_text in cases:
    r = client.put(path, json=payload)
    check(name, r.status_code == expect_status,
          f"got {r.status_code}: {r.text[:160]}")
    if expect_status == 422:
        problems = r.json().get("problems", [])
        check(f"{name} — message names the cause",
              any(expect_text in p for p in problems), str(problems)[:200])

r = client.get("/api/config/validate")
check("a valid config validates", r.json()["ok"] is True, r.text)

# A valid agent that hands off to a valid partner must be accepted, which is the
# case the "handoff tool exists" check exists for.
r = client.put("/api/agents/third", json={
    "id": "third", "system_prompt": "I defer.", "tools": [],
    "handoffs": ["primary"], "rules": [],
})
check("handoff to an existing agent is accepted", r.status_code == 200, r.text)
check("still valid after adding a handoff",
      client.get("/api/config/validate").json()["ok"] is True)
client.delete("/api/agents/third")

# -- tools ------------------------------------------------------------------

section("tools")

r = client.put("/api/tools/adder", json={
    "name": "adder", "description": "Add two numbers.",
    "args_schema": {"properties": {"a": {"type": "number", "description": "left"},
                                   "b": {"type": "number", "description": "right"}}},
    "position": 1,
})
check("tool create is 200", r.status_code == 200, r.text)
r = client.get("/api/tools/adder")
check("tool args_schema round-trips",
      set(r.json()["args_schema"]["properties"]) == {"a", "b"}, r.text)

r = client.put("/api/tools/quiet", json={"name": "quiet", "description": "  "})
check("tool with no description is rejected", r.status_code == 422, r.text)

r = client.delete("/api/tools/adder")
check("tool delete is 200", r.status_code == 200, r.text)

r = client.put("/api/tools/referenced", json={
    "name": "referenced", "description": "Still used by primary."})
check("tool create for a referenced name is 200", r.status_code == 200, r.text)
r = client.delete("/api/tools/referenced")
check("deleting an unreferenced tool works", r.status_code == 200, r.text)

# A tool an agent still lists must be removed from that agent, or the worker's
# next prompt build raises on a tool that no longer exists. Echo is re-added
# first so there is something to delete, then deleted, then checked.
r = client.put("/api/tools/echo", json={
    "name": "echo", "description": "Echo the given text back."})
check("re-adding a tool an agent lists is 200", r.status_code == 200, r.text)
r = client.delete("/api/tools/echo")
check("deleting a tool that an agent lists is 200", r.status_code == 200, r.text)
r = client.get("/api/agents/primary")
check("the agent that listed it no longer does",
      r.json().get("tools", []) == [], r.text)
check("config is still valid after the tool was deleted",
      client.get("/api/config/validate").json()["ok"] is True, r.text)
r = client.put("/api/tools/echo", json={
    "name": "echo", "description": "Echo the given text back.",
    "args_schema": {"properties": {"text": {"type": "string", "description": "x",
                                            "example": "hi"}}}})
check("re-adding the tool is 200", r.status_code == 200, r.text)
client.patch("/api/agents/primary", json={"id": "primary", "tools": ["echo"]})

# -- prompt preview ---------------------------------------------------------

section("prompt preview")

r = client.get("/api/agents/primary/prompt-preview", params={"user_text": "hi"})
check("prompt preview is 200", r.status_code == 200, r.text)
prompt = r.json().get("prompt", "")
check("preview includes the agent's own text", "You are the primary agent" in prompt
      or "Updated prompt" in prompt, prompt[:200])
check("preview includes the tool it was configured with", "echo" in prompt, prompt[:200])
check("preview includes the handoff tool", "handoff_to_helper" in prompt, prompt[:200])
check("preview is not a stub", len(prompt) > 80, f"len={len(prompt)}")

r = client.get("/api/agents/ghost/prompt-preview")
check("preview of a missing agent is 404", r.status_code == 404, r.text)

# -- squads -----------------------------------------------------------------

section("squads")

r = client.put("/api/squads/main", json={
    "id": "main", "name": "Main", "entry_agent": "primary",
    "edges": [{"from": "primary", "to": "helper"}],
})
check("squad create is 200", r.status_code == 200, r.text)
r = client.get("/api/squads")
check("squad lists back", len(r.json()["squads"]) == 1, r.text)
check("squad edges round-trip",
      r.json()["squads"][0]["edges"] == [{"from": "primary", "to": "helper"}], r.text)

r = client.put("/api/squads/orphan", json={
    "id": "orphan", "name": "Orphan", "entry_agent": "helper", "edges": []})
check("squad with an unreachable agent is rejected", r.status_code == 422, r.text)
check("the orphan message names the unreachable agent",
      any("primary" in p for p in r.json().get("problems", [])),
      str(r.json().get("problems"))[:200])

r = client.put("/api/squads/badentry", json={
    "id": "badentry", "entry_agent": "ghost", "edges": []})
check("squad with a missing entry agent is rejected", r.status_code == 422, r.text)

r = client.delete("/api/squads/main")
check("squad delete is 200", r.status_code == 200, r.text)

# -- health -----------------------------------------------------------------

section("health")

r = client.get("/health")
check("health is 200 even when services are down", r.status_code == 200, r.text)
body = r.json()
check("health has a status", body.get("status") in ("ok", "down", "unknown"), str(body)[:200])
check("health lists its services",
      set(body["services"]) == {"livekit", "ollama", "whisper", "piper", "mcp"},
      str(list(body.get("services", {}))))
for name, svc in body["services"].items():
    if svc["status"] == "down":
        check(f"{name} — a down service says how to fix it",
              bool(svc.get("fix")), str(svc)[:200])

r = client.get("/health", params={"track_sample_rate": 24000})
check("health accepts a track sample rate", r.status_code == 200, r.text)

# -- tokens -----------------------------------------------------------------

section("tokens")

os.environ["LIVEKIT_URL"] = "ws://localhost:7880"
os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "devsecret"
r = client.post("/api/calls/token", json={})
check("token mint is 200 when LiveKit is configured", r.status_code == 200, r.text)
body = r.json()
check("token is a JWT with three segments",
      body.get("token", "").count(".") == 2, str(body)[:120])
check("token echoes the url", body.get("url") == "ws://localhost:7880", str(body)[:120])
check("identity is generated, not empty", bool(body.get("identity")), str(body)[:120])

first = client.post("/api/calls/token", json={}).json()["identity"]
second = client.post("/api/calls/token", json={}).json()["identity"]
check("two tokens get different identities", first != second, f"{first} vs {second}")

r = client.post("/api/calls/token", json={"identity": "chosen-name"})
check("a client may still name itself", r.json()["identity"] == "chosen-name", r.text)

del os.environ["LIVEKIT_URL"]
r = client.post("/api/calls/token", json={})
check("token mint is 503 with LiveKit unconfigured", r.status_code == 503, r.text)
check("the 503 says which variables are missing",
      "LIVEKIT_URL" in r.json().get("detail", ""), r.text[:200])
os.environ["LIVEKIT_URL"] = "ws://localhost:7880"

# -- import / export --------------------------------------------------------

section("import and export")

real_agents = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "agent_platform", "agents")
real_tools = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "agent_platform", "tools.json")

# Import must read the real config, so the app is pointed back at it for exactly
# that one call and then put straight back. Export always writes to the temp dir.
#
# The repoint-and-restore is easy to get wrong in a way that rewrites the live
# config: export takes its destination from the same module-level path import
# reads from, so a test that points the app at the real directory and then calls
# export has just committed the test fixtures to the repository. The guard below
# makes that a loud failure instead.
def with_real_config(fn):
    app_module.AGENTS_DIR = real_agents
    app_module.TOOLS_FILE = real_tools
    try:
        return fn()
    finally:
        app_module.AGENTS_DIR = AGENTS_DIR
        app_module.TOOLS_FILE = os.path.join(AGENTS_DIR, "tools.json")


def assert_export_is_safe():
    for label, path in (("AGENTS_DIR", app_module.AGENTS_DIR),
                        ("TOOLS_FILE", app_module.TOOLS_FILE)):
        if os.path.abspath(path).startswith(os.path.abspath(TMP) + os.sep):
            continue
        raise AssertionError(
            f"refusing to export: {label} is {path}, which is outside the temp "
            f"dir {TMP}. Export would overwrite the real config."
        )


r = with_real_config(lambda: client.post("/api/config/import"))
check("import is 200", r.status_code == 200, r.text)
check("import found the real agents", r.json()["agents"] == 3, r.text)
check("import found the real tools", r.json()["tools"] == 6, r.text)
check("config validates after import",
      client.get("/api/config/validate").json()["ok"] is True)

# The real files must be identical after a database round trip, or the JSON the
# worker reads on the next start is not the config the UI just validated.
assert_export_is_safe()
r = client.post("/api/config/export")
check("export is 200", r.status_code == 200, r.text)
check("export wrote the agents",
      os.path.exists(os.path.join(AGENTS_DIR, "primary.json")), r.text)

import json  # noqa: E402

for name in ("primary", "trivia", "scheduler"):
    with open(os.path.join(real_agents, f"{name}.json")) as f:
        before = json.load(f)
    after_path = os.path.join(AGENTS_DIR, f"{name}.json")
    check(f"export wrote {name}.json", os.path.exists(after_path), after_path)
    if not os.path.exists(after_path):
        continue
    with open(after_path) as f:
        after = json.load(f)
    check(f"{name}.json survives a database round trip unchanged", before == after,
          json.dumps({"before": before, "after": after})[:400])

with open(real_tools) as f:
    before = json.load(f)
with open(os.path.join(AGENTS_DIR, "tools.json")) as f:
    after = json.load(f)
check("tools.json survives a database round trip unchanged", before == after,
      json.dumps({"before_keys": list(before), "after_keys": list(after)})[:400])

# The real config must be untouched by any of this. Checked rather than assumed,
# because the failure this guards against is silent: the files look plausible
# afterwards, just wrong.
check("the real config still has exactly three agents",
      sorted(f for f in os.listdir(real_agents) if f.endswith(".json"))
      == ["primary.json", "scheduler.json", "trivia.json"],
      str(sorted(os.listdir(real_agents))))
with open(real_tools) as f:
    check("the real tools.json still has exactly six tools",
          len(json.load(f)) == 6, str(len(json.load(open(real_tools)))))

# -- traces through the API -------------------------------------------------

section("traces")

ts = TraceStore(TRACE_DB)
ts.open_session("sess-1", "caller-1")
ts.log_event("sess-1", "call-1", "stt", {"duration_s": 1.0, "text": "hello"})
ts.log_event("sess-1", "call-1", "llm", {"duration_s": 2.0, "agent": "primary"})
ts.log_event("sess-1", "call-1", "tts", {"duration_s": 3.0, "text": "hi there"})
ts.log_event("sess-1", "call-2", "llm", {"error": "LLM unavailable"})
ts.set_first_agent("call-1", "primary")
ts.close()

r = client.get("/api/sessions")
check("sessions list is 200", r.status_code == 200, r.text)
check("the seeded session is listed",
      any(s["session_id"] == "sess-1" for s in r.json()["sessions"]), r.text)
sess = next(s for s in r.json()["sessions"] if s["session_id"] == "sess-1")
check("session latency totals the three stages",
      sess["latency"]["total"] == 6.0, str(sess["latency"]))
check("session counts its errors", sess["errors"] == 1, str(sess))
# The count is `n_calls` in both endpoints, and `calls` is the turn list in the
# detail view. It used to be `calls` for the count in the list view, which meant
# one key with two types depending on the endpoint.
check("the session list reports a turn count under n_calls",
      sess["n_calls"] == 2, str(sess))
check("the session list does not use `calls` for the count",
      isinstance(sess["n_calls"], int), str(sess))

r = client.get("/api/sessions/sess-1")
check("session detail is 200", r.status_code == 200, r.text)
detail = r.json()
check("session detail has both calls", len(detail["calls"]) == 2, r.text[:200])
check("session detail's n_calls agrees with the turn list",
      detail["n_calls"] == len(detail["calls"]), str(detail.get("n_calls")))
call1 = next(c for c in detail["calls"] if c["call_id"] == "call-1")
check("call detail has the user text", call1["user_text"] == "hello", r.text[:200])
check("call detail has the reply", call1["reply"] == "hi there", r.text[:200])
check("call detail totals latency", call1["latency"]["total"] == 6.0, r.text[:200])
call2 = next(c for c in detail["calls"] if c["call_id"] == "call-2")
check("a turn with an error is marked degraded", call2["degraded"] is True, r.text[:200])
check("the error text is returned", call2["errors"] == ["LLM unavailable"], r.text[:200])

r = client.get("/api/calls/call-1")
check("call detail by id is 200", r.status_code == 200, r.text)
check("call detail by id has its events",
      [e["event"] for e in r.json()["events"]] == ["stt", "llm", "tts"], r.text[:200])

r = client.get("/api/sessions/missing")
check("missing session is 404", r.status_code == 404, r.text)
r = client.get("/api/calls/missing")
check("missing call is 404", r.status_code == 404, r.text)

r = client.get("/api/sessions", params={"search": "caller-1"})
check("session search matches on identity",
      len(r.json()["sessions"]) == 1 and r.json()["total"] == 1, r.text[:200])
r = client.get("/api/sessions", params={"search": "nobody"})
check("session search with no match is empty", r.json()["total"] == 0, r.text[:200])

r = client.get("/api/timeline")
check("timeline is 200", r.status_code == 200, r.text)
check("timeline returns events", len(r.json()["events"]) >= 4, r.text[:200])

# -- config bootstrap --------------------------------------------------------

# The API seeds an empty database from the JSON files at startup, the same rule
# the worker follows. Without it a first run shows an empty Assistants screen
# next to three agent files on disk, and the only way out is knowing that
# POST /api/config/import exists.
#
# A fresh TestClient runs the lifespan, so simply entering the context manager is
# the test — no request needed to trigger it.
fresh_db = os.path.join(TMP, "bootstrap.db")
os.makedirs(os.path.join(TMP, "boot_agents"), exist_ok=True)
with open(os.path.join(TMP, "boot_agents", "a.json"), "w", encoding="utf-8") as fh:
    json.dump({"id": "a", "system_prompt": "be a seed", "tools": [],
               "handoffs": [], "rules": []}, fh)
with open(os.path.join(TMP, "boot_tools.json"), "w", encoding="utf-8") as fh:
    json.dump({}, fh)

saved = (app_module.CONFIG_DB, app_module.AGENTS_DIR, app_module.TOOLS_FILE)
app_module.CONFIG_DB = fresh_db
app_module.AGENTS_DIR = os.path.join(TMP, "boot_agents")
app_module.TOOLS_FILE = os.path.join(TMP, "boot_tools.json")
try:
    with TestClient(app_module.app) as boot:
        listed = boot.get("/api/agents").json()["agents"]
        check("an empty database is seeded from JSON at startup",
              [a["id"] for a in listed] == ["a"], str(listed))
        check("the seeded agent keeps its prompt",
              listed[0]["system_prompt"] == "be a seed", str(listed))

    # A second start must not re-import: the database is the source of truth once
    # it exists, or every restart would undo deletions made in the UI.
    with TestClient(app_module.app) as boot:
        after = boot.get("/api/agents").json()["agents"]
        check("a populated database is not re-seeded on restart",
              len(after) == 1, str(after))

    # Deleting through the API and restarting must stick.
    boot_client = TestClient(app_module.app)
    boot_client.delete("/api/agents/a")
    with TestClient(app_module.app) as boot:
        final = boot.get("/api/agents").json()["agents"]
        check("an agent deleted in the UI stays deleted across a restart",
              final == [], str(final))
finally:
    app_module.CONFIG_DB, app_module.AGENTS_DIR, app_module.TOOLS_FILE = saved

# -- evals ------------------------------------------------------------------

# The eval endpoints shell out to run_eval.py and score_eval.py, which publish
# to LiveKit and take minutes. These tests cover the part that decides whether
# the button is even pressable — the preflight and the job bookkeeping — by
# pointing the module at a directory with no test audio in it, so a run can
# never actually start.

r = client.get("/api/evals")
check("evals is 200", r.status_code == 200, r.text)
check("evals returns a report slot", "report" in r.json(), r.text[:200])
check("evals returns preflight problems", isinstance(r.json()["preflight"], list), r.text[:200])

# Preflight is stubbed to fail here rather than relying on the repo happening to
# have no test audio. A test that passes because of missing files is a test that
# quietly stops testing anything once someone runs generate_test_audio.py — and
# the failure mode is a two-minute subprocess launch inside the suite, which
# looks like a hang.
import control_plane.evals as evals_module  # noqa: E402

original_preflight = evals_module.preflight
original_spawn = evals_module._spawn
# The report cache path is a module global pointing into the real .run/. Without
# redirecting it, a passing test run leaves a fabricated 1/1 report sitting where
# the Evals screen will show it as the last real result.
original_report = evals_module.LAST_REPORT
evals_module.LAST_REPORT = os.path.join(TMP, "last_report.json")

evals_module.preflight = lambda: ["No test audio found. Generate it with: python3 generate_test_audio.py"]

r = client.post("/api/evals", json={"users": 1, "rotate": 0})
check("a run missing test audio is refused", r.status_code == 409, r.text[:200])
check("the refusal says what to fix",
      "generate_test_audio" in r.text.lower(), r.text[:200])

r = client.post("/api/evals", json={"users": 99, "rotate": 0})
check("too many callers is rejected", r.status_code == 422, r.text[:200])

r = client.get("/api/evals/status")
check("eval status is 200", r.status_code == 200, r.text)
check("eval status reports a state", "state" in r.json(), r.text[:200])

# A start with everything present returns 202 and a job id rather than blocking:
# a run takes minutes, and an HTTP request that waits for it would time out in a
# way indistinguishable from a hung agent.
#
# `_spawn` is stubbed as well as preflight. Clearing preflight alone lets the job
# through to a real `subprocess.Popen` of run_eval.py, which joins LiveKit and
# takes minutes — the bookkeeping below is what this tests, not the harness.
import subprocess  # noqa: E402

evals_module.preflight = lambda: []
calls_made = []


def fake_spawn(cmd, job_id):
    calls_made.append(cmd)
    # A complete, passing report — enough for the job to reach "done" through
    # the real parsing and finishing path.
    return subprocess.CompletedProcess(cmd, 0, json.dumps({
        "window": None, "n_sessions": 1, "n_calls": 1, "passed": 1, "total": 1,
        "sessions": [], "latency_overall": {},
    }), "")

evals_module._spawn = fake_spawn


def wait_for(predicate, timeout=10.0):
    """Poll a job state without sleeping the whole suite if it never arrives.

    The job runs on its own thread; the test has to wait for it, and a bare
    `time.sleep` would either be too short on a loaded machine or waste seconds
    on each of the transitions below.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = evals_module.current_job()
        if current and predicate(current):
            return current
        time.sleep(0.02)
    return evals_module.current_job()


try:
    r = client.post("/api/evals", json={"users": 1, "rotate": 0})
    check("a startable run returns 202", r.status_code == 202, r.text[:200])
    job_id = r.json()["id"]
    check("the job has an id", bool(job_id), r.text[:200])
    check("the job starts out running", r.json()["state"] == "running", r.text[:200])
    check("the job carries the log list", isinstance(r.json().get("log"), list), r.text[:200])

    finished = wait_for(lambda j: j["state"] != "running")
    check("a successful run finishes as done", finished["state"] == "done", str(finished)[:200])
    check("the report is attached to the job", finished["report"] is not None, str(finished)[:200])
    check("the report is cached for the next page load",
          client.get("/api/evals").json()["report"] is not None)
    check("the cached report went to the temp directory, not the real .run/",
          os.path.exists(evals_module.LAST_REPORT), evals_module.LAST_REPORT)
    check("the runner is invoked with the requested caller count",
          any("run_eval.py" in " ".join(c) and "--users" in c for c in calls_made),
          str(calls_made))
    check("the scorer is invoked too",
          any("score_eval.py" in " ".join(c) for c in calls_made), str(calls_made))

    # A second POST while one is in flight must return the same job rather than
    # starting a rival: two runs share a room and the scorer cannot attribute
    # cases across them.
    #
    # This needs a harness that actually blocks. The stub above returns
    # instantly, so by the time a second POST arrives the first job is already
    # done and the guard has nothing to catch — which is how the first version
    # of this test passed a broken implementation.
    evals_module._job = None
    release = threading.Event()
    evals_module._spawn = lambda cmd, job_id: (
        release.wait(20),
        subprocess.CompletedProcess(cmd, 0, json.dumps({
            "n_sessions": 1, "n_calls": 1, "passed": 1, "total": 1,
            "sessions": [], "latency_overall": {}}), ""),
    )[1]

    first = client.post("/api/evals", json={"users": 1, "rotate": 0}).json()["id"]
    second = client.post("/api/evals", json={"users": 1, "rotate": 0}).json()["id"]
    check("a second run while one is in flight returns the same job",
          second == first, f"{second} != {first}")
    check("only one job is running",
          len([c for c in calls_made if "run_eval.py" in " ".join(c)]) >= 1,
          str(calls_made))
    release.set()
    wait_for(lambda j: j["state"] != "running")

    # A report with zero scorable cases is a failure, not a 0/0 pass. A run that
    # produced nothing means the worker never joined, and rendering that as a
    # clean score is the one outcome that would send someone away thinking the
    # change they were testing worked.
    evals_module._job = None
    evals_module._spawn = lambda cmd, job_id: subprocess.CompletedProcess(
        cmd, 0,
        json.dumps({"n_sessions": 0, "n_calls": 0, "passed": 0, "total": 0,
                    "sessions": [], "latency_overall": {}}),
        "",
    )
    client.post("/api/evals", json={"users": 1, "rotate": 0})
    empty = wait_for(lambda j: j["state"] != "running")
    check("a run that scored nothing is a failure", empty["state"] == "failed", str(empty)[:200])
    check("the failure says why", "no scorable calls" in (empty.get("error") or ""),
          str(empty.get("error")))

    # A non-zero exit from either tool must surface, not render as an empty
    # score.
    evals_module._job = None
    evals_module._spawn = lambda cmd, job_id: subprocess.CompletedProcess(cmd, 1, "", "")
    client.post("/api/evals", json={"users": 1, "rotate": 0})
    failed = wait_for(lambda j: j["state"] != "running")
    check("a non-zero exit is reported as a failure", failed["state"] == "failed",
          str(failed)[:200])
    check("the exit code is in the message", "exited 1" in (failed.get("error") or ""),
          str(failed.get("error")))
finally:
    evals_module.preflight = original_preflight
    evals_module._spawn = original_spawn
    evals_module.LAST_REPORT = original_report
    evals_module._job = None

# -- summary ----------------------------------------------------------------

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("\nFailures:")
    for name, detail in FAILED:
        print(f"  - {name}: {detail}")
    sys.exit(1)
