"""FastAPI control plane for the voice-agent platform.

The rule this file is built around: the browser never touches the database, the
LiveKit secret, or the model files. It reads and writes config through these
endpoints, and the worker reads the same config from the same SQLite file. That
is what makes "save in the UI, hear it on the next call" true rather than
aspirational — one store, two readers, a version number in between.

No auth, bound to localhost. That is a deliberate v1 scope, stated here because
it is the kind of thing that must be visible in the source rather than assumed
by whoever runs it: anything that can reach this port can mint a LiveKit token
and rewrite any agent prompt. A `--host 0.0.0.0` deployment needs real auth
first.

Validation is shared with the worker (agent_platform/validation.py) so a config
cannot be accepted here and rejected there.
"""

import os
import secrets
import sys
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_platform.config_store import ConfigStore  # noqa: E402
from agent_platform.orchestrator import AgentRegistry  # noqa: E402
from agent_platform import validation  # noqa: E402
from agent_platform.trace_store import TraceStore  # noqa: E402
from control_plane import health  # noqa: E402
from control_plane import schemas  # noqa: E402
from control_plane import evals  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS_DIR = os.path.join(ROOT, "agent_platform", "agents")
TOOLS_FILE = os.path.join(ROOT, "agent_platform", "tools.json")
CONFIG_DB = os.getenv("CONFIG_DB", os.path.join(ROOT, "call_trace.db"))
TRACE_DB = os.getenv("TRACE_DB", os.path.join(ROOT, "call_trace.db"))
# Must match transcribe_test.py's default exactly. These are the same room name
# read by two processes; if they disagree and ROOM_NAME is unset, the browser
# joins a room the agent never enters and the call reports connected with no
# audio, which is a miserable thing to debug from the browser side.
ROOM_NAME = os.getenv("ROOM_NAME", "test-room")
TOKEN_TTL_S = int(os.getenv("TOKEN_TTL_S", "3600"))


def get_store():
    """One store per request.

    A fresh ConfigStore per request rather than a module-level singleton: the
    store holds a sqlite connection, and a connection opened before an event loop
    forks or before a reload would be shared across threads. The cost is opening
    a WAL database, which is microseconds.
    """
    store = ConfigStore(CONFIG_DB)
    try:
        yield store
    finally:
        store.close()


def get_trace_store():
    store = TraceStore(TRACE_DB)
    try:
        yield store
    finally:
        store.close()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Seed an empty config database from the JSON files, once per database.

    The worker already does this (transcribe_test.py) so that a fresh checkout can
    answer calls. The API did not, which meant a first run — start the API, open
    the UI — showed an empty Assistants screen next to three agent files on disk,
    and the only way out was to know that POST /api/config/import existed.

    The rule lives in ConfigStore.seed_from_json_if_empty() so this and the
    worker cannot disagree about when a database counts as empty.
    """
    try:
        store = ConfigStore(CONFIG_DB)
        try:
            if store.seed_from_json_if_empty(AGENTS_DIR, TOOLS_FILE):
                print(
                    f"[CONFIG] Empty database; imported "
                    f"{os.path.basename(AGENTS_DIR)}/*.json and "
                    f"{os.path.basename(TOOLS_FILE)}.",
                    file=sys.stderr,
                )
        finally:
            store.close()
    except Exception as e:  # noqa: BLE001
        # Never let seeding stop the API from starting. A control plane that
        # refuses to boot over a config problem cannot be used to fix the config
        # problem; the health check and /api/config/validate report it instead.
        print(f"[WARN] Config seed failed ({e}).", file=sys.stderr)
    yield


app = FastAPI(
    title="Voice Agent Control Plane",
    version="1.0.0",
    description="Config, traces and tokens for the local voice agent.",
    lifespan=_lifespan,
)

# The dev UI runs on a different port, so the browser needs CORS. Restricted to
# localhost on purpose: this API has no auth, and a wildcard origin would let any
# page the user visits mint tokens against it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(validation.ConfigValidationError)
async def _config_error_handler(request, exc):
    """Config problems are a 422 with the messages, not a 500.

    The status matters: 422 means "fix this and resend", which is what the editor
    does. A 500 would be rendered as a server fault and the user would retry
    without changing anything.
    """
    return JSONResponse(status_code=422, content={
        "detail": "Invalid config",
        "problems": exc.problems,
        "config_version": _current_version(),
    })


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(request, exc):
    """Reshape pydantic's error list into the same {"problems": [...]} contract
    the domain errors use.

    Without this the UI would need two error shapes: one for "the config is
    wrong" and one for "the JSON body is malformed", and a form that only renders
    the first silently accepts the second. The pydantic location path is joined
    into a dotted field name so the message points at the input the user actually
    touched, not at the body model.
    """
    problems = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        message = err.get("msg", "is invalid")
        problems.append(f"{location}: {message}" if location else message)
    return JSONResponse(status_code=422, content={
        "detail": "Invalid request",
        "problems": problems,
        "config_version": _current_version(),
    })


def _current_version():
    with ConfigStore(CONFIG_DB) as store:
        return store.config_version()


def _registry_from(store):
    """An AgentRegistry backed by the store, used for cross-checks and previews.

    Built per request rather than cached: it is the object whose job is to make
    the JSON config and the database config look identical, and a cached one
    would be a second copy of the state this API is meant to be the only writer
    of.
    """
    return AgentRegistry(
        default_llm_model=os.getenv("DEFAULT_LLM_MODEL", "phi4-mini"),
        default_tts_voice=os.getenv("PIPER_MODEL", ""),
        config_store=store,
    )


# -- health -----------------------------------------------------------------

@app.get("/health", response_model=schemas.HealthResponse)
def health_check(track_sample_rate: int = Query(
        default=0, ge=0,
        description="LiveKit track rate; enables the voice sample-rate check.")):
    """Every service, with a fix for anything down.

    Deliberately never 503s on a down service. A monitoring probe that gets a 503
    cannot tell 'the API is broken' from 'Ollama is not running', and those need
    different responses; the body carries the verdict instead.
    """
    result = health.check_all(track_sample_rate or None)
    return result


# -- version ----------------------------------------------------------------

@app.get("/api/config/version", response_model=schemas.ConfigVersion)
def get_version(store: ConfigStore = Depends(get_store)):
    """What the worker polls at the start of each session.

    The cheapest endpoint here on purpose. The worker calls it once per session,
    so it is the highest-frequency request in the system, and it needs nothing
    but the version number.
    """
    return {
        "config_version": store.config_version(),
        "agents": len(store.list_agents()),
        "tools": len(store.list_tools()),
        "squads": len(store.list_squads()),
    }


# -- agents -----------------------------------------------------------------

@app.get("/api/agents")
def list_agents(store: ConfigStore = Depends(get_store)):
    return {"agents": store.list_agents(), "config_version": store.config_version()}


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str, store: ConfigStore = Depends(get_store)):
    agent = store.get_agent(agent_id)
    if agent is None:
        raise HTTPException(404, f"No agent named '{agent_id}'")
    return agent


@app.put("/api/agents/{agent_id}", status_code=200)
def save_agent(agent_id: str, body: schemas.AgentWrite,
               store: ConfigStore = Depends(get_store)):
    """Create or replace an agent.

    PUT and not POST because the id is in the path and identifies the row: the
    same request twice must be the same agent, not two agents.

    Validated against the post-write world, so an agent may hand off to another
    agent created in the same batch, and the handoff tool is accounted for before
    the worker ever builds a prompt.
    """
    if body.id != agent_id:
        raise HTTPException(400, f"body id '{body.id}' does not match path '{agent_id}'")

    payload = body.model_dump(exclude_none=True)
    payload["id"] = agent_id
    if "system_prompt" not in payload:
        payload["system_prompt"] = ""

    existing = {a["id"]: a for a in store.list_agents()}
    tools = {name for name, _ in store.list_tools_with_names()}
    problems = validation.validate_agent_payload(payload, existing, tools)
    if problems:
        raise validation.ConfigValidationError(problems)

    store.save_agent(payload)
    return {"agent": store.get_agent(agent_id), "config_version": store.config_version()}


@app.patch("/api/agents/{agent_id}")
def patch_agent(agent_id: str, body: schemas.AgentWrite,
                store: ConfigStore = Depends(get_store)):
    """Partial update: only the fields present in the body change.

    This is what the editor uses for inline tweaks, so that changing one rule
    does not require the form to have round-tripped every other field correctly
    first. Absent means unchanged; that is the reason every field on AgentWrite
    except the id is optional.

    exclude_unset, not exclude_none, and the difference is the whole point of
    this method. `system_prompt` defaults to "" rather than None, so excluding
    only the Nones would merge that empty default over the stored prompt and
    silently wipe it on every patch that did not mention it — an agent would
    look fine until the next call got a prompt with no task in it.
    """
    current = store.get_agent(agent_id)
    if current is None:
        raise HTTPException(404, f"No agent named '{agent_id}'")

    updates = body.model_dump(exclude_unset=True)
    updates.pop("id", None)
    merged = dict(current)
    merged.update(updates)
    merged["id"] = agent_id

    existing = {a["id"]: a for a in store.list_agents()}
    problems = validation.validate_agent_payload(
        merged, existing, {name for name, _ in store.list_tools_with_names()})
    if problems:
        raise validation.ConfigValidationError(problems)

    store.save_agent(merged)
    return {"agent": store.get_agent(agent_id), "config_version": store.config_version()}


@app.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: str, store: ConfigStore = Depends(get_store)):
    if store.get_agent(agent_id) is None:
        raise HTTPException(404, f"No agent named '{agent_id}'")
    store.delete_agent(agent_id)
    return {"deleted": agent_id, "config_version": store.config_version()}


@app.get("/api/agents/{agent_id}/prompt-preview")
def prompt_preview(agent_id: str, user_text: str = Query(default="Hello",
                                                        max_length=2000),
                   context_note: str = Query(default="", max_length=500),
                   store: ConfigStore = Depends(get_store)):
    """The exact system prompt the worker would build, for this text.

    Built by the worker's own build_prompt rather than a copy in the control
    plane. A preview rendered by different code is a preview that lies — the whole
    reason this endpoint exists is so the editor can see what the model sees, and
    it cannot do that with a reimplementation.
    """
    try:
        prompt = _registry_from(store).build_prompt(
            agent_id, user_text, context_note)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"agent_id": agent_id, "user_text": user_text, "prompt": prompt}


# -- tools ------------------------------------------------------------------

@app.get("/api/tools")
def list_tools(store: ConfigStore = Depends(get_store)):
    # The store keeps a tool's name as its key and out of the spec, so that
    # export reproduces the keyed tools.json shape byte for byte. The UI has no
    # such constraint, so the name is folded back in here rather than changing
    # the stored shape.
    return {
        "tools": [{"name": name, **spec}
                  for name, spec in store.list_tools_with_names()],
        "config_version": store.config_version(),
    }


@app.get("/api/tools/{tool_name}")
def get_tool(tool_name: str, store: ConfigStore = Depends(get_store)):
    spec = store.get_tool(tool_name)
    if spec is None:
        raise HTTPException(404, f"No tool named '{tool_name}'")
    return {"name": tool_name, **spec}


@app.put("/api/tools/{tool_name}")
def save_tool(tool_name: str, body: schemas.ToolWrite,
              store: ConfigStore = Depends(get_store)):
    if body.name != tool_name:
        raise HTTPException(400, f"body name '{body.name}' does not match path '{tool_name}'")
    problems = validation.validate_tool(tool_name, {
        "description": body.description,
        "args_schema": body.args_schema,
    })
    if problems:
        raise validation.ConfigValidationError(problems)
    store.save_tool(tool_name, {
        "description": body.description,
        "args_schema": body.args_schema,
        "enabled": body.enabled,
    }, position=body.position)
    return {"tool": {"name": tool_name, **(store.get_tool(tool_name) or {})},
            "config_version": store.config_version()}


@app.delete("/api/tools/{tool_name}")
def delete_tool(tool_name: str, store: ConfigStore = Depends(get_store)):
    """Removes a tool, detaching it from every agent that referenced it.

    Detached rather than refused: an agent listing a deleted tool fails to build
    a prompt, which breaks the whole squad, not just the tool. An explicit
    `enabled` flag exists for "stop offering this but keep the config", which is
    the reversible choice; delete is for when the tool is genuinely gone.
    """
    if store.get_tool(tool_name) is None:
        raise HTTPException(404, f"No tool named '{tool_name}'")
    store.delete_tool(tool_name)
    return {"deleted": tool_name, "config_version": store.config_version()}


# -- squads -----------------------------------------------------------------

@app.get("/api/squads")
def list_squads(store: ConfigStore = Depends(get_store)):
    return {"squads": store.list_squads(), "config_version": store.config_version()}


@app.put("/api/squads/{squad_id}")
def save_squad(squad_id: str, body: schemas.SquadWrite,
               store: ConfigStore = Depends(get_store)):
    if body.id != squad_id:
        raise HTTPException(400, f"body id '{body.id}' does not match path '{squad_id}'")
    agent_ids = {a["id"] for a in store.list_agents()}
    problems = validation.validate_squad(body.model_dump(), agent_ids)
    if problems:
        raise validation.ConfigValidationError(problems)
    store.save_squad(body.model_dump())
    return {"squad": store.get_squad(squad_id), "config_version": store.config_version()}


@app.delete("/api/squads/{squad_id}")
def delete_squad(squad_id: str, store: ConfigStore = Depends(get_store)):
    if store.get_squad(squad_id) is None:
        raise HTTPException(404, f"No squad named '{squad_id}'")
    store.delete_squad(squad_id)
    return {"deleted": squad_id, "config_version": store.config_version()}


# -- validation -------------------------------------------------------------

@app.get("/api/config/validate")
def validate_config(store: ConfigStore = Depends(get_store)):
    """Check the stored config without changing it.

    The worker's own load-time check, so 'valid' means the same thing here and in
    the worker. Run on read as well as on write because config can be changed
    from two places at once — this API and a hand-edited JSON file — and a write
    that passed validation is not a guarantee the current config passes.
    """
    agents = {a["id"]: a for a in store.list_agents()}
    tools = dict(store.list_tools_with_names())
    problems = validation.validate_config_snapshot(agents, tools)
    for squad in store.list_squads():
        problems.extend(validation.validate_squad(squad, set(agents)))
    return {"ok": not problems, "problems": problems,
            "config_version": store.config_version()}


# -- import / export --------------------------------------------------------

@app.post("/api/config/import", response_model=schemas.ImportResponse)
def import_config(store: ConfigStore = Depends(get_store)):
    """Load the on-disk JSON config into SQLite.

    One-way and explicit. Import overwrites the database from the files, so it is
    destructive to anything written in the UI; the endpoint says so in its
    summary rather than making the user learn it by losing work.
    """
    result = store.import_from_json(AGENTS_DIR, TOOLS_FILE)
    # The store's result calls it "version"; the API contract calls it
    # "config_version" everywhere so the UI has one field name to watch for
    # changes. Renamed here rather than in the store, because "version" is
    # ambiguous in a store that also versions agents individually.
    return {
        "agents": result["agents"],
        "tools": result["tools"],
        "config_version": store.config_version(),
    }


@app.post("/api/config/export", response_model=schemas.ExportResponse)
def export_config(store: ConfigStore = Depends(get_store)):
    """Write the current config back out to the JSON files.

    This is what makes the existing files, config_cli.py and the worker keep
    working unchanged: the database is the source of truth during a run, and the
    files are the durable, diffable, git-committable copy of it.
    """
    store.export_to_json(AGENTS_DIR, TOOLS_FILE)
    return {"agents_dir": AGENTS_DIR, "tools_file": TOOLS_FILE,
            "config_version": store.config_version()}


# -- traces -----------------------------------------------------------------

@app.get("/api/sessions", response_model=schemas.SessionList)
def list_sessions(limit: int = Query(50, ge=1, le=500),
                  offset: int = Query(0, ge=0),
                  search: str = Query("", max_length=200),
                  traces: TraceStore = Depends(get_trace_store)):
    return {
        "sessions": traces.list_sessions(limit=limit, offset=offset, search=search or None),
        "total": traces.count_sessions(search=search or None),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/sessions/{session_id}", response_model=schemas.SessionDetail)
def get_session(session_id: str, traces: TraceStore = Depends(get_trace_store)):
    detail = traces.get_session(session_id)
    if detail is None:
        raise HTTPException(404, f"No session '{session_id}'")
    return detail


@app.get("/api/calls/{call_id}", response_model=schemas.CallDetail)
def get_call(call_id: str, traces: TraceStore = Depends(get_trace_store)):
    detail = traces.get_call(call_id)
    if detail is None:
        raise HTTPException(404, f"No call '{call_id}'")
    return detail


@app.get("/api/timeline")
def timeline(since_ts: float = Query(0.0), session_id: str = Query("", max_length=200),
             limit: int = Query(200, ge=1, le=500),
             traces: TraceStore = Depends(get_trace_store)):
    return {"events": traces.get_timeline(
        since_ts=since_ts or None, session_id=session_id or None, limit=limit)}


# -- tokens -----------------------------------------------------------------

@app.post("/api/calls/token", response_model=schemas.TokenResponse)
def mint_token(body: schemas.TokenRequest):
    """Mint a short-lived LiveKit token for the browser.

    The browser joins the room itself rather than streaming through the control
    plane: audio does not belong in an HTTP request, and a media relay through
    this process would put the API in the critical path of every call. The secret
    stays here, so the browser can have full room access without ever holding
    credentials.

    The identity defaults to a random value, not to something the client chose.
    A client-supplied identity collides across tabs, and session_id is derived
    from it — two tabs sharing an identity would share session state and one
    caller's handoff would change the other caller's active agent.
    """
    try:
        from livekit import api as lk_api
    except ImportError as e:
        raise HTTPException(503, f"livekit client library is not installed: {e}")

    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    url = os.getenv("LIVEKIT_URL")
    if not (key and secret and url):
        raise HTTPException(
            503,
            "LiveKit is not configured: set LIVEKIT_URL, LIVEKIT_API_KEY and "
            "LIVEKIT_API_SECRET (start livekit-server --dev, or use docker compose)",
        )

    identity = body.identity or f"browser-{secrets.token_hex(4)}"
    room = body.room or ROOM_NAME
    token = (lk_api.AccessToken(key, secret)
             .with_identity(identity)
             .with_name(body.name or "Browser")
             .with_grants(lk_api.VideoGrants(room_join=True, room=room))
             # A timedelta, not seconds: the SDK adds this to a datetime, and
             # passing an int raises a TypeError at mint time — which is a 500 on
             # the exact request the Talk screen makes to start a call.
             .with_ttl(timedelta(seconds=TOKEN_TTL_S))
             .to_jwt())
    return {"token": token, "url": url, "room": room, "identity": identity,
            "expires_in": TOKEN_TTL_S}


# -- evals ------------------------------------------------------------------

@app.get("/api/evals")
def get_evals():
    """The most recent eval report, plus the state of any run in flight.

    `preflight` is returned even when there is no report, so the Evals screen can
    explain why the run button is disabled instead of leaving it greyed out for
    no visible reason.
    """
    return {"report": evals.last_report(), "job": evals.current_job(),
            "preflight": evals.preflight()}


@app.post("/api/evals")
def start_eval(body: schemas.EvalRequest):
    problems = evals.preflight()
    if problems:
        raise HTTPException(409, "; ".join(problems))
    job = evals.start(users=body.users, rotate=body.rotate, cases=body.cases)
    # 202 even when a run was already going: the request was accepted, and the
    # job it refers to is the one now in flight. A 409 here would be wrong —
    # double-clicking Run is not a client error.
    return JSONResponse(status_code=202, content=job)


@app.get("/api/evals/status")
def eval_status():
    return evals.current_job() or {"state": "idle"}


def run():
    """Serve the API, bound to localhost unless told otherwise.

    Host is read from the environment rather than hardcoded so the container
    image can bind 0.0.0.0 inside its own network without changing code — while
    the default stays 127.0.0.1, because this API has no auth.
    """
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    run()
