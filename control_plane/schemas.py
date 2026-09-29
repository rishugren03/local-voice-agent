"""Request and response models for the control plane.

Two rules shape this file.

Everything the UI sends is a separate model from what the store keeps, so an
editor form can post a partial (change one field) without inventing a schema that
says "you must send all of them". `AgentWrite` is therefore explicit about
optionality, and a `None` field means "not being changed" — never "clear it",
because a half-filled form that silently wipes an agent's tools is the worst
outcome an editor can have.

Every model validates itself before the store sees it, so an obviously bad
payload (an id with a slash, an empty prompt) comes back as a 422 naming the
field rather than as a 500 from deep inside SQLite.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AgentWrite(BaseModel):
    """An agent as the editor sends it.

    Optional throughout, for the partial-update case: PUT replaces, PATCH
    merges. `tools`, `handoffs` and `rules` are only touched when present.
    """
    id: str = Field(..., min_length=1, max_length=64)
    system_prompt: str = ""
    tools: Optional[List[str]] = None
    handoffs: Optional[List[str]] = None
    rules: Optional[List[str]] = None
    llm_model: Optional[str] = None
    tts_voice: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _id_is_path_safe(cls, v):
        # The id is a filename on export and a URL segment on GET, so anything
        # with a slash, dot or space produces a 404 or a file outside the config
        # directory. Caught here so it is a field error at write time.
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "id may only contain letters, digits, '-' and '_' "
                "(it becomes a filename and a URL)"
            )
        return v

    @field_validator("rules", "tools", "handoffs")
    @classmethod
    def _no_blanks(cls, v):
        if v is None:
            return v
        if any((not isinstance(item, str) or not item.strip()) for item in v):
            raise ValueError("entries must be non-empty strings")
        return v


class ToolWrite(BaseModel):
    """A tool definition: a description plus a JSON-schema-shaped argument map."""
    name: str = Field(..., min_length=1, max_length=64)
    description: str = ""
    args_schema: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    position: int = 0

    @field_validator("name")
    @classmethod
    def _name_is_path_safe(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("name may only contain letters, digits, '-' and '_'")
        return v


class SquadWrite(BaseModel):
    """A squad: an entry agent plus the handoff edges the graph editor drew."""
    id: str = Field(..., min_length=1, max_length=64)
    name: str = ""
    entry_agent: str = ""
    edges: List[Dict[str, str]] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_is_path_safe(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("id may only contain letters, digits, '-' and '_'")
        return v

    @field_validator("edges")
    @classmethod
    def _edges_have_endpoints(cls, v):
        for i, edge in enumerate(v):
            if not edge.get("from") or not edge.get("to"):
                raise ValueError(f"edge {i} needs both 'from' and 'to'")
        return v


class ValidationResponse(BaseModel):
    ok: bool
    problems: List[str] = Field(default_factory=list)
    config_version: int


class ConfigVersion(BaseModel):
    """What the worker polls.

    Cheap on purpose: the worker calls this at the start of every session, and
    the version is the only thing it needs to decide whether to reload. Returning
    the whole config here would make the most frequent request in the system also
    the most expensive one.
    """
    config_version: int
    agents: int
    tools: int
    squads: int


class SessionSummary(BaseModel):
    session_id: str
    participant_identity: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    turns_served: Optional[int] = None
    barge_ins: Optional[int] = None
    final_agent: Optional[str] = None
    # `n_calls`, not `calls`: /api/sessions/{id} returns `calls` as the list of
    # turns, and this response_model silently drops any field it does not
    # declare — so with `calls: int` here, the count appeared in the list view
    # and the list of turns in the detail view, under the same key.
    n_calls: int = 0
    errors: int = 0
    latency: Dict[str, Optional[float]]


class SessionList(BaseModel):
    sessions: List[SessionSummary]
    total: int
    limit: int
    offset: int


class CallSummary(BaseModel):
    call_id: str
    started_at: Optional[str] = None
    first_agent: Optional[str] = None
    user_text: Optional[str] = None
    reply: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    degraded: bool = False
    latency: Dict[str, Optional[float]]
    transitions: List[Dict[str, Any]] = Field(default_factory=list)


class CallDetail(CallSummary):
    session_id: Optional[str] = None
    participant_identity: Optional[str] = None
    events: List[Dict[str, Any]] = Field(default_factory=list)


class SessionDetail(SessionSummary):
    """One session with its turns.

    Declared after CallSummary because it contains a list of them, and extends
    the summary rather than redefining the fields, so the two views cannot drift
    on anything but `calls` — the one field that genuinely differs between them,
    an int in the list view and a list of turns here.
    """

    calls: List[CallSummary] = Field(default_factory=list)
    transitions: List[Dict[str, Any]] = Field(default_factory=list)


class TokenRequest(BaseModel):
    """What the browser needs to join a room itself.

    The client mints nothing: the LiveKit API secret stays on the server, which
    is the whole reason the browser is not a security problem in v1. The
    participant identity is generated server-side for the same reason — a client
    that picked its own could collide with another browser tab, and the session
    id is derived from it.
    """
    identity: Optional[str] = None
    name: Optional[str] = None
    room: Optional[str] = None


class TokenResponse(BaseModel):
    token: str
    url: str
    room: str
    identity: str
    expires_in: int


class HealthResponse(BaseModel):
    status: str
    down: List[str] = Field(default_factory=list)
    unknown: List[str] = Field(default_factory=list)
    services: Dict[str, Dict[str, Any]]


class ImportResponse(BaseModel):
    agents: int
    tools: int
    config_version: int


class ExportResponse(BaseModel):
    agents_dir: str
    tools_file: str
    config_version: int


class EvalRequest(BaseModel):
    """Which eval run to start.

    Defaults are one user, no rotation: the single-caller suite. `users` and
    `rotate` are the stress knobs — two concurrent callers on one CPU, started
    at different points in the script — which is what exercises the scorer's
    per-session attribution.

    Bounded at 4 users because the point is to overload one machine, and a run
    that just times out teaches nothing.
    """

    users: int = Field(default=1, ge=1, le=4)
    rotate: int = Field(default=0, ge=0, le=20)
    cases: Optional[str] = Field(default=None, max_length=500)


class EvalJob(BaseModel):
    id: str
    state: str  # running | done | failed
    started_at: float
    finished_at: Optional[float] = None
    users: int
    rotate: int
    cases: Optional[str] = None
    stage: str
    log: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
