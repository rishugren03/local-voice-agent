"""SQLite-backed config: agents, tools, squads, and a config version counter.

Why this exists
---------------
Agent and tool configuration used to live only in JSON files, which is fine right
up until something else needs to read it. A control plane cannot answer "list the
agents" from a directory listing, a browser cannot edit a file, and two writers
have no way to coordinate. So the database becomes the source of truth and the
JSON files become an import/export format — the same relationship the trace
already has with call_trace.jsonl.

The version counter
-------------------
Every config write bumps a single integer. The worker compares the integer it
loaded against on each new session; when it changes, the config is re-read. That
is what makes "save an agent in the UI, it applies to the next call" work without
a restart, and it is deliberately coarse: reload at session start, never mid-call.
A config swap in the middle of a turn would change the tool list underneath a
prompt that was already built, and the failure would be a tool call answered by
the wrong agent.

Sharing the trace database
--------------------------
By default this writes to the same file as the trace store, because "one
database is the source of truth" is worth more than the tidiness of two files.
SQLite in WAL mode supports many concurrent readers alongside one writer, and the
trace store already holds its own connection to it.
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

DEFAULT_CONFIG_DB = os.getenv("CONFIG_DB", "call_trace.db")

# The description and argument schema given to an auto-created handoff tool. The
# description names the target explicitly because it is the only thing the model
# has to go on when deciding whether to transfer; "transfers the conversation"
# on its own gives a small model no reason to prefer it over answering.
HANDOFF_DESCRIPTION = (
    "Transfers the conversation to the {target} agent. Use this when {target} is "
    "better suited to answer than you are."
)
HANDOFF_ARGS = {
    "reason": {
        "type": "string",
        "description": "Why the user is being transferred",
        "example": "wants to book an appointment",
    }
}
HANDOFF_POSITION = 9000

CONFIG_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id            TEXT PRIMARY KEY,
    system_prompt TEXT NOT NULL DEFAULT '',
    llm_model     TEXT,
    tts_voice     TEXT,
    rules         TEXT NOT NULL DEFAULT '[]',
    handoffs      TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT,
    updated_at    TEXT
);

-- Agent<->tool is many-to-many with an explicit position, because the order the
-- tools are listed in is the order the model sees them, and "order" has to
-- survive a round trip through the database.
CREATE TABLE IF NOT EXISTS agent_tools (
    agent_id  TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    position  INTEGER NOT NULL,
    PRIMARY KEY (agent_id, tool_name)
);

CREATE TABLE IF NOT EXISTS tools (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    args_schema TEXT,
    is_handoff  INTEGER NOT NULL DEFAULT 0,
    target      TEXT,
    position    INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS squads (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    entry_agent TEXT,
    created_at  TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS squad_edges (
    squad_id   TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    to_agent   TEXT NOT NULL,
    PRIMARY KEY (squad_id, from_agent, to_agent)
);

-- Single-row table. CHECK (id = 1) makes "more than one version row" a
-- constraint error rather than a silent ambiguity about which one is current.
CREATE TABLE IF NOT EXISTS config_version (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    version    INTEGER NOT NULL,
    updated_at TEXT
);

-- Key/value flags about the store itself rather than the config it holds.
-- `seeded_from_json` is what makes auto-seeding a once-per-database event
-- instead of a test for emptiness: see seed_from_json_if_empty().
CREATE TABLE IF NOT EXISTS config_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_tools_agent ON agent_tools(agent_id);
CREATE INDEX IF NOT EXISTS idx_squad_edges_squad ON squad_edges(squad_id);
"""


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _loads(text, fallback):
    """Parse a JSON column, tolerating NULL and corruption.

    A malformed config column should not make the whole control plane
    unreadable — the row that is broken is reported by validation, not by every
    read failing at once.
    """
    if text is None or text == "":
        return fallback
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return fallback
    return value if isinstance(value, type(fallback)) else fallback


class ConfigStore:
    """Read/write access to the config tables.

    Thread-safe in the same way as TraceStore: one connection, one lock, WAL.
    The control plane serves requests on a threadpool and the worker reads from
    its own task, so both are in play at once.
    """

    def __init__(self, path=DEFAULT_CONFIG_DB):
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(CONFIG_SCHEMA)
        # Seed the version row on first use so a reader never has to special-case
        # "no version yet" on a database that is simply empty.
        self.conn.execute(
            "INSERT OR IGNORE INTO config_version (id, version, updated_at) VALUES (1, 1, ?)",
            (_now_iso(),),
        )
        self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.commit()
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- version -----------------------------------------------------------

    def config_version(self):
        with self.lock:
            row = self.conn.execute(
                "SELECT version FROM config_version WHERE id = 1"
            ).fetchone()
        return int(row["version"]) if row else 1

    def _bump_version(self):
        """Mark the config as changed. Always called inside a write transaction.

        Bumping on every write — including the ones that turn out to be
        no-ops — is intentional. A no-op write that skipped the bump would leave
        a worker holding config the author believed they had just saved, and the
        resulting "I saved it but nothing changed" is much harder to debug than
        a redundant reload.
        """
        self.conn.execute(
            "UPDATE config_version SET version = version + 1, updated_at = ? WHERE id = 1",
            (_now_iso(),),
        )

    # -- agents ------------------------------------------------------------

    def list_agents(self):
        """Every agent as the shape AgentRegistry expects, tools in order."""
        with self.lock:
            rows = self.conn.execute("SELECT * FROM agents ORDER BY id").fetchall()
            tool_rows = self.conn.execute(
                "SELECT agent_id, tool_name FROM agent_tools ORDER BY agent_id, position"
            ).fetchall()
        return self._agents_from_rows(rows, tool_rows)

    def _agents_from_rows(self, rows, tool_rows):
        tools_by_agent = {}
        for r in tool_rows:
            tools_by_agent.setdefault(r["agent_id"], []).append(r["tool_name"])
        return [self._agent_row(r, tools_by_agent.get(r["id"], [])) for r in rows]

    @staticmethod
    def _agent_row(row, tool_names):
        agent = {
            "id": row["id"],
            "system_prompt": row["system_prompt"] or "",
            "tools": list(tool_names),
            "handoffs": _loads(row["handoffs"], []),
            "rules": _loads(row["rules"], []),
        }
        # Optional keys are omitted rather than emitted as null, so a round trip
        # through this store does not grow "llm_model": null into every agent
        # file on export.
        if row["llm_model"]:
            agent["llm_model"] = row["llm_model"]
        if row["tts_voice"]:
            agent["tts_voice"] = row["tts_voice"]
        return agent

    def get_agent(self, agent_id):
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            tool_rows = self.conn.execute(
                "SELECT tool_name FROM agent_tools WHERE agent_id = ? ORDER BY position",
                (agent_id,),
            ).fetchall()
        if not row:
            return None
        return self._agent_row(row, [r["tool_name"] for r in tool_rows])

    def save_agent(self, agent):
        """Insert or update one agent, replacing its tool list wholesale.

        Replacing rather than merging is what makes "the tools I sent are the
        tools this agent has" true. A merge would keep tools the caller had
        removed, which for this UI means an unchecking a checkbox appears not to
        work until the process restarts.
        """
        agent_id = agent["id"]
        with self.lock:
            self.conn.execute(
                "INSERT INTO agents (id, system_prompt, llm_model, tts_voice, rules, "
                "handoffs, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET system_prompt=excluded.system_prompt, "
                "llm_model=excluded.llm_model, tts_voice=excluded.tts_voice, "
                "rules=excluded.rules, handoffs=excluded.handoffs, "
                "updated_at=excluded.updated_at",
                (
                    agent_id,
                    agent.get("system_prompt", ""),
                    agent.get("llm_model"),
                    agent.get("tts_voice"),
                    json.dumps(agent.get("rules", [])),
                    json.dumps(agent.get("handoffs", [])),
                    _now_iso(),
                    _now_iso(),
                ),
            )
            self.conn.execute("DELETE FROM agent_tools WHERE agent_id = ?", (agent_id,))
            for position, tool_name in enumerate(agent.get("tools", [])):
                self.conn.execute(
                    "INSERT INTO agent_tools (agent_id, tool_name, position) VALUES (?, ?, ?)",
                    (agent_id, tool_name, position),
                )
            for target in agent.get("handoffs", []):
                self._ensure_handoff_tool(target)
            self._bump_version()
            self.conn.commit()
        return self.get_agent(agent_id)

    def _ensure_handoff_tool(self, target):
        """Create `handoff_to_<target>` if it is not already a tool row.

        Called under the lock, so it does not take it itself.

        This exists because a handoff is two facts — "agent A hands off to B" and
        "the model may call the handoff_to_B tool" — and saving only the first
        produces a config that passes validation and then breaks the call that
        uses it. build_prompt() looks the handoff tool up in the tool table and
        raises when it is missing, so a squad whose second agent was added
        through the UI would fail on the first turn that needed the handoff.

        Only created when absent, so a hand-written description survives: the
        existing config's "Transfers the conversation to a scheduling
        specialist" is better than anything generated, and the Tools screen
        treats a handoff tool as an ordinary row the user can edit.
        """
        name = f"handoff_to_{target}"
        if self.conn.execute("SELECT 1 FROM tools WHERE name = ?", (name,)).fetchone():
            return
        self.conn.execute(
            "INSERT INTO tools (name, description, args_schema, is_handoff, target, "
            "position, updated_at) VALUES (?, ?, ?, 1, ?, ?, ?)",
            (
                name,
                HANDOFF_DESCRIPTION.format(target=target),
                json.dumps(HANDOFF_ARGS),
                target,
                # Sorts after every real tool, so handoffs appear at the bottom of
                # the Tools screen rather than interleaved with callable ones.
                HANDOFF_POSITION,
                _now_iso(),
            ),
        )

    def delete_agent(self, agent_id):
        with self.lock:
            cur = self.conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            # agent_tools rows go with it. Without ON DELETE CASCADE this is
            # manual, and a missed row shows up later as a tool attached to an
            # agent that no longer exists.
            self.conn.execute("DELETE FROM agent_tools WHERE agent_id = ?", (agent_id,))
            self.conn.execute("DELETE FROM squad_edges WHERE from_agent = ? OR to_agent = ?",
                              (agent_id, agent_id))
            self.conn.execute("UPDATE squads SET entry_agent = NULL WHERE entry_agent = ?",
                              (agent_id,))
            if cur.rowcount:
                self._bump_version()
            self.conn.commit()
        return cur.rowcount > 0

    # -- tools -------------------------------------------------------------

    def list_tools(self):
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM tools ORDER BY position, name"
            ).fetchall()
        return [self._tool_row(r) for r in rows]

    @staticmethod
    def _tool_row(row):
        tool = {"description": row["description"] or ""}
        if row["args_schema"] is not None:
            tool["args_schema"] = _loads(row["args_schema"], {})
        if row["is_handoff"]:
            tool["is_handoff"] = True
            tool["target"] = row["target"]
        return tool

    def get_tool(self, name):
        with self.lock:
            row = self.conn.execute("SELECT * FROM tools WHERE name = ?", (name,)).fetchone()
        return self._tool_row(row) if row else None

    def save_tool(self, name, spec, position=0):
        """Insert or update one tool definition.

        args_schema is stored as the JSON text it was given, not reformatted:
        the flat form and the JSON Schema form are both valid input, and
        rewriting one into the other here would make export produce files that
        differ from what the author typed.
        """
        schema = spec.get("args_schema")
        with self.lock:
            self.conn.execute(
                "INSERT INTO tools (name, description, args_schema, is_handoff, target, "
                "position, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "args_schema=excluded.args_schema, is_handoff=excluded.is_handoff, "
                "target=excluded.target, position=excluded.position, "
                "updated_at=excluded.updated_at",
                (
                    name,
                    spec.get("description", ""),
                    None if schema is None else json.dumps(schema),
                    1 if spec.get("is_handoff") else 0,
                    spec.get("target"),
                    position,
                    _now_iso(),
                ),
            )
            self._bump_version()
            self.conn.commit()
        return self.get_tool(name)

    def delete_tool(self, name):
        with self.lock:
            cur = self.conn.execute("DELETE FROM tools WHERE name = ?", (name,))
            # Detach from agents as well, or every agent that listed it fails
            # validation with "unknown tool" the moment the tool is deleted.
            self.conn.execute("DELETE FROM agent_tools WHERE tool_name = ?", (name,))
            if cur.rowcount:
                self._bump_version()
            self.conn.commit()
        return cur.rowcount > 0

    # -- squads ------------------------------------------------------------

    def list_squads(self):
        with self.lock:
            squads = self.conn.execute("SELECT * FROM squads ORDER BY id").fetchall()
            edges = self.conn.execute("SELECT * FROM squad_edges").fetchall()
        edges_by_squad = {}
        for e in edges:
            edges_by_squad.setdefault(e["squad_id"], []).append(
                {"from": e["from_agent"], "to": e["to_agent"]}
            )
        return [
            {
                "id": s["id"],
                "name": s["name"] or s["id"],
                "entry_agent": s["entry_agent"],
                "edges": edges_by_squad.get(s["id"], []),
            }
            for s in squads
        ]

    def get_squad(self, squad_id):
        return next((s for s in self.list_squads() if s["id"] == squad_id), None)

    def save_squad(self, squad):
        squad_id = squad["id"]
        with self.lock:
            self.conn.execute(
                "INSERT INTO squads (id, name, entry_agent, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name, entry_agent=excluded.entry_agent, "
                "updated_at=excluded.updated_at",
                (squad_id, squad.get("name") or squad_id, squad.get("entry_agent"),
                 _now_iso(), _now_iso()),
            )
            self.conn.execute("DELETE FROM squad_edges WHERE squad_id = ?", (squad_id,))
            for edge in squad.get("edges", []):
                # self-loops are dropped rather than stored: a handoff to the
                # same agent is a no-op the model could still emit, and a graph
                # edge that points at itself confuses the visual builder.
                if edge.get("from") == edge.get("to"):
                    continue
                self.conn.execute(
                    "INSERT OR IGNORE INTO squad_edges (squad_id, from_agent, to_agent) "
                    "VALUES (?, ?, ?)",
                    (squad_id, edge["from"], edge["to"]),
                )
            self._bump_version()
            self.conn.commit()
        return self.get_squad(squad_id)

    def delete_squad(self, squad_id):
        with self.lock:
            cur = self.conn.execute("DELETE FROM squads WHERE id = ?", (squad_id,))
            self.conn.execute("DELETE FROM squad_edges WHERE squad_id = ?", (squad_id,))
            if cur.rowcount:
                self._bump_version()
            self.conn.commit()
        return cur.rowcount > 0

    # -- import / export ---------------------------------------------------

    def get_meta(self, key):
        row = self.conn.execute(
            "SELECT value FROM config_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key, value):
        with self.lock:
            self.conn.execute(
                "INSERT INTO config_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self.conn.commit()

    def seed_from_json_if_empty(self, agents_dir, tools_file):
        """Import the JSON config once, the first time this database is used.

        The condition is "this database has never been seeded", not "this
        database has no agents". Those differ, and the difference is the whole
        point: someone who deletes the last agent in the UI has made a
        deliberate state, and a restart that resurrects it from the files is a
        bug that looks like the UI losing work. A marker row is the only thing
        that can tell "never used" from "used and then emptied".

        Returns True if it imported, False if it did not. A database that is
        already populated, or already seeded once, is left alone — the database
        is the source of truth from here on.
        """
        with self.lock:
            if self.get_meta("seeded_from_json") is not None:
                return False
            if self.conn.execute("SELECT COUNT(*) AS n FROM agents").fetchone()["n"]:
                # Populated by some other route. Mark it so the check is one
                # cheap read from now on, then leave the content alone.
                self.conn.execute(
                    "INSERT OR REPLACE INTO config_meta (key, value) VALUES "
                    "('seeded_from_json', ?)",
                    ("already-populated",),
                )
                self.conn.commit()
                return False

        self.import_from_json(agents_dir, tools_file)
        self.set_meta("seeded_from_json", _now_iso())
        return True

    def import_from_json(self, agents_dir, tools_file):
        """Load the on-disk JSON config into the database, replacing what is there.

        Bumps the version once for the whole import rather than once per agent,
        so a worker polling for a change does not see a half-imported config
        where the agents exist but their tools do not.

        Replaces rather than merges, and the choice is about what "import" is
        supposed to mean. Merging leaves rows that only exist because someone
        once created an agent in the UI: the files are edited to remove an agent,
        the import runs, and the agent is still there — so the config on screen
        disagrees with the files the next time anyone reads them, with nothing
        recording which one is authoritative. Replace makes the files the whole
        truth for the duration of the call, which is what the endpoint tells the
        user it does. Export first if there is unsaved UI work to keep.
        """
        with self.lock:
            # Child rows first: agent_tools and squad_edges reference the rows
            # below, and the foreign keys are enforced.
            for table in ("agent_tools", "squad_edges", "agents", "squads", "tools"):
                self.conn.execute(f"DELETE FROM {table}")

            tools = {}
            if os.path.exists(tools_file):
                with open(tools_file) as f:
                    tools = json.load(f)
            for position, (name, spec) in enumerate(tools.items()):
                self.conn.execute(
                    "INSERT INTO tools (name, description, args_schema, is_handoff, target, "
                    "position, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                    "args_schema=excluded.args_schema, is_handoff=excluded.is_handoff, "
                    "target=excluded.target, position=excluded.position",
                    (name, spec.get("description", ""),
                     None if spec.get("args_schema") is None else json.dumps(spec["args_schema"]),
                     1 if spec.get("is_handoff") else 0, spec.get("target"),
                     position, _now_iso()),
                )

            if os.path.isdir(agents_dir):
                for filename in sorted(os.listdir(agents_dir)):
                    if not filename.endswith(".json"):
                        continue
                    with open(os.path.join(agents_dir, filename)) as f:
                        agent = json.load(f)
                    self.conn.execute(
                        "INSERT INTO agents (id, system_prompt, llm_model, tts_voice, rules, "
                        "handoffs, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                        "system_prompt=excluded.system_prompt, llm_model=excluded.llm_model, "
                        "tts_voice=excluded.tts_voice, rules=excluded.rules, "
                        "handoffs=excluded.handoffs",
                        (agent["id"], agent.get("system_prompt", ""), agent.get("llm_model"),
                         agent.get("tts_voice"), json.dumps(agent.get("rules", [])),
                         json.dumps(agent.get("handoffs", [])), _now_iso(), _now_iso()),
                    )
                    self.conn.execute("DELETE FROM agent_tools WHERE agent_id = ?", (agent["id"],))
                    for position, tool_name in enumerate(agent.get("tools", [])):
                        self.conn.execute(
                            "INSERT INTO agent_tools (agent_id, tool_name, position) "
                            "VALUES (?, ?, ?)", (agent["id"], tool_name, position),
                        )

            self._bump_version()
            self.conn.commit()

        return {
            "agents": len(self.list_agents()),
            "tools": len(self.list_tools()),
            "version": self.config_version(),
        }

    def export_to_json(self, agents_dir, tools_file, clear_dir=False):
        """Write the database back out to the JSON files.

        For backup and for reviewing a config change as a diff, which is the only
        way to see a prompt edit as a reviewable object.
        """
        os.makedirs(agents_dir, exist_ok=True)
        if clear_dir:
            for filename in os.listdir(agents_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(agents_dir, filename))

        with open(tools_file, "w") as f:
            json.dump(dict(self.list_tools_with_names()), f, indent=2)
            f.write("\n")

        written = []
        for agent in self.list_agents():
            path = os.path.join(agents_dir, f"{agent['id']}.json")
            with open(path, "w") as f:
                json.dump(agent, f, indent=2)
                f.write("\n")
            written.append(agent["id"])
        return written

    def list_tools_with_names(self):
        """(name, spec) pairs, so export can rebuild the keyed tools.json shape."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM tools ORDER BY position, name"
            ).fetchall()
        return [(r["name"], self._tool_row(r)) for r in rows]

    # -- worker support ----------------------------------------------------

    def load_snapshot(self):
        """Everything the worker needs to rebuild a registry, in one read.

        Returned as a dict shaped exactly like the JSON files so AgentRegistry
        can consume it without knowing where it came from.
        """
        # One lock acquisition for the whole snapshot, and the private row helpers
        # rather than the public list_* methods: this lock is not reentrant, so
        # calling a method that takes it again from inside here deadlocks the
        # worker on the first config reload.
        with self.lock:
            agent_rows = self.conn.execute("SELECT * FROM agents ORDER BY id").fetchall()
            tool_rows = self.conn.execute(
                "SELECT agent_id, tool_name FROM agent_tools ORDER BY agent_id, position"
            ).fetchall()
            tool_rows_all = self.conn.execute(
                "SELECT * FROM tools ORDER BY position, name"
            ).fetchall()
            version_row = self.conn.execute(
                "SELECT version FROM config_version WHERE id = 1"
            ).fetchone()

        tools = {r["name"]: self._tool_row(r) for r in tool_rows_all}
        return {
            "agents": {
                a["id"]: a for a in self._agents_from_rows(agent_rows, tool_rows)
            },
            "tools": tools,
            "version": int(version_row["version"]) if version_row else 1,
        }
