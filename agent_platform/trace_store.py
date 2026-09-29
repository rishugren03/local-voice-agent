"""SQLite trace store — sessions, calls, events and agent transitions.

Replaces the flat call_trace.jsonl. The JSONL had one line per event with a
different key set per event type, so every consumer (the dashboard, the scorer,
the concurrency analyzer) had to re-read the whole file and re-group it in a dict
to answer a question. Here the relationships the questions are actually about are
the schema, so they are asked in SQL instead:

    SELECT * FROM agent_transitions WHERE session_id = ? ORDER BY ts;
    SELECT call_id, SUM(duration_s) FROM events WHERE ts BETWEEN ? AND ? GROUP BY call_id;
    SELECT event_type, COUNT(*) FROM events WHERE session_id = ? GROUP BY event_type;

  sessions            one row per participant connection
  calls               one row per conversational turn (call_id is fresh per turn)
  events              one row per pipeline stage / lifecycle event
  agent_transitions   one row per actual change of active agent

agent_transitions is a table rather than a column on sessions because "show me
every agent transition in session X" is the query this platform most often needs
to answer, and a JSON history column would put it back to parsing.

Each event keeps its full payload in the `content` column as JSON, so nothing the
old JSONL could record is lost, but the handful of fields that get filtered or
aggregated on are also promoted to real columns where SQL can reach them.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime

DEFAULT_DB = "call_trace.db"

# Promoted out of the content blob into real columns. Each one is a field a
# reader filters, aggregates or joins on, which is the whole reason for the
# table: TEXT LIKE '%12 plus 15%' inside a JSON blob is not an indexable query.
# `error` is promoted because every degraded turn (LLM down, tool failed, piper
# missing, STT crashed) records one under the same key, so "which turns degraded
# and why" is one query across all of them.
# They stay in content as well, deliberately: a NULL column cannot be told apart
# from a field that was absent, so without the copy an event carrying
# "tool_name": null would come back with no tool_name key at all.
PROMOTED = ("duration_s", "agent", "next_agent", "text", "tool_name", "error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id            TEXT PRIMARY KEY,
    participant_identity  TEXT,
    started_at            TEXT,
    started_ts            REAL,
    ended_at              TEXT,
    ended_ts              REAL,
    turns_served          INTEGER,
    barge_ins             INTEGER,
    final_agent           TEXT
);

CREATE TABLE IF NOT EXISTS calls (
    call_id      TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    started_at   TEXT,
    started_ts   REAL,
    first_agent  TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    call_id     TEXT,
    event_type  TEXT NOT NULL,
    started_at  TEXT,
    ts          REAL,
    duration_s  REAL,
    agent       TEXT,
    next_agent  TEXT,
    text        TEXT,
    tool_name   TEXT,
    error       TEXT,
    content     TEXT
);

CREATE TABLE IF NOT EXISTS agent_transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    call_id     TEXT,
    from_agent  TEXT,
    to_agent    TEXT NOT NULL,
    reason      TEXT,
    started_at  TEXT,
    ts          REAL
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_call    ON events(call_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(event_type, ts);
CREATE INDEX IF NOT EXISTS idx_transitions    ON agent_transitions(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_calls_session  ON calls(session_id, started_ts);

-- One transition per call: a turn has exactly one responding agent, so the
-- handoff event and the llm event that reports the same move are one fact, not
-- two. The handoff is logged first and carries the reason, so it is the row that
-- survives. NULL call_ids (session lifecycle) stay unconstrained, because SQLite
-- treats NULLs as distinct in a unique index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_transitions_once
    ON agent_transitions(session_id, call_id, to_agent);
"""


def row_to_event(row):
    """Rebuilds the flat dict the old JSONL produced, from a row.

    The consumers downstream of the loader (the scorer's matching logic, the
    concurrency analyzer's window maths) are written against that shape, so
    handing it back unchanged keeps this a loader swap rather than a rewrite of
    600 lines of scoring.
    """
    event = {
        "session_id": row["session_id"],
        "call_id": row["call_id"],
        "timestamp": row["started_at"],
        "event": row["event_type"],
    }
    event.update(json.loads(row["content"] or "{}"))
    # The columns are the queryable copy of the same values; they only overwrite
    # when set, so a field that is present-but-null keeps the value content gave.
    for column in PROMOTED:
        if row[column] is not None:
            event[column] = row[column]
    return event


def _round(value):
    """Seconds rounded for display. None stays None: 0.0 would claim to know the
    duration when the truth is that the event never recorded one, and the UI shows
    a dash for the former and a real number for the latter."""
    return None if value is None else round(float(value), 3)


def _session_summary(row):
    """The shape the session list and the session header both use.

    The aggregate columns (stt_s, llm_s, tts_s, calls, errors) come from the
    list query's subqueries, but the detail view also builds one of these from a
    bare sessions row, so every aggregate is read defensively. Computing them
    again in the detail path is not worth a second code path that can disagree.
    """
    def col(name, default=None):
        return row[name] if name in row.keys() else default

    stt, llm, tts = _round(col("stt_s")), _round(col("llm_s")), _round(col("tts_s"))
    total = None
    if stt is not None and llm is not None and tts is not None:
        # The user's wait is the three stages in series. A missing stage makes
        # the sum unknowable rather than smaller, so it is reported as unknown
        # instead of quietly omitting the slowest part.
        total = round(stt + llm + tts, 3)
    return {
        "session_id": row["session_id"],
        "participant_identity": row["participant_identity"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "turns_served": row["turns_served"],
        "barge_ins": row["barge_ins"],
        "final_agent": row["final_agent"],
        # `n_calls` rather than `calls`: get_session() replaces this same key
        # with the list of turns, and a key that is an int from one endpoint and
        # a list from the next is a trap for any client that reads both — the
        # list endpoint happens to be the one a client hits first.
        "n_calls": col("calls", 0),
        "latency": {"stt": stt, "llm": llm, "tts": tts, "total": total},
        "errors": col("errors", 0),
    }


def _call_summary(row, events, transitions):
    """One turn: its stages, its text, and whether it degraded.

    Latency is read from the events rather than stored on the call, so it cannot
    disagree with the trace the drill-down shows for the same turn.
    """
    def stage_duration(name):
        for e in events:
            if e.get("event") == name and e.get("duration_s") is not None:
                return _round(e["duration_s"])
        return None

    stt = stage_duration("stt")
    llm = stage_duration("llm")
    tts = stage_duration("tts")
    total = round(stt + llm + tts, 3) if None not in (stt, llm, tts) else None

    errors = [e["error"] for e in events if e.get("error")]
    # The transcript is carried by the stt event, which is where it is produced.
    # A `user_speech` event is also accepted, because the worker logs one when
    # STT fails: then the error is the story and there is no text to show, and a
    # blank cell would read as "the user said nothing".
    user_text = next((e.get("text") for e in events
                      if e.get("event") in ("user_speech", "stt") and e.get("text")), None)
    reply = next((e.get("text") for e in events
                  if e.get("event") in ("tts", "reply") and e.get("text")), None)
    tools = [e.get("tool_name") for e in events
             if e.get("event") == "tool_result" and e.get("tool_name")]

    return {
        "call_id": row["call_id"],
        "started_at": row["started_at"],
        "first_agent": row["first_agent"],
        "latency": {"stt": stt, "llm": llm, "tts": tts, "total": total},
        "user_text": user_text,
        "reply": reply,
        "tools": tools,
        "errors": errors,
        "degraded": bool(errors),
        "transitions": transitions,
        "events": events,
    }


def _add_missing_columns(conn):
    """Bring a database written by an older version up to the current schema.

    CREATE TABLE IF NOT EXISTS will not add a column to a table that already
    exists, so a trace database created before a column was introduced keeps its
    old shape. That failure is nasty rather than loud: log_event() deliberately
    swallows write errors so a broken trace cannot silence the caller, which
    means a missing column would drop events with nothing on the console to
    explain it. The migration only ever adds columns — nothing here renames,
    retypes or drops, so it cannot lose the data already stored.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
    if not existing:  # a fresh database already matches SCHEMA
        return
    for column in PROMOTED:
        if column not in existing:
            conn.execute(f"ALTER TABLE events ADD COLUMN {column} TEXT")


class TraceStore:
    """Thread-safe writer for one trace database.

    Sessions append from concurrent tasks, and the dashboard/scorer read the same
    file while the agent is still running, so the connection is opened in WAL mode
    (readers do not block the writer) and every write is serialized by a lock —
    the same reason the JSONL path was guarded by TRACE_LOCK.
    """

    def __init__(self, path=DEFAULT_DB):
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL lets the dashboard and the scorer query a trace the agent is still
        # writing; without it a read takes the write lock and the pipeline stalls.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        _add_missing_columns(self.conn)
        self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.commit()
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- writing ---------------------------------------------------------

    def open_session(self, session_id, participant_identity):
        """Records a session the moment its participant connects, so the row
        exists even for a call that never produces an event."""
        now = datetime.now()
        with self.lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, participant_identity, "
                "started_at, started_ts) VALUES (?, ?, ?, ?)",
                (session_id, participant_identity, now.isoformat(), now.timestamp()),
            )
            self.conn.commit()

    def log_event(self, session_id, call_id, event_type, data, at=None):
        """Writes one event, and the session/call/transition rows it implies.

        Same signature as the old log_event() for the live path. `at` exists for
        the two callers that already know when an event happened: the JSONL
        importer and the synthetic-trace test builder. Without it both would be
        stamped with the time they were replayed, which collapses a whole trace
        onto one instant and puts every event outside the eval window.
        """
        now = at or datetime.now()
        started_at = now.isoformat()
        ts = now.timestamp()

        promoted = {k: data.get(k) for k in PROMOTED}
        # The whole payload is kept, promoted fields included, so a trace read
        # back is indistinguishable from the one that was logged.
        content = dict(data)

        with self.lock:
            if session_id:
                # An event can be the first thing seen of a session when the
                # caller skipped open_session(), so the row is created here too.
                self.conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, started_at, started_ts) "
                    "VALUES (?, ?, ?)", (session_id, started_at, ts))
            if call_id:
                self.conn.execute(
                    "INSERT OR IGNORE INTO calls (call_id, session_id, started_at, started_ts) "
                    "VALUES (?, ?, ?, ?)", (call_id, session_id, started_at, ts))

            self.conn.execute(
                "INSERT INTO events (session_id, call_id, event_type, started_at, ts, "
                "duration_s, agent, next_agent, text, tool_name, error, content) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, call_id, event_type, started_at, ts,
                 promoted["duration_s"], promoted["agent"], promoted["next_agent"],
                 promoted["text"], promoted["tool_name"], promoted["error"],
                 json.dumps(content, default=str)),
            )

            for transition in self._transitions_for(event_type, data, session_id,
                                                    call_id, started_at, ts):
                self.conn.execute(
                    "INSERT OR IGNORE INTO agent_transitions (session_id, call_id, "
                    "from_agent, to_agent, reason, started_at, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, call_id, *transition, started_at, ts),
                )

            if event_type == "session_end":
                self.conn.execute(
                    "UPDATE sessions SET ended_at = ?, ended_ts = ?, turns_served = ?, "
                    "barge_ins = ?, final_agent = ?, participant_identity = COALESCE("
                    "participant_identity, ?) WHERE session_id = ?",
                    (started_at, ts, data.get("turns_served"), data.get("barge_ins"),
                     data.get("final_agent"), data.get("identity"), session_id),
                )
            self.conn.commit()

    @staticmethod
    def _transitions_for(event_type, data, session_id, call_id, started_at, ts):
        """(from_agent, to_agent, reason) for each real change of active agent.

        An llm event carries agent -> next_agent on every turn, which is almost
        always the same agent twice. Only actual changes are recorded, so the
        table stays a record of handoffs rather than a second copy of every turn.
        """
        if not session_id:
            return []
        if event_type == "handoff":
            to_agent = data.get("to_agent")
            if to_agent and to_agent != data.get("from_agent"):
                return [(data.get("from_agent"), to_agent, data.get("reason"))]
        elif event_type == "llm":
            to_agent = data.get("next_agent")
            if to_agent and to_agent != data.get("agent"):
                return [(data.get("agent"), to_agent, None)]
        return []

    def set_first_agent(self, call_id, agent):
        """Notes the agent that opened a call, so calls can be grouped by agent
        without scanning their events."""
        with self.lock:
            self.conn.execute(
                "UPDATE calls SET first_agent = COALESCE(first_agent, ?) WHERE call_id = ?",
                (agent, call_id))
            self.conn.commit()

    # -- read helpers -------------------------------------------------------
    #
    # Added for the Calls screen. Until now the only readers were the scorer and
    # the dashboard build script, which open their own connections and write their
    # own SQL. The UI needs the same data over HTTP, and the alternative — having
    # the API shell out to the scorer or re-derive the aggregation in the endpoint —
    # means the number shown in the UI and the number the scorer prints are
    # computed by different code and eventually disagree. These live here so
    # there is one definition of "how long was this turn".

    def list_sessions(self, limit=50, offset=0, search=None):
        """Sessions newest first, with the aggregates the session list shows.

        Aggregates are computed in one pass with correlated subqueries rather than
        a per-row Python loop: at 50 rows the difference is invisible, but this
        list is also the entry point for a user hunting for one bad call in a
        long-running database, and a row-per-subquery loop on a 10k-row table is
        a multi-second response.
        """
        params = []
        where = ""
        if search:
            where = ("WHERE s.session_id LIKE ? OR s.participant_identity LIKE ? "
                     "OR s.final_agent LIKE ?")
            params += [f"%{search}%"] * 3
        params += [limit, offset]
        rows = self.conn.execute(
            "SELECT s.*, "
            "  (SELECT COUNT(*) FROM calls c WHERE c.session_id = s.session_id) AS calls, "
            "  (SELECT MIN(e.duration_s) FROM events e WHERE e.session_id = s.session_id "
            "     AND e.event_type = 'stt' AND e.duration_s IS NOT NULL) AS stt_s, "
            "  (SELECT MIN(e.duration_s) FROM events e WHERE e.session_id = s.session_id "
            "     AND e.event_type = 'llm' AND e.duration_s IS NOT NULL) AS llm_s, "
            "  (SELECT MIN(e.duration_s) FROM events e WHERE e.session_id = s.session_id "
            "     AND e.event_type = 'tts' AND e.duration_s IS NOT NULL) AS tts_s, "
            "  (SELECT COUNT(*) FROM events e WHERE e.session_id = s.session_id "
            "     AND e.error IS NOT NULL) AS errors "
            "FROM sessions s " + where +
            " ORDER BY s.started_ts DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_session_summary(r) for r in rows]

    def count_sessions(self, search=None):
        if search:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE session_id LIKE ? "
                "OR participant_identity LIKE ? OR final_agent LIKE ?",
                (f"%{search}%",) * 3,
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()
        return row["n"]

    def get_session(self, session_id):
        """One session with its full turn list, in call order.

        Events are fetched for the whole session in a single query rather than
        per call. Round-tripping the DB once instead of once per turn is the
        difference between an instant response and a visible stall on a session
        with a few dozen calls.
        """
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None

        calls = self.conn.execute(
            "SELECT * FROM calls WHERE session_id = ? ORDER BY started_ts",
            (session_id,),
        ).fetchall()
        events = self.conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY ts, id",
            (session_id,),
        ).fetchall()
        transitions = self.conn.execute(
            "SELECT * FROM agent_transitions WHERE session_id = ? ORDER BY ts, id",
            (session_id,),
        ).fetchall()

        events_by_call = {}
        for e in events:
            events_by_call.setdefault(e["call_id"], []).append(row_to_event(e))
        transitions_by_call = {}
        for t in transitions:
            transitions_by_call.setdefault(t["call_id"], []).append({
                "from": t["from_agent"],
                "to": t["to_agent"],
                "reason": t["reason"],
                "at": t["started_at"],
            })

        detail = _session_summary(row)
        # The bare sessions row has no aggregate subqueries, so the count has to
        # come from the turns that were just loaded. Set before `calls` is
        # overwritten with the list itself.
        detail["n_calls"] = len(calls)
        detail["transitions"] = [
            {k: t[k] for k in ("from_agent", "to_agent", "reason", "started_at")}
            for t in transitions
        ]
        detail["calls"] = [
            _call_summary(c, events_by_call.get(c["call_id"], []),
                          transitions_by_call.get(c["call_id"], []))
            for c in calls
        ]
        return detail

    def get_call(self, call_id):
        """One turn with its events, for the trace drill-down.

        Looked up by call_id alone: the UI has a call in hand from the session
        view and should not have to know the session id first, and call_id is
        already unique.
        """
        call = self.conn.execute(
            "SELECT * FROM calls WHERE call_id = ?", (call_id,)
        ).fetchone()
        if call is None:
            return None
        events = self.conn.execute(
            "SELECT * FROM events WHERE call_id = ? ORDER BY ts, id", (call_id,)
        ).fetchall()
        transitions = self.conn.execute(
            "SELECT * FROM agent_transitions WHERE call_id = ? ORDER BY ts, id",
            (call_id,),
        ).fetchall()
        session = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (call["session_id"],)
        ).fetchone()

        summary = _call_summary(call, [row_to_event(e) for e in events], [{
            "from": t["from_agent"], "to": t["to_agent"],
            "reason": t["reason"], "at": t["started_at"],
        } for t in transitions])
        summary["session_id"] = call["session_id"]
        summary["participant_identity"] = session["participant_identity"] if session else None
        return summary

    def get_timeline(self, since_ts=None, session_id=None, limit=500):
        """Recent events across sessions, for the live trace feed.

        Limited hard: this backs a view that streams, and an unbounded scan of
        the events table is how an API ends up holding a connection open long
        enough to block the worker's writes on the same database.
        """
        sql = "SELECT * FROM events"
        clauses, params = [], []
        if since_ts is not None:
            clauses.append("ts > ?")
            params.append(since_ts)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(limit)
        return [row_to_event(r) for r in self.conn.execute(sql, params).fetchall()]

    def import_jsonl(self, path):
        """Loads a legacy call_trace.jsonl into the database.

        The scored reliability numbers in the README came from JSONL traces, and
        re-running the scorer against them should not require the old writer.
        Events that are already imported are skipped, so re-running an import
        after a live run has appended to the same database does not double it.
        """
        imported = skipped = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                exists = self.conn.execute(
                    "SELECT 1 FROM events WHERE session_id IS ? AND call_id IS ? "
                    "AND event_type = ? AND started_at = ?",
                    (entry.get("session_id"), entry.get("call_id"),
                     entry.get("event"), entry.get("timestamp")),
                ).fetchone()
                if exists:
                    skipped += 1
                    continue
                data = {k: v for k, v in entry.items()
                        if k not in ("session_id", "call_id", "timestamp", "event")}
                # The importer reuses log_event() rather than reimplementing the
                # insert, so an imported trace populates sessions, calls and
                # agent_transitions exactly as a live one would. The recorded
                # timestamp is carried over, not replaced with the import time.
                self.log_event(entry.get("session_id"), entry.get("call_id"),
                               entry.get("event"), data,
                               at=datetime.fromisoformat(entry["timestamp"]))
                imported += 1
        return imported, skipped


# --- reading ---------------------------------------------------------------

def connect_readonly(path=DEFAULT_DB):
    """Read-only connection for the dashboard, the scorer and one-off queries."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no trace database at {path} — run the agent first, or import an old "
            f"trace with score_eval.py --import-jsonl <file>"
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_events(start_ts=None, end_ts=None, path=DEFAULT_DB):
    """Every event in a time window, oldest first, as the legacy flat dicts.

    The window is a SQL range on the epoch column rather than a filter applied
    after loading the file, which is what let the scorer read a whole trace to
    find the handful of events inside the eval run.
    """
    conn = connect_readonly(path)
    try:
        sql = "SELECT * FROM events"
        clauses, params = [], []
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("ts <= ?")
            params.append(end_ts)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts"
        return [row_to_event(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def session_transitions(session_id, path=DEFAULT_DB):
    """Every agent transition in one session, in order. The query the JSONL
    could not express."""
    conn = connect_readonly(path)
    try:
        rows = conn.execute(
            "SELECT from_agent, to_agent, reason, call_id, started_at FROM "
            "agent_transitions WHERE session_id = ? ORDER BY ts", (session_id,))
        return [dict(row) for row in rows]
    finally:
        conn.close()
