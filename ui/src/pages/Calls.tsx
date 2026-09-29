/**
 * U3 — Calls.
 *
 * Two levels: sessions (a participant's conversation) and calls (one turn). The
 * split is the backend's, and it is the right one — "why did it hand off at
 * turn 4" is a question about a session, "why was that turn 9 seconds" is a
 * question about a call, and conflating them makes both hard to answer.
 *
 * The default sort is most-recent-first with a deliberate bias toward problems.
 * An index that sorts by time hides the one degraded call you wanted among a
 * hundred good ones, so failures float to the top of the list instead.
 */

import { useCallback, useEffect, useState } from "react";
import { Chip, Empty, LatencyBar, Problems, Spinner } from "../components";
import { useDebounced, usePoll } from "../hooks";
import { ApiError, type CallDetail, type SessionDetail, type SessionSummary, api } from "../api";

function ago(iso: string | null): string {
  if (!iso) return "—";
  const then = new Date(iso.endsWith("Z") ? iso : `${iso}Z`).getTime();
  if (Number.isNaN(then)) return iso;
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export default function Calls() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [onlyProblems, setOnlyProblems] = useState(false);

  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [openCall, setOpenCall] = useState<string | null>(null);
  const [callDetail, setCallDetail] = useState<CallDetail | null>(null);

  const [loading, setLoading] = useState(true);
  const [problems, setProblems] = useState<string[]>([]);

  const debouncedSearch = useDebounced(search);

  const load = useCallback(async () => {
    try {
      const result = await api.listSessions(50, 0, debouncedSearch);
      setSessions(result.sessions);
      setTotal(result.total);
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    } finally {
      setLoading(false);
    }
  }, [debouncedSearch]);

  useEffect(() => {
    void load();
  }, [load]);

  usePoll(load, 5000);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    api.getSession(selected).then(setDetail).catch(() => setDetail(null));
  }, [selected]);

  useEffect(() => {
    if (!openCall) {
      setCallDetail(null);
      return;
    }
    api.getCall(openCall).then(setCallDetail).catch(() => setCallDetail(null));
  }, [openCall]);

  const visible = onlyProblems
    ? sessions.filter((s) => s.errors > 0 || s.latency.total === null)
    : sessions;

  if (loading) return <Spinner />;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Calls</h1>
          <p>
            Every session served by the worker, with per-turn latency and any tool
            failure. Auto-refreshing.
          </p>
        </div>
        <div className="row tight">
          <input
            type="text"
            placeholder="Search identity or agent"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: 200 }}
          />
          <button
            className={`btn sm ${onlyProblems ? "primary" : ""}`}
            onClick={() => setOnlyProblems(!onlyProblems)}
          >
            Problems only
          </button>
        </div>
      </div>

      <Problems problems={problems} />

      {sessions.length === 0 ? (
        <div className="card">
          <Empty>
            No calls recorded yet. Start one on the Talk screen, or run the eval
            harness to populate this view.
          </Empty>
        </div>
      ) : (
        <div className="split">
          <div className="card">
            <div className="spread" style={{ marginBottom: 8 }}>
              <h2 style={{ margin: 0 }}>Sessions</h2>
              <span className="faint" style={{ fontSize: 12 }}>
                {visible.length} of {total}
              </span>
            </div>
            <div className="list">
              {visible.map((session) => (
                <button
                  key={session.session_id}
                  className={`list-item ${session.session_id === selected ? "active" : ""}`}
                  onClick={() => {
                    setSelected(session.session_id);
                    setOpenCall(null);
                  }}
                >
                  <div className="spread">
                    <span className="mono">
                      {session.participant_identity ?? session.session_id.slice(0, 12)}
                    </span>
                    {session.errors > 0 ? (
                      <Chip tone="error">{session.errors} err</Chip>
                    ) : session.latency.total === null ? (
                      <Chip tone="warn">incomplete</Chip>
                    ) : null}
                  </div>
                  <span className="sub">
                    {ago(session.started_at)} · {session.n_calls} turn
                    {session.n_calls === 1 ? "" : "s"}
                    {session.barge_ins ? ` · ${session.barge_ins} barge-in` : ""}
                    {session.final_agent ? ` · ${session.final_agent}` : ""}
                  </span>
                </button>
              ))}
              {visible.length === 0 && <Empty>No sessions match.</Empty>}
            </div>
          </div>

          <div>
            {!detail ? (
              <div className="card">
                <Empty>Pick a session.</Empty>
              </div>
            ) : (
              <>
                <div className="card">
                  <div className="spread">
                    <h2 style={{ margin: 0 }} className="mono">
                      {detail.participant_identity ?? detail.session_id}
                    </h2>
                    <div className="row tight">
                      {(detail.barge_ins ?? 0) > 0 && (
                        <Chip tone="accent">{detail.barge_ins} barge-in</Chip>
                      )}
                      {detail.errors > 0 && <Chip tone="error">{detail.errors} errors</Chip>}
                      {detail.ended_at ? (
                        <Chip tone="ok">ended</Chip>
                      ) : (
                        <Chip tone="warn" dot>
                          in progress
                        </Chip>
                      )}
                    </div>
                  </div>
                  <p className="hint" style={{ marginTop: 6 }}>
                    Started {ago(detail.started_at)} · {detail.turns_served ?? detail.n_calls} turns
                    served
                    {detail.final_agent ? ` · ended on ${detail.final_agent}` : ""}
                  </p>
                </div>

                {detail.transitions.length > 0 && (
                  <div className="card">
                    <h2>Handoffs</h2>
                    <p className="hint">
                      Each transition is a turn spent on the decision. A handoff that
                      bounces back and forth usually means a handoff tool is described
                      too loosely.
                    </p>
                    <table>
                      <thead>
                        <tr>
                          <th>From</th>
                          <th>To</th>
                          <th>Reason</th>
                          <th style={{ width: 80 }}>When</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.transitions.map((t, i) => (
                          <tr key={i}>
                            <td className="mono">{t.from_agent ?? "—"}</td>
                            <td className="mono">{t.to_agent}</td>
                            <td className="dim">{t.reason ?? "—"}</td>
                            <td className="faint">{ago(t.started_at)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <div className="card">
                  <h2>Turns</h2>
                  <p className="hint">Click a turn for the full event stream.</p>
                  {detail.calls.length === 0 ? (
                    <Empty>No turns recorded.</Empty>
                  ) : (
                    <table>
                      <thead>
                        <tr>
                          <th style={{ width: 60 }}>#</th>
                          <th>Said → replied</th>
                          <th style={{ width: 150 }}>Latency</th>
                          <th style={{ width: 80 }}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.calls.map((call, i) => (
                          <tr
                            key={call.call_id}
                            className={`clickable ${call.call_id === openCall ? "selected" : ""}`}
                            onClick={() =>
                              setOpenCall(call.call_id === openCall ? null : call.call_id)
                            }
                          >
                            <td className="faint">{i + 1}</td>
                            <td>
                              <div className="truncate dim">
                                <strong style={{ color: "var(--text)" }}>you:</strong>{" "}
                                {call.user_text ?? "—"}
                              </div>
                              <div className="truncate">
                                <strong className="faint">agent:</strong> {call.reply ?? "—"}
                              </div>
                              {call.tools.length > 0 && (
                                <div className="row tight" style={{ marginTop: 3 }}>
                                  {call.tools.map((t) => (
                                    <Chip key={t} tone="accent" mono>
                                      {t}
                                    </Chip>
                                  ))}
                                </div>
                              )}
                            </td>
                            <td>
                              <LatencyBar latency={call.latency} />
                            </td>
                            <td>
                              {call.errors.length > 0 ? (
                                <Chip tone="error" dot>
                                  {call.errors.length}
                                </Chip>
                              ) : call.degraded ? (
                                <Chip tone="warn" dot>
                                  degraded
                                </Chip>
                              ) : (
                                <Chip tone="ok" dot>
                                  ok
                                </Chip>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>

                {callDetail && (
                  <div className="card">
                    <div className="spread">
                      <h2>Turn detail</h2>
                      <button className="btn sm" onClick={() => setOpenCall(null)}>
                        Close
                      </button>
                    </div>
                    <p className="hint">
                      <span className="mono">{callDetail.call_id}</span> ·{" "}
                      {callDetail.session_id}
                    </p>
                    <div className="row tight" style={{ marginBottom: 10 }}>
                      {(
                        [
                          ["stt", callDetail.latency.stt],
                          ["llm", callDetail.latency.llm],
                          ["tts", callDetail.latency.tts],
                          ["total", callDetail.latency.total],
                        ] as const
                      ).map(([stage, value]) => (
                        <Chip key={stage} mono>
                          {stage} {value === null ? "—" : `${value.toFixed(2)}s`}
                        </Chip>
                      ))}
                    </div>
                    {callDetail.errors.length > 0 && (
                      <Problems problems={callDetail.errors} />
                    )}
                    {callDetail.transitions.length > 0 && (
                      <table style={{ marginBottom: 10 }}>
                        <thead>
                          <tr>
                            <th>Transition</th>
                            <th>Reason</th>
                          </tr>
                        </thead>
                        <tbody>
                          {callDetail.transitions.map((t, i) => (
                            <tr key={i}>
                              <td className="mono">
                                {t.from ?? "—"} → {t.to}
                              </td>
                              <td className="dim">{t.reason ?? "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                    <pre className="preview">{JSON.stringify(callDetail.events, null, 2)}</pre>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
