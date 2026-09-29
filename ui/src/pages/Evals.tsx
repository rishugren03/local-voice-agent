/**
 * U6 — Evals.
 *
 * Runs the scripted suite against the live worker and shows the score. The
 * screen is a monitor, not a form: a run takes minutes, so the job is started
 * with one click and this page polls for progress, streaming the harness's own
 * output into a log panel.
 *
 * Showing the raw log is deliberate. When a case fails, the reason is almost
 * always in the runner's output — a missing audio file, a room mismatch, a
 * worker that never joined — and a bare "4/7 passed" leaves the user with
 * nowhere to go but a terminal.
 *
 * Latency is shown as p50 and p90, not a mean. A mean over a handful of calls
 * is dominated by whichever call hit a cold model, and p90 is the number that
 * tells you whether the slow path is the normal path.
 */

import { useCallback, useEffect, useState } from "react";
import { Chip, Empty, Problems } from "../components";
import { usePoll } from "../hooks";
import { ApiError, type EvalJob, type EvalReport, api } from "../api";

const STATUS_TONE: Record<string, "ok" | "warn" | "error" | "neutral"> = {
  PASS: "ok",
  PARTIAL: "warn",
  FAIL: "error",
};

function elapsed(from: number, to: number | null): string {
  const seconds = Math.floor((to ?? Date.now() / 1000) - from);
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function Evals() {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [job, setJob] = useState<EvalJob | null>(null);
  const [preflight, setPreflight] = useState<string[]>([]);
  const [users, setUsers] = useState(1);
  const [rotate, setRotate] = useState(0);
  const [starting, setStarting] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [showLog, setShowLog] = useState(false);

  const running = job?.state === "running";

  const load = useCallback(async () => {
    try {
      const result = await api.evals();
      setReport(result.report);
      setJob(result.job);
      setPreflight(result.preflight ?? []);
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Only poll while a run is in flight. A finished report is a file on disk; it
  // does not change, and re-fetching it every few seconds forever is just
  // noise in the logs.
  usePoll(load, 2000, running);

  async function start() {
    setStarting(true);
    setProblems([]);
    try {
      setJob(await api.startEval(users, rotate));
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    } finally {
      setStarting(false);
    }
  }

  const overall = report?.latency_overall;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Evals</h1>
          <p>
            Runs a scripted caller through a conversation with the live worker and
            scores the result from the trace database. Everything runs locally.
          </p>
        </div>
        <div className="row tight">
          {running ? (
            <>
              <Chip tone="warn" dot>
                {job?.stage ?? "running"} · {elapsed(job?.started_at ?? 0, null)}
              </Chip>
              <button className="btn sm" onClick={() => setShowLog(!showLog)}>
                {showLog ? "Hide log" : "Show log"}
              </button>
            </>
          ) : (
            <button
              className="btn primary"
              onClick={start}
              disabled={starting || preflight.length > 0}
            >
              {starting ? "Starting…" : "Run eval"}
            </button>
          )}
        </div>
      </div>

      <Problems problems={[...preflight, ...problems]} />

      {!running && (
        <div className="card">
          <h2>Run settings</h2>
          <p className="hint">
            One user is the baseline. Two concurrent callers on one CPU is the stress
            case — it is what proves the scorer attributes each call to the right
            conversation.
          </p>
          <div className="row">
            <label className="field" style={{ margin: 0, minWidth: 150 }}>
              <span className="field-label">Callers</span>
              <select
                value={users}
                onChange={(e) => setUsers(Number(e.target.value))}
              >
                {[1, 2, 3, 4].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label className="field" style={{ margin: 0, minWidth: 150 }}>
              <span className="field-label">Rotation</span>
              <select
                value={rotate}
                onChange={(e) => setRotate(Number(e.target.value))}
              >
                {[0, 1, 2, 3].map((n) => (
                  <option key={n} value={n}>
                    {n === 0 ? "off" : n}
                  </option>
                ))}
              </select>
            </label>
            <div className="faint" style={{ fontSize: 12, flex: 1, minWidth: 220 }}>
              Rotation starts each caller at a different point in the script, so the
              call order in the trace stops matching the suite order.
            </div>
          </div>
        </div>
      )}

      {showLog && job && (
        <div className="card">
          <div className="spread">
            <h2>Runner output</h2>
            <span className="faint" style={{ fontSize: 12 }}>
              job {job.id}
            </span>
          </div>
          <pre className="preview" style={{ maxHeight: 260 }}>
            {job.log.length ? job.log.join("\n") : "Waiting for output…"}
          </pre>
        </div>
      )}

      {!report && !running && (
        <div className="card">
          <Empty>
            No eval has been run yet. Start one above — it takes a couple of minutes
            and needs the worker running in the same room.
          </Empty>
        </div>
      )}

      {report && (
        <>
          <div className="card">
            <div className="spread">
              <div>
                <h2 style={{ margin: 0, fontSize: 22 }}>
                  {report.passed}
                  <span className="faint" style={{ fontWeight: 400 }}>
                    /{report.total} passed
                  </span>
                </h2>
                <p className="hint" style={{ marginTop: 4 }}>
                  {report.n_calls} calls across {report.n_sessions} session
                  {report.n_sessions === 1 ? "" : "s"}
                  {report.window
                    ? ` · ${new Date(report.window.start * 1000).toLocaleTimeString()} – ${new Date(
                        report.window.end * 1000,
                      ).toLocaleTimeString()}`
                    : ""}
                </p>
              </div>
              <div className="row tight">
                {report.passed === report.total ? (
                  <Chip tone="ok">all cases pass</Chip>
                ) : (
                  <Chip tone="error">{report.total - report.passed} failing</Chip>
                )}
              </div>
            </div>

            {overall && (
              <table style={{ marginTop: 14 }}>
                <thead>
                  <tr>
                    <th>Latency</th>
                    <th>STT p50</th>
                    <th>LLM p50</th>
                    <th>TTS p50</th>
                    <th>Total p50</th>
                    <th>Total p90</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="dim">all calls</td>
                    {(
                      [
                        overall.stt_p50,
                        overall.llm_p50,
                        overall.tts_p50,
                        overall.total_p50,
                        overall.total_p90,
                      ] as const
                    ).map((value, i) => (
                      <td key={i} className="mono">
                        {value === null || value === undefined ? "—" : `${value.toFixed(2)}s`}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            )}
          </div>

          {report.sessions.map((session) => (
            <div className="card" key={session.session_id}>
              <div className="spread">
                <h2 className="mono" style={{ margin: 0 }}>
                  {session.identity ?? session.session_id}
                </h2>
                <div className="row tight">
                  {!session.known_cases && <Chip tone="warn">cases assumed</Chip>}
                  {session.rejected_silent_turns.length > 0 && (
                    <Chip tone="warn">
                      {session.rejected_silent_turns.length} silent dropped
                    </Chip>
                  )}
                  <Chip tone={session.passed === session.results.length ? "ok" : "error"}>
                    {session.passed}/{session.results.length}
                  </Chip>
                </div>
              </div>

              {session.session_end && (
                <p className="hint" style={{ marginTop: 6 }}>
                  {session.session_end.turns_served} turns ·{" "}
                  {session.session_end.barge_ins} barge-in · ended on{" "}
                  {session.session_end.final_agent ?? "unknown"}
                </p>
              )}

              <table style={{ marginTop: 10 }}>
                <thead>
                  <tr>
                    <th style={{ width: 70 }}>Result</th>
                    <th style={{ width: 150 }}>Case</th>
                    <th>Detail</th>
                    <th style={{ width: 100 }}>Match</th>
                  </tr>
                </thead>
                <tbody>
                  {session.results.map((result, i) => (
                    <tr key={i}>
                      <td>
                        <Chip tone={STATUS_TONE[result.status] ?? "neutral"}>{result.status}</Chip>
                      </td>
                      <td className="mono">{result.case}</td>
                      <td className="dim">{result.detail}</td>
                      <td className="faint" style={{ fontSize: 11.5 }}>
                        {result.match_mode}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {(session.unmatched_detail.length > 0 || session.rejected_silent_turns.length > 0) && (
                <div className="notice">
                  {session.unmatched_detail.length > 0 && (
                    <div>
                      <strong>{session.unmatched_detail.length} call(s) matched no expected case:</strong>
                      {session.unmatched_detail.map((u) => (
                        <div key={u.call_id} className="mono" style={{ fontSize: 12 }}>
                          {u.call_id}: heard “{u.heard}”
                        </div>
                      ))}
                    </div>
                  )}
                  {session.rejected_silent_turns.length > 0 && (
                    <div style={{ marginTop: 6 }}>
                      <strong>Silent turns dropped by the VAD:</strong> not scored as
                      calls, which is usually the right call — a turn with no speech is
                      not a failure.
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </>
      )}

      {job?.state === "failed" && (
        <div className="card">
          <h2>Run failed</h2>
          <Problems problems={[job.error ?? "unknown error"]} />
          <button className="btn sm" onClick={() => setShowLog(true)}>
            Show the runner output
          </button>
        </div>
      )}
    </>
  );
}
