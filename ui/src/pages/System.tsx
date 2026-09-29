/**
 * U7 — System.
 *
 * What is actually running, and what to do about the things that are not. The
 * health checks come from the backend because only the backend can tell the
 * difference between "Ollama is not installed" and "Ollama is not warm" — the
 * first is a PATH problem and the second just makes the first call slow, and
 * conflating them sends people down the wrong path.
 *
 * The import/export buttons live here too, next to the version number, because
 * they are the boundary between the SQLite database the worker reads and the
 * JSON files that survive a reinstall. That is a system-level fact, not a
 * per-agent one.
 */

import { useCallback, useEffect, useState } from "react";
import { Chip, Problems, ServiceRow, Spinner } from "../components";

import { ApiError, type ConfigVersion, type Health, api } from "../api";

export default function System() {
  const [health, setHealth] = useState<Health | null>(null);
  const [version, setVersion] = useState<ConfigVersion | null>(null);
  const [validation, setValidation] = useState<{ ok: boolean; problems: string[] } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [notice, setNotice] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [h, v, val] = await Promise.all([api.health(), api.version(), api.validate()]);
      setHealth(h);
      setVersion(v);
      setValidation(val);
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function act(kind: "import" | "export") {
    setBusy(kind);
    setProblems([]);
    setNotice(null);
    try {
      if (kind === "import") {
        const result = await api.importConfig();
        setNotice(
          `Imported ${result.agents} agents and ${result.tools} tools from JSON. ` +
            `The worker picks this up on the next call.`,
        );
      } else {
        const result = await api.exportConfig();
        setNotice(`Wrote ${result.agents_dir} and ${result.tools_file}.`);
      }
      await load();
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>System</h1>
          <p>
            Service status, model availability, and the JSON files that back up the
            live configuration.
          </p>
        </div>
        {version && (
          <Chip mono tone="neutral">
            config v{version.config_version}
          </Chip>
        )}
      </div>

      <Problems problems={problems} />
      {notice && <div className="notice">{notice}</div>}

      <div className="card">
        <h2>Services</h2>
        <p className="hint">
          Checked live on every load. A "down" service is the usual reason an agent
          sounds worse than its prompt suggests.
        </p>
        {!health ? (
          <Spinner />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Service</th>
                <th>Status</th>
                <th>Detail and fix</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(health.services).map(([name, check]) => (
                <ServiceRow key={name} name={name} check={check} />
              ))}
            </tbody>
          </table>
        )}
      </div>

      {validation && (
        <div className="card">
          <h2>Configuration</h2>
          <p className="hint">
            Validated with the same code the worker uses, so this cannot pass here and
            fail there.
          </p>
          {validation.ok ? (
            <Chip tone="ok" dot>
              valid
            </Chip>
          ) : (
            <Problems problems={validation.problems} />
          )}
        </div>
      )}

      <div className="card">
        <h2>Backup and restore</h2>
        <p className="hint">
          The worker reads <span className="mono">call_trace.db</span>. These JSON files
          are the durable copy — export before reinstalling, import to restore.
        </p>
        <div className="row">
          <button className="btn" onClick={() => act("export")} disabled={busy !== null}>
            {busy === "export" ? "Exporting…" : "Export to JSON"}
          </button>
          <button className="btn" onClick={() => act("import")} disabled={busy !== null}>
            {busy === "import" ? "Importing…" : "Import from JSON"}
          </button>
        </div>
        <div className="notice" style={{ marginTop: 12 }}>
          <strong>Import replaces the live configuration.</strong> Anything edited in
          the UI since the last export is overwritten, and the change is not
          undoable. Export first.
        </div>
      </div>

      <div className="card">
        <h2>Where things live</h2>
        <table>
          <thead>
            <tr>
              <th>What</th>
              <th>Where</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Live configuration</td>
              <td className="mono">call_trace.db</td>
            </tr>
            <tr>
              <td>Agent definitions</td>
              <td className="mono">agent_platform/agents/*.json</td>
            </tr>
            <tr>
              <td>Tool definitions</td>
              <td className="mono">agent_platform/tools.json</td>
            </tr>
            <tr>
              <td>Worker</td>
              <td className="mono">transcribe_test.py</td>
            </tr>
            <tr>
              <td>Eval audio</td>
              <td className="mono">test_audio/</td>
            </tr>
            <tr>
              <td>Latest eval report</td>
              <td className="mono">.run/last_report.json</td>
            </tr>
          </tbody>
        </table>
        <p className="field-hint" style={{ marginTop: 8 }}>
          No auth on this API, bound to localhost. Anything that can reach port 8080
          can rewrite any prompt. Do not bind it to 0.0.0.0.
        </p>
      </div>
    </>
  );
}
