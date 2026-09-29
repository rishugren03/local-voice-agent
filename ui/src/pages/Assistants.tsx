/**
 * U1 — Assistants.
 *
 * The screen exists to make one loop fast: change a prompt, see exactly what the
 * model will see, start a call, hear the difference. Three decisions follow from
 * that:
 *
 *  - The prompt preview is next to the editor, not behind a button. A preview
 *    you have to go looking for is a preview you stop checking.
 *  - Tools and handoffs are checkboxes against the real registry. A dropdown
 *    free-text field lets you save a typo that only fails on the next call.
 *  - Saving is explicit and per-agent, and the version number is shown, because
 *    the worker picks changes up at the *next* session. Without that on screen,
 *    "saved" reads as "live" and the first call after a save looks like a bug.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Chip, Problems, Spinner } from "../components";

import { ApiError, type Agent, type Tool, api } from "../api";

interface Draft {
  id: string;
  system_prompt: string;
  tools: string[];
  handoffs: string[];
  rules: string[];
  llm_model: string;
  tts_voice: string;
}

function toDraft(agent: Agent): Draft {
  return {
    id: agent.id,
    system_prompt: agent.system_prompt,
    tools: agent.tools ?? [],
    handoffs: agent.handoffs ?? [],
    rules: agent.rules ?? [],
    llm_model: agent.llm_model ?? "",
    tts_voice: agent.tts_voice ?? "",
  };
}

export default function Assistants() {
  const navigate = useNavigate();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [tools, setTools] = useState<Tool[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [version, setVersion] = useState(0);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [saved, setSaved] = useState(false);

  const [previewText, setPreviewText] = useState("What is 12 plus 15?");
  const [preview, setPreview] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [creating, setCreating] = useState(false);
  const [newId, setNewId] = useState("");

  const load = useCallback(async () => {
    try {
      const [agentList, toolList, v] = await Promise.all([
        api.listAgents(),
        api.listTools(),
        api.version(),
      ]);
      setAgents(agentList.agents);
      setTools(toolList.tools);
      setVersion(v.config_version);
      setSelected((current) => current ?? agentList.agents[0]?.id ?? null);
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Re-seed the editor whenever the selection changes, and discard an unsaved
  // edit silently switching away — which would otherwise look like the app lost
  // the user's work.
  useEffect(() => {
    const agent = agents.find((a) => a.id === selected);
    setDraft(agent ? toDraft(agent) : null);
    setProblems([]);
    setSaved(false);
  }, [selected, agents]);

  // Real tools only: handoff tools are generated from the handoff list, so
  // listing them here as checkboxes would let the user set a tool and a handoff
  // that contradict each other.
  const callableTools = useMemo(() => tools.filter((t) => !t.is_handoff), [tools]);

  const dirty = useMemo(() => {
    if (!draft) return false;
    const original = agents.find((a) => a.id === draft.id);
    return original ? JSON.stringify(toDraft(original)) !== JSON.stringify(draft) : false;
  }, [draft, agents]);

  const refreshPreview = useCallback(async () => {
    if (!draft) return;
    try {
      const result = await api.promptPreview(draft.id, previewText);
      setPreview(result.prompt);
      setPreviewError(null);
    } catch (e) {
      // A preview failure is expected while the agent is half-edited — the
      // backend validates the saved config, and a new handoff target that does
      // not exist yet has no tool to render. Shown inline, not as a page error.
      setPreview(null);
      setPreviewError(e instanceof ApiError ? e.problems.join(" ") : String(e));
    }
  }, [draft, previewText]);

  useEffect(() => {
    if (!draft) return;
    const timer = setTimeout(() => void refreshPreview(), 200);
    return () => clearTimeout(timer);
  }, [draft, refreshPreview]);

  async function save() {
    if (!draft) return;
    setSaving(true);
    setProblems([]);
    try {
      // PUT, not PATCH: the editor holds the whole agent, and replacing is what
      // makes "unchecking a tool removes it" true. A merge would keep tools the
      // user just removed until the process restarted.
      const result = await api.putAgent(draft.id, {
        id: draft.id,
        system_prompt: draft.system_prompt,
        tools: draft.tools,
        handoffs: draft.handoffs,
        rules: draft.rules,
        llm_model: draft.llm_model || null,
        tts_voice: draft.tts_voice || null,
      });
      setVersion(result.config_version);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
      await load();
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    } finally {
      setSaving(false);
    }
  }

  async function remove() {
    if (!draft) return;
    if (!confirm(`Delete agent "${draft.id}"? Any squad pointing at it will be left with a broken entry point.`)) {
      return;
    }
    try {
      await api.deleteAgent(draft.id);
      setSelected(null);
      await load();
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    }
  }

  function toggle(list: "tools" | "handoffs", value: string) {
    if (!draft) return;
    setDraft({
      ...draft,
      [list]: draft[list].includes(value)
        ? draft[list].filter((v) => v !== value)
        : [...draft[list], value],
    });
  }

  if (loading) return <Spinner />;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Assistants</h1>
          <p>
            Each assistant is a system prompt, a set of tools it may call, and the
            other assistants it can hand off to. Changes take effect on the next
            call.
          </p>
        </div>
        <div className="row tight">
          <Chip tone="neutral" mono>
            config v{version}
          </Chip>
          <button className="btn sm" onClick={() => setCreating(true)}>
            + New
          </button>
        </div>
      </div>

      {creating && (
        <div className="card">
          <h2>New assistant</h2>
          <p className="hint">
            The id becomes a filename on export and a URL, so letters, digits,{" "}
            <code>-</code> and <code>_</code> only.
          </p>
          <div className="row">
            <input
              type="text"
              placeholder="billing"
              value={newId}
              autoFocus
              onChange={(e) => setNewId(e.target.value)}
              style={{ maxWidth: 260 }}
            />
            <button
              className="btn primary"
              disabled={!newId.trim()}
              onClick={async () => {
                const id = newId.trim();
                setCreating(false);
                setNewId("");
                try {
                  await api.putAgent(id, {
                    id,
                    system_prompt: "",
                    tools: [],
                    handoffs: [],
                    rules: [],
                  });
                  setSelected(id);
                  await load();
                } catch (e) {
                  setProblems(e instanceof ApiError ? e.problems : [String(e)]);
                }
              }}
            >
              Create
            </button>
            <button className="btn" onClick={() => setCreating(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="split">
        <div className="card">
          <div className="list">
            {agents.map((agent) => (
              <button
                key={agent.id}
                className={`list-item ${agent.id === selected ? "active" : ""}`}
                onClick={() => setSelected(agent.id)}
              >
                {agent.id}
                <span className="sub">
                  {agent.tools.length} tool{agent.tools.length === 1 ? "" : "s"} ·{" "}
                  {agent.handoffs.length} handoff{agent.handoffs.length === 1 ? "" : "s"}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div>
          {!draft ? (
            <div className="card">
              <p className="empty">No assistants yet.</p>
            </div>
          ) : (
            <>
              <div className="card">
                <div className="spread">
                  <h2>{draft.id}</h2>
                  <div className="row tight">
                    {dirty && <Chip tone="warn">unsaved</Chip>}
                    {saved && <Chip tone="ok">saved</Chip>}
                    <button className="btn primary" onClick={save} disabled={saving || !dirty}>
                      {saving ? "Saving…" : "Save"}
                    </button>
                    <button className="btn danger sm" onClick={remove}>
                      Delete
                    </button>
                  </div>
                </div>

                <Problems problems={problems} />

                <label className="field">
                  <span className="field-label">System prompt</span>
                  <textarea
                    rows={5}
                    value={draft.system_prompt}
                    onChange={(e) => setDraft({ ...draft, system_prompt: e.target.value })}
                  />
                  <p className="field-hint">
                    The task definition. A small local model has nothing to fall back on
                    if this is vague.
                  </p>
                </label>

                <label className="field">
                  <span className="field-label">Rules</span>
                  <textarea
                    rows={3}
                    className="code"
                    placeholder={"One rule per line. e.g.\nAlways use the calculate tool for arithmetic."}
                    value={draft.rules.join("\n")}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        rules: e.target.value.split("\n").filter((line) => line.trim()),
                      })
                    }
                  />
                  <p className="field-hint">
                    Hard constraints appended verbatim to the prompt. These are what stop a
                    4B model deciding to "just answer" instead of calling a tool.
                  </p>
                </label>

                <div className="grid-2">
                  <div>
                    <span className="field-label">Tools</span>
                    <div className="list" style={{ marginTop: 4 }}>
                      {callableTools.map((tool) => (
                        <label key={tool.name} className="check">
                          <input
                            type="checkbox"
                            checked={draft.tools.includes(tool.name)}
                            onChange={() => toggle("tools", tool.name)}
                          />
                          <span className="mono">{tool.name}</span>
                        </label>
                      ))}
                      {callableTools.length === 0 && (
                        <span className="faint" style={{ fontSize: 12.5 }}>
                          No tools registered.
                        </span>
                      )}
                    </div>
                  </div>

                  <div>
                    <span className="field-label">Hand off to</span>
                    <div className="list" style={{ marginTop: 4 }}>
                      {agents
                        .filter((a) => a.id !== draft.id)
                        .map((agent) => (
                          <label key={agent.id} className="check">
                            <input
                              type="checkbox"
                              checked={draft.handoffs.includes(agent.id)}
                              onChange={() => toggle("handoffs", agent.id)}
                            />
                            <span className="mono">{agent.id}</span>
                          </label>
                        ))}
                    </div>
                    <p className="field-hint">
                      Each handoff becomes a <code>handoff_to_X</code> tool the model can
                      call. A handoff to this agent is rejected — it costs a turn and
                      changes nothing.
                    </p>
                  </div>
                </div>

                <div className="grid-2" style={{ marginTop: 4 }}>
                  <label className="field">
                    <span className="field-label">LLM override</span>
                    <input
                      type="text"
                      placeholder="(platform default)"
                      value={draft.llm_model}
                      onChange={(e) => setDraft({ ...draft, llm_model: e.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span className="field-label">Voice override</span>
                    <input
                      type="text"
                      placeholder="(platform default)"
                      value={draft.tts_voice}
                      onChange={(e) => setDraft({ ...draft, tts_voice: e.target.value })}
                    />
                    <p className="field-hint">
                      Must match the room's audio rate or playback fails mid-call. Checked
                      on the System screen.
                    </p>
                  </label>
                </div>
              </div>

              <div className="card">
                <h2>What the model will see</h2>
                <p className="hint">
                  Built by the same code the worker uses, so this is the real prompt — not
                  an approximation of it.
                </p>
                <div className="row" style={{ marginBottom: 10 }}>
                  <input
                    type="text"
                    value={previewText}
                    onChange={(e) => setPreviewText(e.target.value)}
                    placeholder="Type something the caller might say"
                    style={{ maxWidth: 340 }}
                  />
                  <button className="btn sm" onClick={() => navigate("/talk")}>
                    Test this in a call →
                  </button>
                </div>
                {previewError ? (
                  <div className="notice">{previewError}</div>
                ) : (
                  preview && <pre className="preview">{preview}</pre>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
