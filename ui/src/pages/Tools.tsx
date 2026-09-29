/**
 * U4 — Tools.
 *
 * Tool definitions as JSON, because that is what they are. A schema builder form
 * would be more clickable and would also be wrong more often: the argument
 * schema is passed to the model verbatim, and the interesting mistakes —
 * `"required": ["expression"]` left off, a property typed `string` where the
 * handler wants a number — are far easier to see as text than as a set of
 * dropdowns. The validator runs on every keystroke, so the error tells you which
 * field is wrong before you save.
 */

import { useCallback, useEffect, useState } from "react";
import { Chip, Problems, Spinner } from "../components";

import { ApiError, type Tool, api } from "../api";

const EXAMPLES = [
  {
    name: "calculate",
    description: "Evaluate an arithmetic expression and return the result.",
    args_schema: {
      type: "object",
      properties: { expression: { type: "string", description: "e.g. 12 + 15" } },
      required: ["expression"],
    },
  },
  {
    name: "get_weather",
    description: "Current weather for a city.",
    args_schema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
];

export default function Tools() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Tool | null>(null);
  const [version, setVersion] = useState(0);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    try {
      const [result, v] = await Promise.all([api.listTools(), api.version()]);
      setTools(result.tools);
      setVersion(v.config_version);
      setSelected((current) => current ?? result.tools[0]?.name ?? null);
    } catch (e) {
      setProblems(e instanceof ApiError ? e.problems : [String(e)]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const tool = tools.find((t) => t.name === selected);
    setDraft(tool ? { ...tool } : null);
    setProblems([]);
    setSaved(false);
  }, [selected, tools]);

  const [schemaText, setSchemaText] = useState("{}");

  useEffect(() => {
    setSchemaText(JSON.stringify(draft?.args_schema ?? {}, null, 2));
  }, [selected, draft?.name]);

  // Parsed on every keystroke so a broken schema is underlined while typing
  // rather than rejected on save, where it would be one step removed from the
  // mistake.
  const parsedSchema = (): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(schemaText);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        setJsonError("The schema must be a JSON object.");
        return null;
      }
      setJsonError(null);
      return parsed as Record<string, unknown>;
    } catch (e) {
      setJsonError((e as Error).message);
      return null;
    }
  };

  async function save() {
    if (!draft) return;
    const schema = parsedSchema();
    if (schema === null) return;

    setSaving(true);
    setProblems([]);
    try {
      const result = await api.putTool(draft.name, {
        name: draft.name,
        description: draft.description,
        args_schema: schema,
        is_handoff: draft.is_handoff ?? false,
        target: draft.target,
        enabled: draft.enabled ?? true,
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

  function create(name: string) {
    const example = EXAMPLES.find((e) => e.name === name) ?? EXAMPLES[0];
    api
      .putTool(example.name, example)
      .then(async () => {
        setSelected(example.name);
        await load();
      })
      .catch((e) => setProblems(e instanceof ApiError ? e.problems : [String(e)]));
  }

  if (loading) return <Spinner />;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Tools</h1>
          <p>
            The functions an assistant may call. The name and description are what the
            model sees, so they matter more than the implementation.
          </p>
        </div>
        <div className="row tight">
          <Chip mono tone="neutral">
            config v{version}
          </Chip>
          <button className="btn sm" onClick={() => create("calculate")}>
            + New tool
          </button>
        </div>
      </div>

      <Problems problems={problems} />

      <div className="split">
        <div className="card">
          <div className="list">
            {tools.map((tool) => (
              <button
                key={tool.name}
                className={`list-item ${tool.name === selected ? "active" : ""}`}
                onClick={() => setSelected(tool.name)}
              >
                <div className="spread">
                  <span className="mono">{tool.name}</span>
                  {tool.is_handoff ? (
                    <Chip tone="accent">handoff</Chip>
                  ) : tool.enabled === false ? (
                    <Chip tone="warn">disabled</Chip>
                  ) : null}
                </div>
                <span className="sub truncate">{tool.description}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          {!draft ? (
            <div className="card">
              <Empty>No tools defined.</Empty>
            </div>
          ) : (
            <div className="card">
              <div className="spread">
                <h2 className="mono" style={{ margin: 0 }}>
                  {draft.name}
                </h2>
                <div className="row tight">
                  {saved && <Chip tone="ok">saved</Chip>}
                  <button className="btn primary" onClick={save} disabled={saving || !!jsonError}>
                    {saving ? "Saving…" : "Save"}
                  </button>
                  {!draft.is_handoff && (
                    <button
                      className="btn danger sm"
                      onClick={async () => {
                        if (!confirm(`Delete tool "${draft.name}"?`)) return;
                        try {
                          await api.deleteTool(draft.name);
                          setSelected(null);
                          await load();
                        } catch (e) {
                          setProblems(e instanceof ApiError ? e.problems : [String(e)]);
                        }
                      }}
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>

              {draft.is_handoff && (
                <div className="notice">
                  This is a generated handoff tool pointing at{" "}
                  <span className="mono">{draft.target}</span>. It is created and removed
                  automatically by the Hand off to checkboxes on the Assistants screen,
                  and cannot be edited here.
                </div>
              )}

              <label className="field">
                <span className="field-label">Description</span>
                <textarea
                  rows={2}
                  value={draft.description}
                  onChange={(e) => setDraft({ ...draft, description: e.target.value })}
                />
                <p className="field-hint">
                  The only thing the model has to go on when deciding to call this. Say
                  what it does and when to use it, not how it is implemented.
                </p>
              </label>

              <label className="field">
                <span className="field-label">Arguments (JSON Schema)</span>
                <textarea
                  rows={10}
                  className={`code ${jsonError ? "invalid" : ""}`}
                  value={schemaText}
                  onChange={(e) => setSchemaText(e.target.value)}
                  spellCheck={false}
                />
                {jsonError && (
                  <p className="field-hint" style={{ color: "var(--error)" }}>
                    {jsonError}
                  </p>
                )}
              </label>

              <label className="check">
                <input
                  type="checkbox"
                  checked={draft.enabled !== false}
                  onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })}
                />
                <span>Enabled</span>
              </label>
              <p className="field-hint" style={{ marginTop: 2 }}>
                A disabled tool stays in the registry but is not offered to the model.
                Useful for switching behaviour without deleting work.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="empty">{children}</p>;
}
