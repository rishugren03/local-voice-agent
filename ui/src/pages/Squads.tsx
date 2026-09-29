/**
 * U5 — Squads.
 *
 * A squad is an entry agent plus a set of edges. That is a directed graph, and it
 * is drawn as one, because the questions people actually ask about routing are
 * graph questions: can the entry agent reach the agent that handles billing? is
 * anything unreachable? what does a bounce loop look like?
 *
 * Nodes are draggable but positions are local to this browser — the backend has
 * no layout field, because a saved node position is a property of one person's
 * screen and would be wrong on everyone else's. The layout is a cyclic
 * arrangement seeded from the id, so the same squad always draws the same way.
 *
 * Edges to agents that do not exist are drawn dashed red instead of being hidden
 * or silently dropped. A squad pointing at a deleted agent is a real failure mode
 * and it should be obvious here, not at call time.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chip, Empty, Problems, Spinner } from "../components";

import { ApiError, type Agent, type Squad, api } from "../api";

const NODE_W = 148;
const NODE_H = 58;
const GRAPH_H = 360;

interface Placed extends Agent {
  x: number;
  y: number;
}

export default function Squads() {
  const [squads, setSquads] = useState<Squad[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Squad | null>(null);
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [version, setVersion] = useState(0);

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [problems, setProblems] = useState<string[]>([]);
  const [saved, setSaved] = useState(false);

  const graphRef = useRef<HTMLDivElement | null>(null);
  const dragging = useRef<{ id: string; dx: number; dy: number } | null>(null);

  const load = useCallback(async () => {
    try {
      const [squadList, agentList, v] = await Promise.all([
        api.listSquads(),
        api.listAgents(),
        api.version(),
      ]);
      setSquads(squadList.squads);
      setAgents(agentList.agents);
      setVersion(v.config_version);
      setSelected((current) => current ?? squadList.squads[0]?.id ?? null);
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
    const squad = squads.find((s) => s.id === selected);
    setDraft(squad ? { ...squad, edges: [...squad.edges] } : null);
    setPositions({});
    setSelectedEdge(null);
    setProblems([]);
    setSaved(false);
  }, [selected, squads]);

  // Cyclic layout around the entry agent, stable for a given set of ids. Stable
  // matters more than optimal: a graph that reshuffles every render is impossible
  // to drag things between.
  const placed = useMemo<Placed[]>(() => {
    if (!draft) return [];
    const members = new Set<string>([draft.entry_agent, ...draft.edges.map((e) => e.to)]);
    const ids = [...members];
    const cx = 520;
    const cy = GRAPH_H / 2;
    const radius = Math.min(130, 70 + ids.length * 14);

    return ids.map((id, index) => {
      const agent = agents.find((a) => a.id === id);
      const seed = positions[id];
      if (seed) return { ...(agent as Agent), ...seed };
      if (id === draft.entry_agent) {
        return { ...(agent as Agent), x: cx, y: cy - radius };
      }
      // Offset by index rather than by hash so the ordering follows the edge
      // list, which is ordered the way the user built it.
      const angle = ((index - 1) / Math.max(1, ids.length - 1)) * Math.PI * 1.6 + 0.5;
      return {
        ...(agent as Agent),
        x: cx + Math.cos(angle) * radius * 1.7,
        y: cy + Math.sin(angle) * radius,
      };
    });
  }, [draft, agents, positions]);

  const byId = useMemo(() => new Map(placed.map((p) => [p.id, p])), [placed]);

  const orphanEdges = useMemo(() => {
    if (!draft) return [];
    return draft.edges.filter((e) => !byId.has(e.to) || !byId.has(e.from));
  }, [draft, byId]);

  async function onPointerDown(e: React.PointerEvent, id: string) {
    const rect = graphRef.current?.getBoundingClientRect();
    if (!rect) return;
    const node = byId.get(id);
    if (!node) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    dragging.current = { id, dx: e.clientX - rect.left - node.x, dy: e.clientY - rect.top - node.y };
  }

  function onPointerMove(e: React.PointerEvent) {
    const drag = dragging.current;
    const rect = graphRef.current?.getBoundingClientRect();
    if (!drag || !rect) return;
    // Clamp to the canvas so a node cannot be dragged out of reach and
    // effectively lost until the page reloads.
    const x = Math.max(4, Math.min(rect.width - NODE_W - 4, e.clientX - rect.left - drag.dx));
    const y = Math.max(4, Math.min(GRAPH_H - NODE_H - 4, e.clientY - rect.top - drag.dy));
    setPositions((prev) => ({ ...prev, [drag.id]: { x, y } }));
  }

  function onPointerUp() {
    dragging.current = null;
  }

  function addEdge() {
    if (!draft) return;
    const target = agents.find((a) => a.id !== draft.entry_agent && !draft.edges.some((e) => e.to === a.id));
    if (!target) return;
    setDraft({
      ...draft,
      edges: [...draft.edges, { from: draft.entry_agent, to: target.id }],
    });
  }

  function removeEdge(index: number) {
    if (!draft) return;
    setDraft({ ...draft, edges: draft.edges.filter((_, i) => i !== index) });
  }

  async function save() {
    if (!draft) return;
    setSaving(true);
    setProblems([]);
    try {
      const result = await api.putSquad({
        id: draft.id,
        name: draft.name,
        entry_agent: draft.entry_agent,
        edges: draft.edges.map((e) => ({ from: e.from, to: e.to })),
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

  if (loading) return <Spinner />;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Squads</h1>
          <p>
            A squad routes a call between assistants. The entry agent starts the
            conversation; edges say who it can hand off to.
          </p>
        </div>
        <div className="row tight">
          <Chip mono tone="neutral">
            config v{version}
          </Chip>
          <button
            className="btn sm"
            onClick={async () => {
              const id = `squad-${squads.length + 1}`;
              try {
                await api.putSquad({
                  id,
                  name: "New squad",
                  entry_agent: agents[0]?.id ?? "",
                  edges: [],
                });
                setSelected(id);
                await load();
              } catch (e) {
                setProblems(e instanceof ApiError ? e.problems : [String(e)]);
              }
            }}
          >
            + New
          </button>
        </div>
      </div>

      <Problems problems={problems} />

      {squads.length === 0 ? (
        <div className="card">
          <Empty>No squads. A single-assistant setup does not need one.</Empty>
        </div>
      ) : (
        <>
          <div className="card">
            <div className="row tight" style={{ marginBottom: 14 }}>
              {squads.map((squad) => (
                <button
                  key={squad.id}
                  className={`btn sm ${squad.id === selected ? "primary" : ""}`}
                  onClick={() => setSelected(squad.id)}
                >
                  {squad.name}
                </button>
              ))}
            </div>

            {draft && (
              <>
                <div className="grid-2" style={{ marginBottom: 14 }}>
                  <label className="field">
                    <span className="field-label">Name</span>
                    <input
                      type="text"
                      value={draft.name}
                      onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                    />
                  </label>
                  <label className="field">
                    <span className="field-label">Entry agent</span>
                    <select
                      value={draft.entry_agent}
                      onChange={(e) => setDraft({ ...draft, entry_agent: e.target.value })}
                    >
                      {agents.map((agent) => (
                        <option key={agent.id} value={agent.id}>
                          {agent.id}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                {orphanEdges.length > 0 && (
                  <Problems
                    problems={[
                      `This squad points at ${[
                        ...new Set(orphanEdges.flatMap((e) => [e.from, e.to])),
                      ]
                        .filter((id) => !agents.some((a) => a.id === id))
                        .join(", ")}, which no longer exist. Calls routed through this squad will fail.`,
                    ]}
                  />
                )}

                <div
                  className="graph"
                  ref={graphRef}
                  onPointerMove={onPointerMove}
                  onPointerUp={onPointerUp}
                  onPointerLeave={onPointerUp}
                >
                  <svg className="edges">
                    {draft.edges.map((edge, i) => {
                      const from = byId.get(edge.from);
                      const to = byId.get(edge.to);
                      if (!from || !to) return null;
                      const key = `${edge.from}->${edge.to}`;
                      const orphan = !agents.some((a) => a.id === edge.to);
                      return (
                        <g key={`${key}-${i}`}>
                          <line
                            x1={from.x + NODE_W / 2}
                            y1={from.y + NODE_H}
                            x2={to.x + NODE_W / 2}
                            y2={to.y}
                            className={`${selectedEdge === key ? "selected" : ""} ${orphan ? "orphan-edge" : ""}`}
                            onClick={() => setSelectedEdge(selectedEdge === key ? null : key)}
                            style={{ pointerEvents: "stroke", cursor: "pointer", strokeWidth: 3 }}
                          />
                        </g>
                      );
                    })}
                  </svg>

                  {placed.map((node) => (
                    <div
                      key={node.id}
                      className={`node ${node.id === draft.entry_agent ? "entry" : ""} ${!agents.some((a) => a.id === node.id) ? "orphan" : ""}`}
                      style={{ left: node.x, top: node.y }}
                      onPointerDown={(e) => onPointerDown(e, node.id)}
                    >
                      <div className="name">{node.id}</div>
                      <div className="role">
                        {node.id === draft.entry_agent
                          ? "entry"
                          : !agents.some((a) => a.id === node.id)
                            ? "missing"
                            : `${node.tools.length} tools`}
                      </div>
                    </div>
                  ))}
                </div>

                <p className="field-hint" style={{ marginTop: 6 }}>
                  Drag to arrange. Positions are local to this browser and are not saved.
                  Click an edge to select it.
                </p>

                {selectedEdge && (
                  <div className="row" style={{ marginTop: 10 }}>
                    <span className="mono" style={{ fontSize: 12.5 }}>
                      {selectedEdge}
                    </span>
                    <button
                      className="btn danger sm"
                      onClick={() => {
                        const index = draft.edges.findIndex(
                          (e) => `${e.from}->${e.to}` === selectedEdge,
                        );
                        if (index >= 0) removeEdge(index);
                        setSelectedEdge(null);
                      }}
                    >
                      Remove edge
                    </button>
                  </div>
                )}

                <div className="row" style={{ marginTop: 14 }}>
                  <button className="btn" onClick={addEdge}>
                    + Add edge
                  </button>
                  <button className="btn primary" onClick={save} disabled={saving}>
                    {saving ? "Saving…" : "Save squad"}
                  </button>
                  {saved && <Chip tone="ok">saved</Chip>}
                  <button
                    className="btn danger sm"
                    onClick={async () => {
                      if (!confirm(`Delete squad "${draft.id}"?`)) return;
                      try {
                        await api.deleteSquad(draft.id);
                        setSelected(null);
                        await load();
                      } catch (e) {
                        setProblems(e instanceof ApiError ? e.problems : [String(e)]);
                      }
                    }}
                  >
                    Delete
                  </button>
                </div>

                <table style={{ marginTop: 14 }}>
                  <thead>
                    <tr>
                      <th>From</th>
                      <th>To</th>
                      <th style={{ width: 90 }} />
                    </tr>
                  </thead>
                  <tbody>
                    {draft.edges.map((edge, i) => (
                      <tr key={i}>
                        <td className="mono">{edge.from}</td>
                        <td className="mono">
                          {edge.to}{" "}
                          {!agents.some((a) => a.id === edge.to) && (
                            <Chip tone="error">missing</Chip>
                          )}
                        </td>
                        <td>
                          <button className="btn sm danger" onClick={() => removeEdge(i)}>
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                    {draft.edges.length === 0 && (
                      <tr>
                        <td colSpan={3} className="faint" style={{ textAlign: "center" }}>
                          No edges — the entry agent handles everything.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </>
            )}
          </div>

          <div className="card">
            <h2>Handoff visibility</h2>
            <p className="hint">
              Handoffs are also declared per assistant. An edge here and a missing
              handoff tool on the assistant mean different things, so both are shown.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Assistant</th>
                  <th>Declared handoffs</th>
                </tr>
              </thead>
              <tbody>
                {agents.map((agent) => (
                  <tr key={agent.id}>
                    <td className="mono">{agent.id}</td>
                    <td>
                      {agent.handoffs.length === 0 ? (
                        <span className="faint">none</span>
                      ) : (
                        <div className="row tight">
                          {agent.handoffs.map((h) => (
                            <Chip key={h} mono tone="accent">
                              → {h}
                            </Chip>
                          ))}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
