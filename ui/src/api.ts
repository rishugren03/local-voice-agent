/**
 * Typed client for the control plane.
 *
 * Every error the API can produce is turned into an ApiError carrying the
 * `problems` list, because the backend deliberately returns the same shape for
 * a domain failure ("agent 'primary' lists tool 'nope'") and a malformed request
 * ("id: must match pattern"). The UI renders that list as-is, which means the
 * messages a user reads are written by the validator that rejected the config —
 * not restated here, and not able to drift from it.
 */

// Empty by default, so requests go to the same origin. In development that is
// the Vite dev server, which proxies /api and /health to 127.0.0.1:8080
// (see vite.config.ts) — one hop, and errors surface as a clean proxy failure
// rather than a CORS message that hides the real status.
//
// Set VITE_API_BASE to an absolute URL to point the built bundle at a control
// plane somewhere else. The API allows 5173 and 3000 by default, so a dev server
// on either port can also just use http://127.0.0.1:8080 directly.
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  readonly problems: string[];
  readonly status: number;

  constructor(status: number, problems: string[]) {
    super(problems[0] ?? `Request failed with ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.problems = problems;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    // A connection failure is the single most common thing to go wrong on a
    // local setup, and "Failed to fetch" does not tell the user the API is not
    // running. Say so, and say how to start it. The underlying TypeError is not
    // in the message: it is always "network error", which is the information the
    // user already had.
    throw new ApiError(0, [
      `Could not reach the control plane${BASE ? ` at ${BASE}` : ""}. Is it running? Start it with: python3 -m control_plane.app`,
    ]);
  }

  if (!response.ok) {
    let problems: string[] = [];
    try {
      const body = await response.json();
      if (Array.isArray(body?.problems)) problems = body.problems;
      else if (typeof body?.detail === "string") problems = [body.detail];
      else if (Array.isArray(body?.detail))
        problems = body.detail.map((d: { msg?: string }) => d.msg ?? String(d));
    } catch {
      // A non-JSON error body (a proxy's HTML 502, say) is still worth showing.
    }
    throw new ApiError(response.status, problems.length ? problems : [`HTTP ${response.status}`]);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// -- types ------------------------------------------------------------------

export interface Agent {
  id: string;
  system_prompt: string;
  tools: string[];
  handoffs: string[];
  rules: string[];
  llm_model?: string | null;
  tts_voice?: string | null;
}

export interface Tool {
  name: string;
  description: string;
  args_schema?: Record<string, unknown>;
  is_handoff?: boolean;
  target?: string;
  enabled?: boolean;
}

export interface SquadEdge {
  from: string;
  to: string;
}

export interface Squad {
  id: string;
  name: string;
  entry_agent: string;
  edges: SquadEdge[];
}

export interface Latency {
  stt: number | null;
  llm: number | null;
  tts: number | null;
  total: number | null;
}

export interface SessionSummary {
  session_id: string;
  participant_identity: string | null;
  started_at: string | null;
  ended_at: string | null;
  turns_served: number | null;
  barge_ins: number | null;
  final_agent: string | null;
  n_calls: number;
  errors: number;
  latency: Latency;
}

export interface SessionDetail {
  session_id: string;
  participant_identity: string | null;
  started_at: string | null;
  ended_at: string | null;
  turns_served: number | null;
  barge_ins: number | null;
  final_agent: string | null;
  n_calls: number;
  errors: number;
  latency: Latency;
  calls: CallSummary[];
  transitions: { from_agent: string | null; to_agent: string; reason: string | null; started_at: string }[];
}

export interface CallSummary {
  call_id: string;
  started_at: string | null;
  first_agent: string | null;
  user_text: string | null;
  reply: string | null;
  tools: string[];
  errors: string[];
  degraded: boolean;
  latency: Latency;
  transitions: { from: string | null; to: string; reason: string | null; at: string }[];
}

export interface CallDetail extends CallSummary {
  session_id: string | null;
  participant_identity: string | null;
  events: Record<string, unknown>[];
}

export interface ServiceCheck {
  status: "ok" | "down" | "unknown";
  detail: string;
  fix: string;
  [key: string]: unknown;
}

export interface Health {
  status: "ok" | "down" | "unknown";
  down: string[];
  unknown: string[];
  services: Record<string, ServiceCheck>;
}

export interface ConfigVersion {
  config_version: number;
  agents: number;
  tools: number;
  squads: number;
}

// -- evals ------------------------------------------------------------------

export interface EvalCase {
  case: string;
  status: string;
  call_id: string | null;
  match_mode: string;
  detail: string;
}

export interface LatencyStats {
  stt_p50: number | null;
  llm_p50: number | null;
  tts_p50: number | null;
  total_p50: number | null;
  total_p90: number | null;
}

export interface EvalSession {
  session_id: string;
  identity: string | null;
  known_cases: boolean;
  n_calls: number;
  cases_expected: number;
  passed: number;
  results: EvalCase[];
  unmatched_calls: string[];
  unmatched_detail: { call_id: string; heard: string }[];
  rejected_silent_turns: { audio_s: number; peak: number }[];
  session_end?: { final_agent: string | null; turns_served: number; barge_ins: number };
  latency: LatencyStats;
}

export interface EvalReport {
  window: { start: number; end: number } | null;
  n_sessions: number;
  n_calls: number;
  passed: number;
  total: number;
  sessions: EvalSession[];
  latency_overall: LatencyStats;
}

export interface EvalJob {
  id: string;
  state: "running" | "done" | "failed";
  started_at: number;
  finished_at: number | null;
  users: number;
  rotate: number;
  stage: string;
  log: string[];
  error: string | null;
  report: EvalReport | null;
}

// -- endpoints --------------------------------------------------------------

export const api = {
  health: () => request<Health>("/health"),
  version: () => request<ConfigVersion>("/api/config/version"),
  validate: () =>
    request<{ ok: boolean; problems: string[]; config_version: number }>("/api/config/validate"),

  listAgents: () => request<{ agents: Agent[]; config_version: number }>("/api/agents"),
  getAgent: (id: string) => request<Agent>(`/api/agents/${id}`),
  putAgent: (id: string, agent: Partial<Agent> & { id: string }) =>
    request<{ agent: Agent; config_version: number }>(`/api/agents/${id}`, {
      method: "PUT",
      body: JSON.stringify(agent),
    }),
  patchAgent: (id: string, patch: Partial<Agent> & { id: string }) =>
    request<{ agent: Agent; config_version: number }>(`/api/agents/${id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  deleteAgent: (id: string) =>
    request<{ deleted: string; config_version: number }>(`/api/agents/${id}`, { method: "DELETE" }),
  promptPreview: (id: string, userText: string) =>
    request<{ prompt: string }>(
      `/api/agents/${id}/prompt-preview?user_text=${encodeURIComponent(userText)}`,
    ),

  listTools: () => request<{ tools: Tool[]; config_version: number }>("/api/tools"),
  putTool: (name: string, tool: Partial<Tool> & { name: string }) =>
    request<{ tool: Tool; config_version: number }>(`/api/tools/${name}`, {
      method: "PUT",
      body: JSON.stringify(tool),
    }),
  deleteTool: (name: string) =>
    request<{ deleted: string; config_version: number }>(`/api/tools/${name}`, { method: "DELETE" }),

  listSquads: () => request<{ squads: Squad[]; config_version: number }>("/api/squads"),
  putSquad: (squad: Squad) =>
    request<{ squad: Squad; config_version: number }>(`/api/squads/${squad.id}`, {
      method: "PUT",
      body: JSON.stringify(squad),
    }),
  deleteSquad: (id: string) =>
    request<{ deleted: string; config_version: number }>(`/api/squads/${id}`, { method: "DELETE" }),

  importConfig: () =>
    request<{ agents: number; tools: number; config_version: number }>("/api/config/import", {
      method: "POST",
    }),
  exportConfig: () =>
    request<{ agents_dir: string; tools_file: string; config_version: number }>(
      "/api/config/export",
      { method: "POST" },
    ),

  listSessions: (limit = 50, offset = 0, search = "") =>
    request<{ sessions: SessionSummary[]; total: number; limit: number; offset: number }>(
      `/api/sessions?limit=${limit}&offset=${offset}${search ? `&search=${encodeURIComponent(search)}` : ""}`,
    ),
  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),
  getCall: (id: string) => request<CallDetail>(`/api/calls/${id}`),

  mintToken: (identity?: string) =>
    request<{ token: string; url: string; room: string; identity: string; expires_in: number }>(
      "/api/calls/token",
      { method: "POST", body: JSON.stringify({ identity }) },
    ),

  evals: () =>
    request<{
      report: EvalReport | null;
      job: EvalJob | null;
      preflight: string[];
    }>("/api/evals"),
  startEval: (users: number, rotate: number) =>
    request<EvalJob>("/api/evals", {
      method: "POST",
      body: JSON.stringify({ users, rotate }),
    }),
};
