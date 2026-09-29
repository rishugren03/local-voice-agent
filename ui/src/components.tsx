/** Small building blocks shared by every screen. */

import type { ReactNode } from "react";
import type { Health, Latency, ServiceCheck } from "./api";

/** A status pill. `ok`/`warn`/`error` map to the palette in styles.css. */
export function Chip({
  tone = "neutral",
  children,
  dot,
  mono,
}: {
  tone?: "ok" | "warn" | "error" | "accent" | "neutral";
  children: ReactNode;
  dot?: boolean;
  mono?: boolean;
}) {
  return (
    <span className={`chip ${tone === "neutral" ? "" : tone} ${mono ? "mono" : ""}`}>
      {dot && <i className="dot" />}
      {children}
    </span>
  );
}

export function healthTone(status: string): "ok" | "warn" | "error" {
  if (status === "ok") return "ok";
  if (status === "down") return "error";
  return "warn";
}

/**
 * The always-visible service banner.
 *
 * It shows the fix, not the status: "Ollama is not reachable" is a fact the
 * user cannot act on, while "start it with `ollama serve`" is. The backend
 * writes those strings, so this component never invents its own advice.
 */
export function HealthBanner({ health }: { health: Health | null }) {
  if (!health) {
    return (
      <div className="health-banner unknown">
        <i className="dot" />
        <span>Checking services…</span>
      </div>
    );
  }

  if (health.status === "ok") {
    const count = Object.keys(health.services).length;
    return (
      <div className="health-banner ok">
        <i className="dot" />
        <span>
          All {count} services ready.
        </span>
      </div>
    );
  }

  const broken = health.down.length ? health.down : health.unknown;
  return (
    <div className={`health-banner ${health.status}`}>
      <i className="dot" />
      <div>
        <strong>
          {health.down.length ? `${health.down.length} service` : "Some services"}
          {health.down.length > 1 ? "s" : ""} down:
        </strong>{" "}
        {broken.join(", ")}
        <div className="fix">
          {broken.map((name) => {
            const check = health.services[name];
            if (!check?.fix) return null;
            return (
              <div key={name}>
                <strong>{name}</strong>: {check.fix}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/**
 * Per-service rows for the System screen, where each check gets its own row
 * rather than being collapsed into the banner.
 */
export function ServiceRow({ name, check }: { name: string; check: ServiceCheck }) {
  return (
    <tr>
      <td style={{ fontWeight: 550 }}>{name}</td>
      <td style={{ width: 90 }}>
        <Chip tone={healthTone(check.status)} dot>
          {check.status}
        </Chip>
      </td>
      <td>
        {check.detail}
        {check.fix && (
          <div className="faint" style={{ fontSize: 12, marginTop: 2 }}>
            Fix: {check.fix}
          </div>
        )}
      </td>
    </tr>
  );
}

/**
 * A stacked latency bar, scaled to a fixed budget.
 *
 * Fixed rather than relative: if the only calls on screen took 0.6s and 0.7s,
 * a relative scale would draw two full-width bars and make a fast system look
 * broken. The budget is the reference that makes the colour mean something —
 * over it is a warning regardless of what else is on screen.
 */
const BUDGET_S = 6;

export function LatencyBar({ latency, showTotal = true }: { latency: Latency; showTotal?: boolean }) {
  const total = latency.total;
  if (total === null || total === undefined) {
    return <span className="latency-unknown">—</span>;
  }

  const pct = (v: number) => `${Math.min(100, (v / BUDGET_S) * 100)}%`;
  const tone = total > BUDGET_S ? "bad" : total > 4 ? "slow" : "";

  return (
    <div className="latency" title={`stt ${latency.stt ?? "?"}s · llm ${latency.llm ?? "?"}s · tts ${latency.tts ?? "?"}s`}>
      <div className="latency-bar">
        {latency.stt !== null && <span className="stt" style={{ width: pct(latency.stt) }} />}
        {latency.llm !== null && <span className="llm" style={{ width: pct(latency.llm) }} />}
        {latency.tts !== null && <span className="tts" style={{ width: pct(latency.tts) }} />}
      </div>
      {showTotal && <span className={`latency-total ${tone}`}>{total.toFixed(2)}s</span>}
    </div>
  );
}

/**
 * The validation result block.
 *
 * Renders the backend's `problems` verbatim. Rewriting them into friendlier
 * text here would mean two sets of wording to keep in sync, and the one that
 * drifts is always the copy that has been through a translation.
 */
export function Problems({ problems }: { problems: string[] }) {
  if (problems.length === 0) return null;
  return (
    <div className="problems">
      <strong>
        {problems.length === 1 ? "This cannot be saved:" : `${problems.length} problems:`}
      </strong>
      <ul>
        {problems.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
    </div>
  );
}

export function Spinner() {
  return <span className="spinner" aria-label="loading" />;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
