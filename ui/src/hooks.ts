/**
 * Hooks, kept apart from the components.
 *
 * They live in their own file because a module that exports both components and
 * other things defeats React Fast Refresh: editing a component in a mixed file
 * reloads the whole page instead of hot-swapping, which on the Call Logs screen
 * means losing the session you were reading.
 */

import { useEffect, useState } from "react";

/** Debounces a rapidly-changing value, so typing in a search box is not a
 * request per keystroke against a local API. */
export function useDebounced<T>(value: T, ms = 250): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return debounced;
}

/** Polls a callback on an interval, pausing while the tab is hidden.
 *
 * Pausing matters here: the Calls screen polls the trace database, and a
 * background tab that keeps polling is a background tab that holds a SQLite
 * connection open against the worker's writes.
 */
export function usePoll(fn: () => void | Promise<void>, intervalMs: number, enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    const tick = async () => {
      if (document.hidden || cancelled) return;
      try {
        await fn();
      } catch {
        // A failed poll is not worth surfacing on its own; the next tick will
        // either succeed or the page's own error state will show it.
      }
    };

    const timer = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [fn, intervalMs, enabled]);
}
