// The one way pages fetch anything besides the snapshot. A page asks for a
// URL and gets whatever is cached, possibly nothing; the fetch runs in the
// background and re-renders when it lands. Rendering is idempotent, so no
// page tracks request ownership. Live windows share one entry and in-flight
// request; only entries used by the current render can repaint it.

import { get, ui } from "./store.js";

const entries = new Map();
const FRESH_MS = 15000;
const MAX_ENTRIES = 24;
let generation = 0;
let repaint = 0;

export const beginRender = () => generation++;
export function endRender() {
  for (const [scope, entry] of entries) {
    if (entries.size <= MAX_ENTRIES) break;
    if (entry.used === generation) continue;
    entry.controller?.abort();
    entries.delete(scope);
  }
}

export function use(url, scope = url) {
  let entry = entries.get(scope);
  if (!entry) {
    entry = { url: null, data: null, error: null, errorStatus: null, fetchedAt: 0, pending: false, controller: null, used: generation };
  }
  entry.used = generation;
  entries.delete(scope);
  entries.set(scope, entry);
  if (!ui.nodeNavigating && !entry.pending && (entry.url !== url || Date.now() - entry.fetchedAt >= FRESH_MS)) {
    entry.url = url;
    entry.pending = true;
    entry.controller = new AbortController();
    get(url, entry.controller.signal)
      .then((data) => {
        entry.error = data.error || null;
        entry.errorStatus = null;
        if (!entry.error || !entry.data) entry.data = data;
      })
      .catch((error) => {
        if (error.name !== "AbortError") { entry.error = error.message; entry.errorStatus = error.status || null; }
      })
      .finally(() => {
        entry.fetchedAt = Date.now();
        entry.pending = false;
        entry.controller = null;
        if (entries.get(scope) === entry && entry.used === generation) {
          // Several metrics requests can finish together; paint their results once.
          cancelAnimationFrame(repaint);
          const used = generation;
          repaint = requestAnimationFrame(() => { if (used === generation) ui.render(); });
        }
      });
  }
  return entry;
}
