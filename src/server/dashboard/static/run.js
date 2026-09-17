// One run's activity across its workers, with optional metrics and pod logs.

import { runIncidents } from "./timeline.js";
import { escape, encode, empty, button, runStatus } from "./ui.js";
import { chart } from "./charts.js";
import { runActivity } from "./nodes.js";
import { ui, get, nodeNow } from "./store.js";
import { use } from "./cache.js";

// The inspected run's window, and the log page that was loaded for it.
export const runView = { id: null, origin: "overview", nodeBack: null, windowMinutes: 30, windowEnd: 0, eventAt: null, follow: true };
export const logState = { key: null, records: [], cursor: null, loading: false, error: null, source: null, q: "", pod: "", params: null };
const MAX_LOGS = 2000;
let active = "",
  poll,
  controller,
  request = 0;

function stopLogs(clear = false) {
  controller?.abort();
  request++;
  logState.loading = false;
  if (!logState.params) logState.key = null;
  if (clear) Object.assign(logState, { key: null, records: [], cursor: null, error: null, source: null, params: null });
}

// The only background log work belongs to the visible run's Logs tab.
export function syncRunRoute(page, id, tab = "activity") {
  const next = page === "run" ? `${id}/${tab}` : "";
  if (active === next) return;
  stopLogs();
  clearInterval(poll);
  active = next;
  if (page === "run" && tab === "logs" && !ui.state?.recorded_at)
    poll = setInterval(() => {
      if (!document.hidden && runView.follow && !logState.loading) loadLogs(id);
    }, 5000);
}

export function resetRunWindow() {
  runView.eventAt = null;
  runView.follow = true;
  runView.windowEnd = nodeNow();
  stopLogs(true);
}

export function openRun(id) {
  runView.id = id;
  runView.origin = "overview";
  runView.nodeBack = null;
  logState.q = logState.pod = "";
  resetRunWindow();
}

export function setLogFilter(field, value) {
  if (logState[field] === value) return;
  logState[field] = value;
  stopLogs(true);
}

export function setLogEvent(at) {
  runView.eventAt = at;
  runView.follow = at === null;
  if (at === null) runView.windowEnd = nodeNow();
  stopLogs(true);
}

export function toggleLogFollow(follow = !runView.follow) {
  if (follow) resetRunWindow();
  else {
    runView.follow = false;
    if (logState.params) runView.windowEnd = Date.parse(logState.params.get("until")) / 1000;
    stopLogs();
  }
  ui.render();
}

function runRange(nearEvent = false) {
  const start = runView.windowEnd - runView.windowMinutes * 60;
  const eventAt = nearEvent ? runView.eventAt : null;
  return {
    since: new Date(Math.max(start, eventAt === null ? start : eventAt - 120) * 1000).toISOString(),
    until: new Date(Math.min(runView.windowEnd, eventAt === null ? runView.windowEnd : eventAt + 120) * 1000).toISOString(),
  };
}

export function runPage(id, tab = "activity") {
  if (runView.id !== id) openRun(id);
  // A snapshot advances time once; cache completions never create new windows.
  const observedAt = Date.parse(ui.state.observed_at) / 1000;
  if (runView.follow && Number.isFinite(observedAt)) runView.windowEnd = observedAt;
  const run = ui.state.runs.find((r) => r.run_id === id);
  const back = `<p class="overview-back"><a href="${escape(runView.nodeBack || "#overview")}">← ${runView.origin === "nodes" ? "Nodes" : "Overview"}</a></p>`;
  const sourceError = ui.state.store_error ? `<p class="source-error" role="status">${escape(ui.state.store_error)}</p>` : "";
  if (!run) return `${back}<h1 class="heading">Run</h1>${sourceError || empty("Run not found")}`;
  if (!["activity", "metrics", "logs"].includes(tab)) tab = "activity";
  const title = [(run.model || "Run").split("/").at(-1), run.run_id.slice(0, 8), { lora: "LoRA", full: "FFT", fft: "FFT" }[run.fine_tuning_type]].filter(Boolean).join(" · ");
  const description = [run.display_name, run.recipe_name].filter((value, index, values) => value && value !== title && values.indexOf(value) === index).join(" · ");
  const tabs = ["activity", "logs"].map((t) => `<a href="#run/${encode(id)}/${t}" ${(tab === "metrics" ? "activity" : tab) === t ? 'aria-current="page"' : ""}>${t[0].toUpperCase() + t.slice(1)}</a>`).join("");
  const customWindow = !runView.follow || ![10, 30, 60].includes(runView.windowMinutes);
  const range = runRange();
  const ranges = (customWindow ? `<option value="${runView.windowMinutes}" selected>${runView.origin === "nodes" ? "Selected node window" : "Selected time window"}</option>` : "") +
    [10, 30, 60].map((value) => `<option value="${value}" ${!customWindow && runView.windowMinutes === value ? "selected" : ""}>Last ${value} minutes</option>`).join("");
  return `${back}
    <div class="run-heading"><h1 class="heading" title="${escape(run.run_id)}">${escape(title)}</h1>${runStatus(run.display_status || run.status)}</div>
    ${sourceError}
    ${description ? `<p class="run-description">${escape(description)}</p>` : ""}
    <div class="run-toolbar" data-key="run-toolbar"><nav class="workspace-tabs" aria-label="Run views">${tabs}</nav><div class="run-time-controls"><label>Time range <select id="event-range" aria-label="Time range" title="${range.since} – ${range.until}">${ranges}</select></label>${button("Copy link", "data-copy-view")}</div></div>
    ${runIncidents(run, runView.windowMinutes, runView.windowEnd)}
    <div id="run-panel">${tab === "logs" ? logsPanel(id, run) : activityPanel(id, run, tab)}</div>
    <p class="run-json-link"><a href="/api/v1/dashboard/runs/${encode(id)}">Agent JSON ↗</a></p>`;
}

function activityPanel(id, run, tab) {
  const range = { start: runView.windowEnd - runView.windowMinutes * 60, now: runView.windowEnd, live: runView.follow };
  const metricsOpen = tab === "metrics" || document.querySelector(`details[data-run-metrics="${CSS.escape(id)}"]`)?.open;
  return `${runActivity(id, range)}<details class="run-metrics" data-run-metrics="${escape(id)}" data-key="metrics:${escape(id)}:${tab === "metrics"}" ${tab === "metrics" ? "open" : ""}><summary>Metrics</summary>${metricsOpen ? metricsPanel(id, run) : ""}</details>`;
}

function metricsPanel(id, run) {
  const range = runRange();
  const start = Date.parse(range.since) / 1000;
  const end = Date.parse(range.until) / 1000;
  const url = `/api/v1/dashboard/runs/${encode(id)}/metrics?${new URLSearchParams(range)}`;
  const metrics = use(url, runView.follow ? `run:${id}:${runView.windowMinutes}` : url);
  const samples = (metrics.data?.samples || []).filter((s) => s.at >= start && s.at <= end);
  const names = [
    ...new Set(
      samples.flatMap((s) =>
        Object.entries(s.metrics || {})
          .filter(([, value]) => Number.isFinite(value))
          .map(([name]) => name),
      ),
    ),
  ].slice(0, 12);
  const charts = [
    ...[...new Set(samples.length ? samples.map((s) => s.role || "process") : ["process"])].map((role) =>
      chart({ title: `${{ trainer: "Trainer", sampler: "Sampler", process: "Process" }[role] || role} operation duration`, unit: "s", points: samples.filter((s) => (s.role || "process") === role).map((s) => [s.at, s.elapsed_seconds]), start, end, min: 0, tone: "accent" }),
    ),
    ...names.map((name) =>
      chart({ title: name, points: samples.filter((s) => Number.isFinite(s.metrics?.[name])).map((s) => [s.at, s.metrics[name]]), start, end, tone: "accent" }),
    ),
  ];
  const error = metrics.error || metrics.data?.error;
  const status = error ? `${error}${samples.length ? " · Showing previously fetched metrics" : ""}` : !metrics.data ? "Loading metrics…" : metrics.pending ? "Updating…" : "";
  const rows = (run.pods || [])
    .map(
      (p) =>
        `<tr data-key="${escape(p.uid || p.name)}"><td>${escape(p.name)}</td><td>${escape(p.role ? p.role[0].toUpperCase() + p.role.slice(1) : "Unknown")}</td><td>${escape(p.node)}</td><td>${runStatus(p.problem || p.phase)}</td><td>${escape(p.restarts)}</td></tr>`,
    )
    .join("");
  return `<div class="run-metric-summary" data-key="metric-summary"><span>Completed steps <strong>${escape(run.steps)}</strong></span>${status ? `<span class="${error ? "source-error" : "muted"}" role="status">${escape(status)}</span>` : ""}</div>
    <div class="chart-grid" data-key="run-charts">${samples.length ? charts.join("") : metrics.data && !error && !metrics.pending ? empty("No operations recorded in this time range") : ""}</div>
    <h2 class="scheduler-title">Processes</h2>${run.shared_runtime ? '<p class="muted">Shared LoRA runtime</p>' : ""}
    <div class="table-scroll" data-key="run-processes"><table class="run-table"><thead><tr><th>Process</th><th>Kind</th><th>Node</th><th>State</th><th>Restarts</th></tr></thead><tbody>${rows}</tbody></table></div>${!run.pods?.length ? empty(ui.state.cluster.available ? "No current pods" : ui.state.cluster.error || "Process information unavailable") : ""}`;
}

// ---- logs -----------------------------------------------------------------------------

const logQuery = () => {
  const params = new URLSearchParams({ q: logState.q, limit: "200", ...runRange(true) });
  if (logState.pod) params.set("pod", logState.pod);
  return params;
};
const recordKey = (r) => r._key ?? r.id ?? JSON.stringify([r.timestamp, r.pod, r.container, r.message]);

export async function loadLogs(id, more = false) {
  if (active !== `${id}/logs` || runView.id !== id || logState.loading || (more && !logState.cursor)) return;
  if (more) {
    runView.follow = false;
    runView.windowEnd = Date.parse(logState.params.get("until")) / 1000;
  } else if (runView.follow) runView.windowEnd = nodeNow();
  // Paging reuses the exact query that issued the cursor, including its end time.
  const params = more ? new URLSearchParams(logState.params) : logQuery();
  const key = `${id}?${params}`;
  const current = ++request;
  controller = new AbortController();
  const query = new URLSearchParams(params);
  if (more) params.set("cursor", logState.cursor);
  logState.loading = true;
  logState.key = key;
  ui.render();
  try {
    const data = await get(`/api/v1/dashboard/runs/${encode(id)}/logs?${params}`, controller.signal);
    if (current !== request) return;
    if (!data.error) {
      const occurrences = new Map();
      const incoming = (data.records || []).map((r) => {
        const key = recordKey(r),
          occurrence = occurrences.get(key) || 0;
        occurrences.set(key, occurrence + 1);
        return { ...r, _key: r.id ?? `${key}/${occurrence}` };
      });
      const records = [...(more ? logState.records : []), ...incoming];
      logState.records = [...new Map(records.map((r) => [recordKey(r), r])).values()].slice(0, MAX_LOGS);
      logState.cursor = logState.records.length < MAX_LOGS ? data.next_cursor : null;
      logState.params = query;
    }
    logState.error = data.error || null;
    logState.source = data.source;
  } catch (error) {
    if (current === request && error.name !== "AbortError") logState.error = error.message;
  } finally {
    if (current === request) {
      logState.loading = false;
      ui.render();
    }
  }
}

function logsPanel(id, run) {
  const range = runRange(true);
  const scope =
    runView.eventAt === null
      ? ""
      : `<div class="log-time-scope" data-key="log-scope"><span>${escape(range.since.slice(0, 10))} · ${escape(range.since.slice(11, 19))}–${escape(range.until.slice(11, 19))}</span><a href="#run/${encode(id)}/logs" data-all-logs="true">All logs</a></div>`;
  const sources = new Map((run.pods || []).map((p) => [p.name, p]));
  for (const record of logState.records) if (record.pod && !sources.has(record.pod)) sources.set(record.pod, { ...record, name: record.pod });
  if (logState.pod && !sources.has(logState.pod)) sources.set(logState.pod, { name: logState.pod });
  const pods = [...sources.values()]
    .map(
      (p) =>
        `<option value="${escape(p.name)}" ${logState.pod === p.name ? "selected" : ""}>${escape(p.role || "Unknown")} · ${escape(p.node || p.name)} / ${escape(p.name)}</option>`,
    )
    .join("");
  const rows = logState.records
    .map((r) => {
      const at = Date.parse(r.timestamp);
      const timestamp = Number.isFinite(at) ? new Date(at).toISOString() : "";
      const severity = ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"].includes(r.severity) ? "error" : r.severity === "WARNING" ? "warning" : "";
      return `<div class="workspace-logrow" data-key="${escape(recordKey(r))}" data-severity="${severity}"><time class="log-time" datetime="${escape(timestamp)}" title="${escape(timestamp || "No timestamp")}">${timestamp ? timestamp.slice(11, 23) : "—"}</time><pre class="log-message">${escape(r.message)}${r.message_truncated ? '<span class="muted">\n[Message truncated by source]</span>' : ""}</pre></div>`;
    })
    .join("");
  const source = { gke: "Cloud Logging", demo: "Demo logs", kubernetes: "Kubernetes pod logs" }[logState.source] || logState.source || "Logs";
  const status = [source, rows || !logState.error ? `${logState.records.length.toLocaleString()} ${logState.records.length === 1 ? "record" : "records"} · Newest first` : "", logState.loading ? (rows ? "Updating…" : "Loading…") : logState.error ? "" : ui.state.recorded_at ? "Recorded logs" : runView.follow ? "Updates every 5s" : "Paused"].filter(Boolean).join(" · ");
  return `${run.shared_runtime ? '<p class="muted">These pods serve a shared LoRA runtime. Their logs can include other runs.</p>' : ""}${scope}
    <div class="log-toolbar" data-key="log-toolbar"><input id="log-search" type="search" value="${escape(logState.q)}" placeholder="Search logs" aria-label="Search logs"><select id="log-source" aria-label="Pod"><option value="" ${!logState.pod ? "selected" : ""}>All pods</option>${pods}</select>${ui.state.recorded_at ? "" : button(runView.follow ? "Pause updates" : "Follow logs", 'data-log-follow="true"')}</div>
    <div id="log-status" class="log-status" role="status"><span>${escape(status)}</span>${logState.error ? `<span class="log-error">${escape(logState.error)}${rows ? " · Showing previously fetched records" : ""}</span>` : ""}</div>
    <div id="log-lines" tabindex="0" aria-label="Run logs">${rows || (logState.loading || logState.error ? "" : empty("No logs match this time range and filter"))}</div>
    <div id="log-more">${logState.cursor ? button(logState.loading ? "Loading…" : "Older logs", `data-older="true" ${logState.loading ? "disabled" : ""}`) : logState.records.length === MAX_LOGS ? '<p class="muted">2,000 records shown. Narrow the time range or search to inspect more.</p>' : ""}</div>`;
}

// Input changes clear the current query; rendering the loading state never refetches.
export function ensureLogs(id) {
  const lines = document.getElementById("log-lines");
  if (!lines) return;
  lines.onscroll = () => {
    if (runView.follow && lines.scrollTop > 24) toggleLogFollow(false);
  };
  if (!logState.key) loadLogs(id);
}
