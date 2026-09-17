// URLs carry inspection state; copying a view also pins a live time window.
import { root, ui, route, viewParams, viewLink, nodeLink } from "./store.js";
import { timeWindow } from "./node-time.js";
import { runView, logState, openRun } from "./run.js";

const number = (params, key, fallback) => params.get(key)?.trim() && Number.isFinite(Number(params.get(key))) ? Number(params.get(key)) : fallback;
const windowFrom = (params) => ({ duration: Math.max(60, Math.min(86400, Math.round(number(params, "duration", 1800)))), end: number(params, "end", null) });
let appliedHash;
export const viewReady = () => location.hash === appliedHash;

function restoreNodes(params) {
  ui.nodeSelection = windowFrom(params);
  ui.expanded = params.get("placement") || null;
  ui.inspectorNode = params.get("node") || null;
  ui.device = params.get("gpu") || "all";
}

export function restoreView() {
  appliedHash = location.hash;
  const [page, id] = route(), params = viewParams();
  if (page === "nodes") restoreNodes(params);
  if (page === "run") {
    openRun(id);
    const window = windowFrom(params), back = params.get("back");
    runView.windowMinutes = window.duration / 60;
    if (window.end !== null && window.end >= window.duration && window.end <= 8.64e12) {
      runView.windowEnd = window.end;
      runView.follow = false;
    }
    runView.nodeBack = back && route(back)[0] === "nodes" && back.startsWith("#nodes?") ? back : null;
    runView.origin = runView.nodeBack ? "nodes" : "overview";
    if (runView.nodeBack) restoreNodes(viewParams(runView.nodeBack));
    const event = number(params, "event", null);
    runView.eventAt = event !== null && event >= runView.windowEnd - window.duration && event <= runView.windowEnd ? event : null;
    logState.q = params.get("q") || "";
    logState.pod = params.get("pod") || "";
  }
}

function nodeParams(freeze = false) {
  const range = timeWindow();
  return { duration: range.now - range.start, end: freeze || !range.live ? range.now : null,
    node: ui.inspectorNode, placement: ui.expanded, gpu: ui.inspectorNode && ui.device !== "all" ? ui.device : null };
}

function runParams(freeze = false) {
  return { duration: runView.windowMinutes * 60, end: freeze || !runView.follow ? runView.windowEnd : null,
    back: runView.nodeBack, event: runView.eventAt, q: logState.q, pod: logState.pod };
}

export function currentView(freeze = false, tab = route()[2]) {
  const [page, id] = route();
  if (page === "nodes") return viewLink("nodes", nodeParams(freeze));
  if (page === "run") {
    const metrics = root.querySelector("details[data-run-metrics]");
    if (metrics && ["activity", "metrics", undefined].includes(tab)) tab = metrics.open ? "metrics" : "activity";
    return viewLink(`run/${encodeURIComponent(id)}/${tab || "activity"}`, runParams(freeze));
  }
  return location.hash;
}

export function syncViewURL() {
  if (ui.nodeNavigating || !viewReady()) return;
  const [page, id] = route(), hash = currentView();
  if (location.hash !== hash) history.replaceState(history.state, "", hash);
  appliedHash = hash;
  root.querySelector('.appbar nav a[href^="#nodes"]').href = viewLink("nodes", nodeParams());
  root.querySelectorAll('a[href^="#run/"]').forEach((link) => {
    const [, runId, tab] = route(link.getAttribute("href"));
    if (page === "run" && id === runId) link.setAttribute("href", currentView(false, tab));
    else if (page === "nodes") {
      const range = timeWindow();
      link.setAttribute("href", viewLink(`run/${encodeURIComponent(runId)}/${tab || "activity"}`, { duration: range.now - range.start, end: range.now, back: currentView(true) }));
    }
  });
  root.querySelectorAll("a[data-scheduler-placement]").forEach((link) => { link.setAttribute("href", nodeLink(link.dataset.schedulerPlacement, timeWindow())); });
}

export async function copyView(button) {
  const url = new URL(location.href);
  url.hash = currentView(true);
  let copied = false;
  try { await navigator.clipboard.writeText(url.href); copied = true; } catch {
    // The private HTTP preview lacks the secure-context clipboard API.
    const input = Object.assign(document.createElement("textarea"), { value: url.href });
    input.style.cssText = "position:fixed;opacity:0";
    document.body.append(input);
    input.select();
    try { copied = document.execCommand("copy"); } catch { copied = false; }
    input.remove();
    button.focus({ preventScroll: true });
  }
  button.textContent = copied ? "Link copied" : "Copy failed";
  setTimeout(() => { if (button.isConnected) button.textContent = "Copy link"; }, 2000);
}
