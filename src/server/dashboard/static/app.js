// Entry point: routing, the poll loop, and the page-wide event handlers.
// Every page is a function of the shared store; this file decides which page
// runs, keeps the snapshot fresh, and turns user input into state changes.

import { morph } from "./ui.js";
import { runs, scheduler, health, experiments } from "./views.js";
import { hoverChart, inspectChart } from "./charts.js";
import { installActivityHover, hideActivityHover } from "./activity.js";
import { root, content, ui, route, get, nodeNow } from "./store.js";
import { renderNodes } from "./nodes.js";
import { installNodeTime, cancelNodeGesture, timeWindow } from "./node-time.js";
import { runPage, ensureLogs, loadLogs, runView, resetRunWindow, syncRunRoute, setLogFilter, setLogEvent, toggleLogFollow } from "./run.js";
import { restoreView, syncViewURL, currentView, copyView, viewReady } from "./navigation.js";
import { use, beginRender, endRender } from "./cache.js";

const pageOf = (page) => (!page || ["run", "runs"].includes(page) ? "overview" : page);
let scrollToSelection = false;

function render() {
  if (!ui.state || !viewReady()) return;
  hideActivityHover();
  const [page, id, tab] = route();
  syncRunRoute(page, id, tab);
  beginRender();
  try {
    if (page === "nodes") renderNodes();
    else if (page === "scheduler") morph(content, scheduler(ui.state));
    else if (page === "health") morph(content, health(ui.state));
    else if (page === "experiments") morph(content, experiments(use("/api/v1/dashboard/experiments")));
    else if (page === "run") morph(content, runPage(id, tab));
    else morph(content, runs(ui.state));
  } finally {
    endRender();
  }
  const current = page === "run" ? runView.origin : pageOf(page);
  root.querySelectorAll(".appbar nav a").forEach((link) => {
    if (route(link.hash)[0] === current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  syncViewURL();
  if (page === "nodes" && scrollToSelection) {
    content.querySelector("#placement-detail")?.scrollIntoView({ block: "start" });
    scrollToSelection = false;
  }
  if (page === "run" && tab === "logs") ensureLogs(id);
}
ui.render = render;
restoreView();
scrollToSelection = route()[0] === "nodes" && !!(ui.inspectorNode || ui.expanded);
installNodeTime();
installActivityHover();
root.addEventListener("toggle", (event) => {
  if (!event.target.matches("details[data-run-metrics]")) return;
  syncViewURL();
  if (event.target.open && !event.target.querySelector(".run-metric-summary")) render();
}, true);

// ---- interactions ----------------------------------------------------------

function closeDetails() {
  const previous = ui.inspectorNode;
  ui.expanded = ui.inspectorNode = null;
  root.style.removeProperty("min-height");
  render();
  if (previous) content.querySelector(`[data-node-details="${CSS.escape(previous)}"]`)?.focus({ preventScroll: true });
}

root.addEventListener("click", (event) => {
  const incident = event.target.closest("[data-event-at], [data-all-logs]");
  if (incident) {
    event.preventDefault();
    setLogEvent(incident.dataset.eventAt ? Number(incident.dataset.eventAt) : null);
    if (route()[2] === "logs") render();
    else location.hash = currentView(false, "logs");
    return;
  }
  const allocation = event.target.closest("[data-placement]");
  if (allocation) {
    const node = allocation.closest(".node-placement-group").dataset.key;
    if (node !== ui.inspectorNode) {
      root.style.removeProperty("min-height");
    }
    if (!allocation.closest("#placement-detail")) ui.device = "all";
    ui.expanded = allocation.dataset.placement;
    ui.inspectorNode = node;
    render();
    return;
  }
  const target = event.target.closest("button");
  if (!target) return;
  if (target.hasAttribute("data-copy-view")) return void copyView(target);
  if (target.hasAttribute("data-close-details")) return closeDetails();
  if (target.dataset.nodeDetails) {
    if (ui.inspectorNode === target.dataset.nodeDetails) return closeDetails();
    root.style.removeProperty("min-height");
    ui.inspectorNode = target.dataset.nodeDetails;
    ui.expanded = null;
    ui.device = "all";
    render();
  }
  if (target.hasAttribute("data-retry-snapshot")) refresh();
  if (target.dataset.device) {
    keepViewport();
    ui.device = target.dataset.device;
    render();
  }
  if (target.dataset.timeLive) {
    ui.nodeSelection = { ...ui.nodeSelection, end: null };
    render();
  }
  if (target.dataset.timeShift) {
    const range = timeWindow();
    const end = range.now + Number(target.dataset.timeShift) * (range.now - range.start);
    ui.nodeSelection = { ...ui.nodeSelection, end: end >= nodeNow() ? null : end };
    render();
  }
  if (target.dataset.logFollow !== undefined) toggleLogFollow();
  if (target.dataset.older) loadLogs(route()[1], true);
});

function keepViewport() {
  // A smaller GPU section must not shorten the document past the viewport
  // and make the browser clamp scroll. Any extra space stays below the fleet.
  root.style.minHeight = `${window.scrollY + innerHeight}px`;
}

root.addEventListener("keydown", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot && inspectChart(plot, event.key)) {
    event.preventDefault();
    return;
  }
  if (event.key === "Escape") {
    const picker = root.querySelector(".time-picker[open]");
    if (picker) {
      picker.open = false;
      picker.querySelector("summary").focus();
      event.preventDefault();
      return;
    }
  }
  if (!event.defaultPrevented && event.key === "Escape" && route()[0] === "nodes" && (ui.inspectorNode || ui.expanded)) {
    closeDetails();
  }
});

let searchTimer;
root.addEventListener("input", (event) => {
  if (event.target.id === "log-search") {
    setLogFilter("q", event.target.value);
    syncViewURL();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      if (route()[0] === "run" && route()[2] === "logs") loadLogs(route()[1]);
    }, 250);
  }
});

root.addEventListener("change", (event) => {
  const { target } = event;
  if (target.id === "log-source") {
    setLogFilter("pod", target.value);
    loadLogs(route()[1]);
  }
  if (target.id === "event-range") {
    runView.windowMinutes = Number(target.value);
    resetRunWindow();
    render();
  }
  if (target.dataset.timeDuration !== undefined) {
    ui.nodeSelection = { duration: Number(target.value), end: ui.nodeSelection.end === null ? null : timeWindow().now };
    render();
  }
  if (target.dataset.timeEnd !== undefined) {
    if (!target.validity.valid) return;
    const end = target.value ? Date.parse(`${target.value}Z`) / 1000 : null;
    ui.nodeSelection = { ...ui.nodeSelection, end: Number.isFinite(end) ? Math.min(end, nodeNow()) : null };
    render();
  }
});

root.addEventListener("pointermove", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot && !ui.nodeNavigating) hoverChart(plot, event.clientX);
});
root.addEventListener("pointerleave", (event) => event.target.classList?.contains("chart-plot") && (event.target.querySelector(".chart-hover").hidden = true), true);
root.addEventListener("focusout", (event) => {
  if (event.target.matches(".chart-plot")) event.target.querySelector(".chart-hover")?.setAttribute("hidden", "");
});
window.addEventListener("resize", () => root.querySelectorAll(".chart-hover").forEach((hover) => { hover.hidden = true; }));
document.addEventListener("click", (event) => {
  root.querySelectorAll(".time-picker[open]").forEach((picker) => {
    if (!picker.contains(event.target)) picker.open = false;
  });
});

window.addEventListener("hashchange", () => {
  cancelNodeGesture(false);
  root.style.removeProperty("min-height");
  clearTimeout(searchTimer);
  restoreView();
  scrollToSelection = route()[0] === "nodes" && !!(ui.inspectorNode || ui.expanded);
  syncRunRoute(...route());
  content.innerHTML = "";
  render();
  if (route()[0] === "experiments") content.querySelector('.experiment-row[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
});

// ---- polling -----------------------------------------------------------------

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    ui.state = await get("/api/v1/dashboard/snapshot");
    document.getElementById("snapshot-error").hidden = true;
    document.getElementById("connection").textContent = ui.state.recorded_at ? "Recording" : ui.state.demo ? "Demo" : ui.state.cluster.available ? "Connected" : "Cluster unavailable";
    document.getElementById("connection").title = ui.state.recorded_at ? `Captured ${ui.state.recorded_at}` : "";
    render();
  } catch (error) {
    document.getElementById("connection").textContent = ui.state ? "Stale snapshot" : "Unavailable";
    document.getElementById("connection").title = error.message;
    const notice = document.getElementById("snapshot-error");
    notice.querySelector("span").textContent = `${error.message}${ui.state ? ` Showing the snapshot from ${ui.state.observed_at}.` : ""}`;
    notice.hidden = false;
    if (!ui.state) content.replaceChildren();
  } finally {
    refreshing = false;
  }
}
refresh();
setInterval(() => {
  if (!document.hidden && !ui.state?.recorded_at) refresh();
}, 10000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
