// One window for node allocation blocks, run activity, and GPU telemetry.
import { escape, button } from "./ui.js";
import { root, ui, route, nodeNow, nodeTime } from "./store.js";

export const WINDOWS = [[60, "1 minute"], [600, "10 minutes"], [1800, "30 minutes"], [3600, "1 hour"], [10800, "3 hours"], [21600, "6 hours"], [86400, "24 hours"]];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

export function timeWindow() {
  const observed = nodeNow();
  const duration = clamp(Math.round(Number(ui.nodeSelection.duration) || 1800), 60, 86400);
  const now = clamp(Number.isFinite(ui.nodeSelection.end) ? ui.nodeSelection.end : observed, observed - 86400 + duration, observed);
  return { start: now - duration, now, live: ui.nodeSelection.end === null };
}

export function timeControl(range) {
  const duration = range.now - range.start, seconds = duration <= 120;
  const local = (at) => new Date(at * 1000).toISOString().slice(0, seconds ? 19 : 16);
  const recording = !!ui.state?.recorded_at, startDate = local(range.start).slice(0, 10), endDate = local(range.now).slice(0, 10);
  const custom = [[Math.floor(duration / 3600), "h"], [Math.floor(duration / 60) % 60, "m"], [duration % 60, "s"]].filter(([n]) => n).map(([n, unit]) => `${n}${unit}`).join(" ");
  const options = WINDOWS.some(([n]) => n === duration) ? WINDOWS : [[duration, `${custom} (custom)`], ...WINDOWS];
  const chevron = (rotation = 0) => `<svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(${rotation} 8 8)"/></svg>`;
  const label = range.live ? `Last ${WINDOWS.find(([n]) => n === duration)?.[1] || custom}` : `${startDate.slice(5)} · ${nodeTime(range.start, seconds)} – ${startDate === endDate ? "" : `${endDate.slice(5)} · `}${nodeTime(range.now, seconds)}`;
  return `<div class="time-control" aria-label="Node time range">
    <button type="button" class="time-shift" data-time-shift="-1" aria-label="Previous time window" ${range.start <= nodeNow() - 86400 ? "disabled" : ""}>${chevron(90)}</button>
    <details class="time-picker" data-key="node-time-picker"><summary class="time-summary" title="${escape(label)}"><span>${escape(label)}</span>${chevron()}</summary>
      <div class="time-popover"><label>Window <select data-time-duration>${options.map(([n, text]) => `<option value="${n}" ${n === duration ? "selected" : ""}>${escape(text)}</option>`).join("")}</select></label>
      <label>Until (UTC)<input type="datetime-local" data-time-end value="${range.live ? "" : local(range.now)}" min="${local(nodeNow() - 86400 + duration)}" max="${local(nodeNow())}" step="${seconds ? 1 : 60}"></label>
      ${range.live ? `<span class="muted">${recording ? "At recording end" : "Following current time"}</span>` : button(recording ? "Return to recording end" : "Return to live", 'data-time-live="true"')}</div>
    </details><button type="button" class="time-shift" data-time-shift="1" aria-label="Next time window" ${range.now >= nodeNow() ? "disabled" : ""}>${chevron(-90)}</button>${button("Copy link", "data-copy-view")}</div>`;
}

const SURFACES = ".capacity-track, .node-axis, .activity-track, .activity-axis, #placement-detail .chart-plot svg, #placement-detail .chart-empty";
let gesture = null, drag = null, pinch = null, frame = 0, settle = 0, ignoreClickUntil = 0, installed = false;
const surface = (target) => route()[0] === "nodes" && target.closest?.(SURFACES);
const fraction = (x, box) => clamp((x - box.left) / (box.width || 1), 0, 1);

function begin(kind) {
  if (gesture) return;
  gesture = { kind, selection: { ...ui.nodeSelection } };
  ui.nodeQueryRange = timeWindow();
  ui.nodeNavigating = true;
  root.classList.add("node-navigating");
}

function select(duration, end) {
  duration = clamp(Math.round(duration), 60, 86400);
  const now = nodeNow();
  ui.nodeSelection = { duration, end: clamp(end, now - 86400 + duration, now) };
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(() => { frame = 0; ui.render(); });
}

function finish(cancel = false, render = true) {
  clearTimeout(settle);
  cancelAnimationFrame(frame);
  frame = 0;
  if (drag?.pointerId !== undefined && root.hasPointerCapture(drag.pointerId)) root.releasePointerCapture(drag.pointerId);
  drag = pinch = null;
  if (!gesture) return;
  if (cancel) ui.nodeSelection = gesture.selection;
  if (gesture.kind === "drag") ignoreClickUntil = performance.now() + 400;
  gesture = null;
  ui.nodeQueryRange = null;
  ui.nodeNavigating = false;
  root.classList.remove("node-navigating");
  if (render) ui.render();
}

export const cancelNodeGesture = (render = true) => finish(true, render);

function moveDrag(x, y, touch = false) {
  if (!drag) return false;
  const dx = x - drag.x, dy = y - drag.y;
  if (!gesture) {
    if (touch && Math.abs(dy) > 6 && Math.abs(dy) > Math.abs(dx)) { drag = null; return false; }
    if (Math.abs(dx) < 6) return false;
    begin("drag");
    if (drag.pointerId !== undefined) root.setPointerCapture(drag.pointerId);
  }
  const duration = drag.range.now - drag.range.start;
  select(duration, drag.range.now - (dx / (drag.box.width || 1)) * duration);
  return true;
}

export function installNodeTime() {
  if (installed) return;
  installed = true;
  root.addEventListener("wheel", (event) => {
    const track = surface(event.target);
    if (!track || !(event.ctrlKey || event.metaKey) || (gesture && gesture.kind !== "wheel")) return;
    event.preventDefault();
    begin("wheel");
    const range = timeWindow(), at = fraction(event.clientX, track.getBoundingClientRect());
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? track.clientHeight : 1);
    const duration = clamp(Math.round((range.now - range.start) * Math.exp(clamp(delta * 0.002, -1, 1))), 60, 86400);
    select(duration, range.start + at * (range.now - range.start) + (1 - at) * duration);
    clearTimeout(settle);
    settle = setTimeout(() => finish(), 200);
  }, { passive: false });

  root.addEventListener("pointerdown", (event) => {
    const track = surface(event.target);
    if (!track || event.pointerType === "touch" || event.button !== 0 || event.target.closest("[data-placement], a")) return;
    finish();
    drag = { x: event.clientX, y: event.clientY, box: track.getBoundingClientRect(), range: timeWindow(), pointerId: event.pointerId };
  });
  window.addEventListener("pointermove", (event) => {
    if (drag?.pointerId === event.pointerId && moveDrag(event.clientX, event.clientY)) event.preventDefault();
  });
  window.addEventListener("pointerup", (event) => { if (drag?.pointerId === event.pointerId) finish(); });
  window.addEventListener("pointercancel", (event) => { if (drag?.pointerId === event.pointerId) finish(true); });

  root.addEventListener("touchstart", (event) => {
    const track = surface(event.target);
    if (!track) return;
    if (event.touches.length === 2) {
      event.preventDefault();
      if (gesture?.kind === "wheel") finish();
      begin("pinch");
      gesture.kind = "pinch";
      drag = null;
      const [a, b] = event.touches;
      pinch = { box: track.getBoundingClientRect(), range: timeWindow(), x: (a.clientX + b.clientX) / 2, distance: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1 };
    } else if (event.touches.length === 1 && !event.target.closest("[data-placement], a")) {
      const touch = event.touches[0];
      drag = { x: touch.clientX, y: touch.clientY, box: track.getBoundingClientRect(), range: timeWindow() };
    }
  }, { passive: false });
  root.addEventListener("touchmove", (event) => {
    if (pinch && event.touches.length === 2) {
      event.preventDefault();
      const [a, b] = event.touches, x = (a.clientX + b.clientX) / 2;
      const duration = clamp(Math.round((pinch.range.now - pinch.range.start) * pinch.distance / (Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1)), 60, 86400);
      const anchor = pinch.range.start + fraction(pinch.x, pinch.box) * (pinch.range.now - pinch.range.start);
      select(duration, anchor + (1 - fraction(x, pinch.box)) * duration);
    } else if (drag && event.touches.length === 1 && moveDrag(event.touches[0].clientX, event.touches[0].clientY, true)) event.preventDefault();
  }, { passive: false });
  root.addEventListener("touchend", () => { if (drag || pinch) finish(); });
  root.addEventListener("touchcancel", () => { if (drag || pinch) finish(true); });
  root.addEventListener("click", (event) => {
    if (event.detail && performance.now() < ignoreClickUntil) { ignoreClickUntil = 0; event.preventDefault(); event.stopImmediatePropagation(); }
  }, true);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && (gesture || drag || pinch)) { event.preventDefault(); event.stopImmediatePropagation(); finish(true); }
  }, true);
  window.addEventListener("blur", () => cancelNodeGesture());
}
