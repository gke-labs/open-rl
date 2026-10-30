import { crosshair, MAX_WINDOW } from "./state.svelte.js";

export const WINDOWS = [[60, "1 minute"], [600, "10 minutes"], [1800, "30 minutes"], [3600, "1 hour"], [10800, "3 hours"], [21600, "6 hours"], [86400, "24 hours"], [259200, "3 days"], [MAX_WINDOW, "7 days"]];
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const SURFACES = ".capacity-track, .node-axis, .activity-track, .activity-axis, #placement-detail .chart-plot svg, #placement-detail .chart-empty";

// Native gesture listeners are scoped to the mounted Nodes component. Svelte
// reacts to view changes; the query window stays frozen until a gesture ends.
export function nodeTime(element, { view, getRange, getNow, onGesture }) {
  let gesture = null, drag = null, pinch = null, settle, ignoreClickUntil = 0;
  const events = new AbortController(), root = element.closest("#app");
  const listen = (target, name, handler, options = {}) => target.addEventListener(name, handler, { ...options, signal: events.signal });
  const fraction = (x, box) => clamp((x - box.left) / (box.width || 1), 0, 1);
  function begin(kind) {
    if (gesture) return;
    gesture = { kind, duration: view.duration, end: view.end };
    onGesture(getRange());
    root?.classList.add("node-navigating");
  }
  function select(duration, end) {
    duration = clamp(Math.round(duration), 60, MAX_WINDOW);
    view.duration = duration;
    view.end = clamp(end, getNow() - MAX_WINDOW + duration, getNow());
  }
  function finish(cancel = false) {
    clearTimeout(settle);
    if (drag?.pointerId !== undefined && element.hasPointerCapture(drag.pointerId)) element.releasePointerCapture(drag.pointerId);
    drag = pinch = null;
    if (!gesture) return;
    if (cancel) { view.duration = gesture.duration; view.end = gesture.end; }
    if (gesture.kind === "drag") ignoreClickUntil = performance.now() + 400;
    gesture = null;
    onGesture(null);
    root?.classList.remove("node-navigating");
  }
  function moveDrag(x, y, touch = false) {
    if (!drag) return false;
    const dx = x - drag.x, dy = y - drag.y;
    if (!gesture) {
      if (touch && Math.abs(dy) > 6 && Math.abs(dy) > Math.abs(dx)) { drag = null; return false; }
      if (Math.abs(dx) < 6) return false;
      begin("drag");
      if (drag.pointerId !== undefined) element.setPointerCapture(drag.pointerId);
    }
    select(drag.range.now - drag.range.start, drag.range.now - dx / (drag.box.width || 1) * (drag.range.now - drag.range.start));
    return true;
  }
  listen(element, "wheel", (event) => {
    const track = event.target.closest(SURFACES);
    if (!track || !(event.ctrlKey || event.metaKey) || (gesture && gesture.kind !== "wheel")) return;
    event.preventDefault();
    begin("wheel");
    const range = getRange(), at = fraction(event.clientX, track.getBoundingClientRect());
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? track.clientHeight : 1);
    const duration = clamp(Math.round((range.now - range.start) * Math.exp(clamp(delta * 0.002, -1, 1))), 60, MAX_WINDOW);
    select(duration, range.start + at * (range.now - range.start) + (1 - at) * duration);
    clearTimeout(settle);
    settle = setTimeout(() => finish(), 200);
  }, { passive: false });
  listen(element, "pointerdown", (event) => {
    const track = event.target.closest(SURFACES);
    if (!track || event.pointerType === "touch" || event.button !== 0 || event.target.closest("[data-placement], a")) return;
    finish();
    drag = { x: event.clientX, y: event.clientY, box: track.getBoundingClientRect(), range: getRange(), pointerId: event.pointerId };
  });
  listen(window, "pointermove", (event) => { if (drag?.pointerId === event.pointerId && moveDrag(event.clientX, event.clientY)) event.preventDefault(); });
  listen(window, "pointerup", (event) => { if (drag?.pointerId === event.pointerId) finish(); });
  listen(window, "pointercancel", (event) => { if (drag?.pointerId === event.pointerId) finish(true); });
  listen(element, "touchstart", (event) => {
    const track = event.target.closest(SURFACES);
    if (!track) return;
    if (event.touches.length === 2) {
      event.preventDefault();
      if (gesture?.kind === "wheel") finish();
      begin("pinch");
      gesture.kind = "pinch";
      drag = null;
      const [a, b] = event.touches;
      pinch = { box: track.getBoundingClientRect(), range: getRange(), x: (a.clientX + b.clientX) / 2, distance: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1 };
    } else if (event.touches.length === 1 && !event.target.closest("[data-placement], a")) {
      const touch = event.touches[0];
      drag = { x: touch.clientX, y: touch.clientY, box: track.getBoundingClientRect(), range: getRange() };
    }
  }, { passive: false });
  listen(element, "touchmove", (event) => {
    if (pinch && event.touches.length === 2) {
      event.preventDefault();
      const [a, b] = event.touches, x = (a.clientX + b.clientX) / 2;
      const duration = clamp(Math.round((pinch.range.now - pinch.range.start) * pinch.distance / (Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1)), 60, MAX_WINDOW);
      const anchor = pinch.range.start + fraction(pinch.x, pinch.box) * (pinch.range.now - pinch.range.start);
      select(duration, anchor + (1 - fraction(x, pinch.box)) * duration);
    } else if (drag && event.touches.length === 1 && moveDrag(event.touches[0].clientX, event.touches[0].clientY, true)) event.preventDefault();
  }, { passive: false });
  listen(element, "touchend", () => { if (drag || pinch) finish(); });
  listen(element, "touchcancel", () => { if (drag || pinch) finish(true); });
  listen(element, "click", (event) => {
    if (event.detail && performance.now() < ignoreClickUntil) { ignoreClickUntil = 0; event.preventDefault(); event.stopImmediatePropagation(); }
  }, { capture: true });
  listen(window, "keydown", (event) => {
    if (event.key === "Escape" && (gesture || drag || pinch)) { event.preventDefault(); event.stopImmediatePropagation(); finish(true); }
  }, { capture: true });
  // Perfetto keys: W/S zoom around the crosshair (else the middle), A/D pan by a fifth of the window.
  listen(window, "keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!["w", "a", "s", "d"].includes(key) || event.ctrlKey || event.metaKey || event.altKey || event.target.closest("input, select, textarea") || (gesture && gesture.kind !== "keys")) return;
    event.preventDefault();
    begin("keys");
    const range = getRange(), span = range.now - range.start;
    if (key === "a" || key === "d") {
      select(span, range.now + (key === "d" ? 0.2 : -0.2) * span);
      // Keep the crosshair under the pointer, which did not move.
      if (crosshair.at !== null) crosshair.at += view.end - range.now;
    } else {
      const duration = clamp(Math.round(span * (key === "w" ? 0.8 : 1.25)), 60, MAX_WINDOW);
      const at = crosshair.at === null ? 0.5 : clamp((crosshair.at - range.start) / span, 0, 1);
      select(duration, range.start + at * span + (1 - at) * duration);
    }
    clearTimeout(settle);
    settle = setTimeout(() => finish(), 300);
  });
  listen(window, "blur", () => finish(true));
  listen(window, "hashchange", () => finish());
  return { destroy() { finish(); events.abort(); } };
}
