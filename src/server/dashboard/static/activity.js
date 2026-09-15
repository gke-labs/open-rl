import { chartTime, valueText } from "./charts.js";
import { root, ui } from "./store.js";

const inspected = new WeakMap();
let tooltip, inspection;
export const hideActivityHover = () => {
  if (tooltip) tooltip.hidden = true;
  inspection?.removeAttribute("data-inspecting");
  inspection = undefined;
};

export function installActivityHover() {
  if (tooltip) return;
  tooltip = document.createElement("div");
  tooltip.className = "activity-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.hidden = true;
  const fields = ["label", "time", "detail"].map((name) => {
    const field = document.createElement(name === "label" ? "strong" : "div");
    field.className = `activity-tooltip-${name}`;
    tooltip.append(field);
    return field;
  });
  root.append(tooltip);
  root.addEventListener("pointermove", (event) => {
    if (ui.nodeNavigating || event.buttons) return hideActivityHover();
    const track = event.target.closest(".activity-track"), section = track?.closest(".cross-node-activity");
    if (section !== inspection) {
      hideActivityHover();
      inspection = section;
    }
    if (section) {
      const box = track.getBoundingClientRect();
      section.style.setProperty("--activity-cursor", `${Math.max(0, Math.min(100, (event.clientX - box.left) / box.width * 100))}%`);
      section.setAttribute("data-inspecting", "");
    }
    tooltip.hidden = true;
    const block = event.target.closest(".activity-block");
    if (!track || !block) return;
    const source = block.dataset.intervals || "[]";
    let entry = inspected.get(block);
    if (entry?.source !== source) inspected.set(block, entry = { source, intervals: JSON.parse(source) });
    const box = block.ownerSVGElement.getBoundingClientRect();
    const at = Number(track.dataset.start) + (event.clientX - box.left) / box.width * (Number(track.dataset.end) - Number(track.dataset.start));
    let left = 0, right = entry.intervals.length;
    while (left < right) {
      const middle = Math.floor((left + right) / 2);
      if (entry.intervals[middle][1] < at) left = middle + 1;
      else right = middle;
    }
    const interval = entry.intervals[left];
    if (!interval || at < interval[0] || at > interval[1]) return;
    const [from, to] = interval, date = (time) => new Date(time * 1000).toISOString().slice(0, 10);
    fields[0].textContent = block.dataset.label || "";
    fields[1].textContent = `${date(from)} · ${chartTime(from, true)} – ${date(from) === date(to) ? "" : date(to) + " · "}${chartTime(to, true)}`;
    fields[2].textContent = `Visible interval · ${valueText(to - from, "s")}${block.dataset.source ? " · " + block.dataset.source : ""}`;
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(8, Math.min(event.clientX + 12, innerWidth - tooltip.offsetWidth - 8))}px`;
    tooltip.style.top = `${Math.max(8, Math.min(event.clientY + 12, innerHeight - tooltip.offsetHeight - 8))}px`;
  });
  root.addEventListener("pointerleave", hideActivityHover);
  root.addEventListener("pointerdown", hideActivityHover, true);
  root.addEventListener("click", hideActivityHover, true);
  for (const event of ["scroll", "resize", "hashchange"]) window.addEventListener(event, hideActivityHover, true);
}
