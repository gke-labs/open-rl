// Presentation geometry only: identity and attribution come from the inspection API.
import { family } from "./format.js";
export { shortNodeName } from "./format.js";
const numbers = ["standard", "compact"].map((notation) => new Intl.NumberFormat(undefined, { maximumSignificantDigits: 4, notation }));
const percent = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
export const chartNumber = (value) => numbers[Number(Math.abs(value) >= 10000)].format(value);
export const chartTime = (at, seconds = false) => new Date(at * 1000).toISOString().slice(11, seconds ? 19 : 16);
export const valueText = (value, unit) => unit === "%" ? `${percent.format(value)}%` : `${chartNumber(value)}${unit ? ` ${unit}` : ""}`;
export const axisTime = (at, range) => `${range.now - range.start >= 86400 ? `${new Date(at * 1000).toISOString().slice(5, 10)} ` : ""}${chartTime(at, range.now - range.start <= 120)}`;
export const placementColor = (p) => family(p.run_ids?.[0] || p.runtime_id);
// The exporter's collection interval in k8s/deploy; each engine-activity reading averages it.
export const DCGM_READING_SECONDS = 10;
export const dutyBands = (lanes, device = null) => lanes.filter((p) => p.intervals.length && (!device || p.devices.includes(device))).map((p) => ({ key: p.id, label: `${p.label} · ${p.role || "process"}`, hue: placementColor(p), role: p.role, spans: p.intervals }));
export const deviceLabel = (name) => /(\d+)$/.exec(String(name ?? ""))?.[1] ?? String(name ?? "");
export const acceleratorLabel = (node) => node.accelerator ? node.accelerator.replace(/^nvidia[- ]/i, "").replace(/-\d+gb$/i, "").replace(/-/g, " ").toUpperCase() : node.gpu_capacity ? "GPU" : "CPU";
export function clipPlacements(placements = [], range, error = null, status = null) {
  const clip = (spans = []) => spans.map(([a, b]) => [Math.max(range.start, a), Math.min(range.now, b)]).filter(([a, b]) => b > a);
  // A request failure belongs to every lane; a lane's own error is only about its ops.
  return placements.map((p) => ({ ...p, intervals: clip(p.intervals), ops: Object.fromEntries(Object.entries(p.ops || {}).map(([name, spans]) => [name, clip(spans)])), error: status ? error : p.error, errorStatus: status || p.errorStatus }));
}
export function activitySource(placements) {
  if (!placements.length) return "";
  const source = placements.every((p) => p.exact) ? "GPU turns" : placements.some((p) => p.exact) ? "GPU turns and recorded operations" : "Recorded operations; may include GPU wait time";
  return `${source}. Gaps may include unrecorded activity.`;
}
export function processLabel(p, nodes, { gpus: named = true } = {}) {
  const node = nodes.find((n) => n.name === p.node), devices = p.devices.map((id) => node?.devices.find((d) => d.id === id));
  // A lane rebuilt from recorded ops has no known GPUs, which is not zero GPUs.
  const gpus = devices.length && devices.every(Boolean) ? `GPU${devices.length === 1 ? "" : "s"} ${devices.map((d) => deviceLabel(d.name)).join(", ")}` : p.device_count ? `${p.device_count} GPU${p.device_count === 1 ? "" : "s"}` : null;
  const role = p.role || "Process", accelerator = node && acceleratorLabel(node);
  return [role[0].toUpperCase() + role.slice(1), accelerator === "GPU" ? null : accelerator, named && gpus].filter(Boolean).join(" · ");
}
export const OP_ORDER = ["forward_backward", "forward", "optim_step", "sample", "init", "wake_up", "sleep", "weight_sync"];
export const opRank = (name) => OP_ORDER.includes(name) ? OP_ORDER.indexOf(name) : OP_ORDER.length;
export const opLabel = (name) => name.replace(/_/g, " ");
export function claimLabel(node, placements, live) {
  if (live && !node.ready) return "Offline";
  if (!node.gpu_capacity) return "CPU";
  if (node.openrl_enabled === false) return "Not enabled";
  if (!placements.every((p) => p.devices.length === p.device_count && p.devices.every((id) => node.devices.some((d) => d.id === id)))) return "— claimed";
  return `${new Set(placements.flatMap((p) => p.devices)).size}/${node.gpu_capacity} claimed`;
}
export function capacityBars(devices, placements, range, selected) {
  // Every held GPU draws the turns and ops its holders ran, never one block naming the run; the name is a click away.
  const height = devices.length === 1 ? 30 : 22;
  const span = (from, to) => `left:${100 * Math.max(0, Math.max(range.start, from) - range.start) / (range.now - range.start)}%;width:${100 * Math.max(0, Math.min(range.now, to) - Math.max(range.start, from)) / (range.now - range.start)}%`;
  return devices.flatMap((device, i) => {
    const rows = placements.filter((p) => p.devices.includes(device.id)).sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
    if (!rows.length) return [];
    const from = Math.max(range.start, Math.min(...rows.map((p) => p.start))), to = Math.min(range.now, Math.max(...rows.map((p) => p.end)));
    return [{ key: `gpu:${device.id}`, placement: rows.find((p) => p.id === selected) || rows.at(-1), selected: rows.some((p) => p.id === selected), title: `GPU activity: ${rows.map((p) => `${p.label} ${p.role || ""}`.trim()).join(", ")}`,
      style: `${span(from, to)};top:${i * height + 2}px;height:${height - 4}px`,
      holds: rows.flatMap((p) => p.intervals.map(([a, b]) => ({ placement: p, style: `left:${100 * (Math.max(a, from) - from) / (to - from || 1)}%;width:${100 * (Math.min(b, to) - Math.max(a, from)) / (to - from || 1)}%` }))) }];
  });
}
// sortedPoints must be in time order; unsorted input gives wrong gaps without an error.
export function gapFor(sortedPoints, floor) {
  const spacings = sortedPoints.slice(1).map(([at], i) => at - sortedPoints[i][0]).filter((d) => d > 0).sort((a, b) => a - b);
  return Math.max(floor, 3 * (spacings[Math.floor(spacings.length / 2)] || 0));
}
// Aggregate one physical generation per slot and timestamp; incomplete or ambiguous samples stay gaps.
export function deviceSeries(devices, field) {
  const slots = new Set(devices.map((d) => d.id)).size, byTime = new Map();
  for (const device of devices) for (const [at, value] of device[field] || []) {
    if (!byTime.has(at)) byTime.set(at, new Map());
    const values = byTime.get(at), samples = values.get(device.id) || [];
    if (Number.isFinite(value)) samples.push(value);
    values.set(device.id, samples);
  }
  return [...byTime].sort((a, b) => a[0] - b[0]).map(([at, values]) => [at, values.size === slots && [...values.values()].every((samples) => samples.length === 1) ? [...values.values()].reduce((sum, samples) => sum + samples[0], 0) / (field === "utilization" ? slots : 1) : null]);
}
export const lastValue = (points, range) => points.findLast(([at]) => at >= range.start && at <= range.now)?.[1];
