// Charts are markup: stretched SVG paths with HTML labels. Pointer and keyboard
// inspection use the same samples, with no chart instances or resize observers.
import { escape } from "./ui.js";

export const chartNumber = (value) =>
  new Intl.NumberFormat(undefined, {
    maximumSignificantDigits: 4,
    notation: Math.abs(value) >= 10000 ? "compact" : "standard",
  }).format(value);
export const chartTime = (value, seconds = false) =>
  new Date(value * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    ...(seconds ? { second: "2-digit" } : {}),
    hour12: false,
    timeZone: "UTC",
  });
export const valueText = (value, unit) =>
  unit === "%" ? `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)}%` : `${chartNumber(value)}${unit ? ` ${unit}` : ""}`;

const SCALE = 1000;

export function chart({
  title,
  unit = "",
  points = [],
  start,
  end,
  min,
  max,
  tone = "neutral",
  xFormat = "time",
  gapSeconds = Infinity,
  empty = "No samples in this time range",
}) {
  const samples = points.filter(([x]) => Number.isFinite(x) && x >= start && x <= end).sort((a, b) => a[0] - b[0]);
  const data = samples.filter(([, y]) => Number.isFinite(y));
  const head = (latest) =>
    `<figcaption class="chart-head"><h2>${escape(title)}</h2>${latest ? `<span class="chart-latest" title="${escape(xFormat === "step" ? `Step ${latest[0]}` : new Date(latest[0] * 1000).toISOString())}"><span>Latest</span> <strong>${escape(valueText(latest[1], unit))}</strong></span>` : ""}</figcaption>`;
  if (!data.length) return `<figure class="chart" data-key="${escape(title)}" data-tone="${tone}">${head(null)}<p class="chart-empty">${escape(empty)}</p></figure>`;
  const values = data.map(([, y]) => y);
  let low = Number.isFinite(min) ? min : Math.min(...values);
  let high = Number.isFinite(max) ? max : Math.max(...values);
  if (high <= low) {
    const pad = Math.max(Math.abs(low) * 0.1, 0.1);
    if (!Number.isFinite(min)) low -= pad;
    high += pad;
  }
  let ticks = [high, (high + low) / 2, low];
  if (!Number.isFinite(min) && !Number.isFinite(max)) {
    const raw = (high - low) / 3,
      magnitude = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].find((n) => n >= raw / magnitude) * magnitude;
    low = Math.floor(low / step) * step;
    high = Math.ceil(high / step) * step;
    ticks = Array.from({ length: Math.round((high - low) / step) + 1 }, (_, i) => Number((high - i * step).toPrecision(12)));
  }
  const x = (at) => (((at - start) / (end - start || 1)) * SCALE).toFixed(1);
  const y = (value) => (SCALE - ((Math.max(low, Math.min(high, value)) - low) / (high - low)) * SCALE).toFixed(1);
  const segments = [];
  let current = null;
  for (const sample of samples) {
    if (!Number.isFinite(sample[1])) {
      current = null;
      continue;
    }
    if (!current || sample[0] - current.at(-1)[0] > gapSeconds) segments.push((current = []));
    current.push(sample);
  }
  const areas = [];
  const lines = segments
    .map((segment) => {
      if (segment.length === 1) return `<path class="chart-dot" d="M${x(segment[0][0])},${y(segment[0][1])} h0.01" vector-effect="non-scaling-stroke"/>`;
      const line = segment.map(([at, value], i) => `${i ? "L" : "M"}${x(at)},${y(value)}`).join(" ");
      areas.push(`${line} L${x(segment.at(-1)[0])},${SCALE} L${x(segment[0][0])},${SCALE} Z`);
      return `<path class="chart-line" d="${line}" vector-effect="non-scaling-stroke"/>`;
    })
    .join("");
  const paths = areas.map((d) => `<path class="chart-area" d="${d}"/>`).join("") + lines;
  const tickCount = end === start ? 0 : xFormat === "step" ? Math.max(1, Math.min(4, Math.floor(end - start))) : 4;
  const label = (at) =>
    xFormat === "step" ? `step ${Math.round(at)}` : `${end - start >= 86400 ? new Date(at * 1000).toISOString().slice(5, 10) + " " : ""}${chartTime(at, end - start <= 120)}`;
  const xLabels = Array.from({ length: tickCount + 1 }, (_, i) => `<span>${escape(label(start + ((end - start) * i) / (tickCount || 1)))}</span>`).join("");
  const yLabels = ticks.map((value) => `<span>${escape(chartNumber(value))}</span>`).join("");
  const grid = ticks.map((value) => `<line class="chart-grid" x1="0" x2="${SCALE}" y1="${y(value)}" y2="${y(value)}" vector-effect="non-scaling-stroke"/>`).join("");
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  return `<figure class="chart" data-key="${escape(title)}" data-tone="${tone}">${head(data.at(-1))}
    <div class="chart-plot" tabindex="0" aria-label="${escape(title)}. Use left and right arrow keys to inspect samples." data-points="${escape(JSON.stringify(data))}" data-spans="${escape(JSON.stringify(segments.map((s) => [s[0][0], s.at(-1)[0]])))}" data-start="${start}" data-end="${end}" data-min="${low}" data-max="${high}" data-unit="${escape(unit)}" data-xformat="${xFormat}">
      <div class="chart-y">${yLabels}</div>
      <svg viewBox="0 0 ${SCALE} ${SCALE}" preserveAspectRatio="none" role="img" aria-label="${escape(title)}. Latest ${escape(valueText(data.at(-1)[1], unit))}.">${grid}${paths}</svg>
      <div class="chart-x">${xLabels}</div>
      <div class="chart-hover" hidden><div class="chart-cursor"></div><div class="chart-point"></div><div class="chart-tip"></div></div>
    </div>
    <div class="chart-summary"><span title="Arithmetic mean of the reported samples">Average <strong>${escape(valueText(average, unit))}</strong></span><span>Peak <strong>${escape(valueText(Math.max(...values), unit))}</strong></span>${xFormat === "step" ? '<span class="chart-zone">per step</span>' : ""}</div>
  </figure>`;
}

// Parse once per dataset, not on every pointer movement.
const inspected = new WeakMap();
function chartData(plot) {
  let entry = inspected.get(plot);
  if (!entry || entry.source !== plot.dataset.points || entry.spansSource !== plot.dataset.spans) {
    entry = {
      source: plot.dataset.points,
      spansSource: plot.dataset.spans,
      points: JSON.parse(plot.dataset.points || "[]"),
      spans: JSON.parse(plot.dataset.spans || "[]"),
      index: -1,
    };
    inspected.set(plot, entry);
  }
  return entry;
}
function showSample(plot, index) {
  const entry = chartData(plot),
    hover = plot.querySelector(".chart-hover"),
    svg = plot.querySelector("svg");
  if (!entry.points.length || !hover || !svg) return;
  entry.index = Math.max(0, Math.min(entry.points.length - 1, index));
  const [at, value] = entry.points[entry.index],
    box = svg.getBoundingClientRect();
  const left = ((at - Number(plot.dataset.start)) / (Number(plot.dataset.end) - Number(plot.dataset.start) || 1)) * box.width;
  hover.hidden = false;
  hover.style.left = `${box.left - plot.getBoundingClientRect().left + left}px`;
  hover.querySelector(".chart-point").style.top =
    `${Math.max(0, Math.min(100, ((Number(plot.dataset.max) - value) / (Number(plot.dataset.max) - Number(plot.dataset.min))) * 100))}%`;
  const tip = hover.querySelector(".chart-tip");
  const when = plot.dataset.xformat === "step" ? `step ${Math.round(at)}` : chartTime(at, true);
  tip.textContent = `${when} · ${valueText(value, plot.dataset.unit)}`;
  tip.style.left = `${Math.max(-left, Math.min(8, box.width - left - tip.offsetWidth))}px`;
}

export function hoverChart(plot, clientX) {
  const { points, spans } = chartData(plot),
    svg = plot.querySelector("svg"),
    hover = plot.querySelector(".chart-hover");
  if (!points.length || !svg || !hover) return;
  const box = svg.getBoundingClientRect(),
    start = Number(plot.dataset.start),
    end = Number(plot.dataset.end);
  const at = start + ((clientX - box.left) / (box.width || 1)) * (end - start);
  let left = 0,
    right = points.length - 1;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (points[middle][0] < at) left = middle + 1;
    else right = middle;
  }
  const index = left > 0 && Math.abs(points[left - 1][0] - at) < Math.abs(points[left][0] - at) ? left - 1 : left;
  // Gaps are unknown. Only inspect a visible segment or the nearby endpoint.
  const near = (Math.abs(points[index][0] - at) / (end - start || 1)) * box.width < 6;
  if (clientX < box.left || clientX > box.right || (!near && !spans.some(([from, to]) => at >= from && at <= to))) {
    hover.hidden = true;
    return;
  }
  showSample(plot, index);
}

export function inspectChart(plot, key) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(key)) return false;
  if (key === "Escape") {
    plot.querySelector(".chart-hover").hidden = true;
    plot.blur();
    return true;
  }
  const entry = chartData(plot);
  showSample(plot, key === "Home" ? 0 : key === "End" ? entry.points.length - 1 : (entry.index < 0 ? entry.points.length - 1 : entry.index) + (key === "ArrowLeft" ? -1 : 1));
  return true;
}
