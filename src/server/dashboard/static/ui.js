export const escape = (value) => String(value ?? "—").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export const encode = encodeURIComponent;
export const empty = (text) => `<p class="empty-state">${escape(text)}</p>`;
export const button = (label, attrs = "") => `<button type="button" class="chip" ${attrs}>${escape(label)}</button>`;

export function shortNodeName(name, nodes = []) {
  const full = String(name || "—"), parts = full.split("-");
  const others = nodes.map((node) => typeof node === "string" ? node : node.name).filter((other) => other && other !== full);
  for (let i = parts.length - 1; i >= 0; i--) {
    const suffix = parts.slice(i).join("-");
    if (suffix && !others.some((other) => other === suffix || other.endsWith(`-${suffix}`))) return suffix;
  }
  return full;
}

export const runStatus = (value) => {
  const label = String(value || "unknown");
  const tone =
    {
      running: "running",
      starting: "running",
      queued: "pending",
      unassigned: "pending",
      "needs attention": "failed",
      failed: "failed",
      completed: "completed",
      ended: "completed",
      unknown: "pending",
    }[label.toLowerCase()] || "pending";
  return `<span class="state state-${tone}"><span class="state-dot" aria-hidden="true"></span>${escape(label)}</span>`;
};

const seconds = (value) => (value === null || value === undefined || value === "" ? NaN : typeof value === "number" ? value : Date.parse(value) / 1000);

export function elapsedTime(run, observedAt) {
  const start = seconds(run.created_at);
  const end = ["completed", "failed", "ended"].includes(String(run.status || "").toLowerCase()) && run.completed_at ? seconds(run.completed_at) : seconds(observedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "—";
  return duration(Math.max(0, end - start));
}

export const duration = (total) => {
  if (total < 90) return `${Math.round(total)}s`;
  if (total < 5400) return `${Math.round(total / 60)}m`;
  if (total < 172800) return `${(total / 3600).toFixed(1)}h`;
  return `${(total / 86400).toFixed(1)}d`;
};

// morph patches a container toward new markup instead of replacing it, so a
// refresh keeps scroll position, focus, open panels and typed input.
export function morph(container, html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  morphChildren(container, template.content);
}
function morphChildren(from, to) {
  const keyed = new Map(
    Array.from(from.childNodes)
      .filter(keyOf)
      .map((node) => [keyOf(node), node]),
  );
  let cursor = from.firstChild;
  for (const next of Array.from(to.childNodes)) {
    const key = keyOf(next);
    let node = key ? keyed.get(key) : cursor && !keyOf(cursor) ? cursor : null;
    if (!node) from.insertBefore((node = next), cursor);
    else {
      if (node !== cursor) from.insertBefore(node, cursor);
      node = morphNode(node, next);
    }
    cursor = node.nextSibling;
  }
  while (cursor) {
    const next = cursor.nextSibling;
    cursor.remove();
    cursor = next;
  }
}
const keyOf = (node) => node.nodeType === 1 && (node.id || node.dataset.key);

function morphNode(node, next) {
  if (node.isEqualNode(next)) return node;
  if (node.nodeType !== next.nodeType || (node.nodeType === 1 && node.tagName !== next.tagName)) {
    node.replaceWith(next);
    return next;
  }
  if (node.nodeType !== 1) {
    if (node.data !== next.data) node.data = next.data;
    return node;
  }
  // Pointer state and open pickers survive background updates.
  if (node.classList.contains("chart-hover")) return node;
  if (node.classList.contains("chart-plot") && ["points", "spans", "start", "end", "min", "max"].some((key) => node.dataset[key] !== next.dataset[key])) {
    const hover = node.querySelector(".chart-hover");
    if (hover) hover.hidden = true;
  }
  const preserve = (name) => name === "open" && node.tagName === "DETAILS";
  const value = next.value;
  for (const { name } of Array.from(node.attributes)) if (!preserve(name) && !next.hasAttribute(name)) node.removeAttribute(name);
  for (const { name, value } of Array.from(next.attributes)) if (!preserve(name) && node.getAttribute(name) !== value) node.setAttribute(name, value);
  morphChildren(node, next);
  // Patch options too; skipping SELECT left time ranges and pod lists stale.
  if (["INPUT", "SELECT"].includes(node.tagName) && node !== document.activeElement && node.value !== value) node.value = value;
  return node;
}
