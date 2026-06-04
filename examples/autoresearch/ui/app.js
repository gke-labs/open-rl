import { diffHtml } from "./diff.js";

const S = {
  task: "",
  agents: [],
  rows: new Map(),
  active: sessionStorage.getItem("autoresearch-active"),
  selected: JSON.parse(sessionStorage.getItem("autoresearch-selected") || "{}"),
  tabs: JSON.parse(sessionStorage.getItem("autoresearch-tabs") || "{}"),
  mode: sessionStorage.getItem("autoresearch-mode") || "all",
  texts: {},
  stream: null,
  events: null,
  chartWidth: 1440,
};

const UI = {
  session: document.querySelector("#session-name"),
  agents: document.querySelector("#agents"),
  viewBar: document.querySelector("#view-bar"),
  empty: document.querySelector("#empty-state"),
  chart: document.querySelector("#chart-slot"),
  table: document.querySelector("#table-slot"),
  detail: document.querySelector("#detail-slot"),
};

const ansi = new Map([
  [30, "#8f8f8f"], [31, "#ff7b72"], [32, "#7ee787"], [33, "#f2cc60"], [34, "#58a6ff"], [35, "#d2a8ff"], [36, "#56d4dd"], [37, "#d6d6d6"],
  [90, "#6e7681"], [91, "#ffa198"], [92, "#7ee787"], [93, "#f2cc60"], [94, "#79c0ff"], [95, "#d2a8ff"], [96, "#56d4dd"], [97, "#f0f6fc"],
]);
const LIVE_TAIL_BYTES = 256 * 1024;
const MAX_ROW_TEXT = 512 * 1024;
const APPEND_FIELDS = new Set(["agent"]);

const H = (s = "") => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const esc = (s = "") => String(s).replaceAll("\\u001b", "\x1b").replaceAll("\\x1b", "\x1b");
const textState = (key) => S.texts[key] ||= { text: "", loaded: false, partial: false, offset: 0, pending: null, error: "", scroll: null };

function saveUiSelection() {
  sessionStorage.setItem("autoresearch-active", S.active || "");
  sessionStorage.setItem("autoresearch-selected", JSON.stringify(S.selected));
  sessionStorage.setItem("autoresearch-tabs", JSON.stringify(S.tabs));
  sessionStorage.setItem("autoresearch-mode", S.mode);
}

function updateUiSelection(patch = {}, options = {}) {
  rememberScroll();
  Object.assign(S, patch);
  keepSelection();
  saveUiSelection();
  render(options);
}

function mergeRow(patch) {
  const row = S.rows.get(patch.id) || { id: patch.id };
  for (const [key, value] of Object.entries(patch)) {
    if (APPEND_FIELDS.has(key)) row[key] = ((row[key] || "") + value).slice(-MAX_ROW_TEXT);
    else row[key] = value;
  }
  S.rows.set(row.id, row);
  return row;
}

function applySnapshot(raw) {
  rememberScroll();
  const data = JSON.parse(raw);
  S.task = data.task || "";
  S.agents = data.agents || [];
  S.rows.clear();
  for (const row of data.rows || []) S.rows.set(row.id, row);
  keepSelection();
  saveUiSelection();
  render();
}

function applyRow(raw) {
  rememberScroll();
  mergeRow(JSON.parse(raw));
  keepSelection();
  render();
}

function activeRows() {
  const rows = [...S.rows.values()].filter(row => row.agent_id === S.active);
  rows.sort((a, b) => (a.kind === "activity" ? -1 : b.kind === "activity" ? 1 : (a.number || 0) - (b.number || 0)));
  if (S.mode !== "improvements") return rows;
  const activity = rows.find(row => row.kind === "activity");
  return [activity, ...improvedAttempts(rows.filter(row => row.kind === "attempt"))].filter(Boolean);
}

function keepSelection() {
  if (!S.agents.some(agent => agent.id === S.active)) S.active = S.agents[0]?.id;
  if (!S.active) return;
  const rows = activeRows();
  if (!rows.some(row => row.id === S.selected[S.active])) {
    S.selected[S.active] = rows.find(row => row.kind === "activity")?.id || rows.at(-1)?.id || "";
  }
  if (S.active && !S.selected[S.active]) delete S.selected[S.active];
}

function selectedView() {
  const agent = S.agents.find(row => row.id === S.active);
  const rows = activeRows();
  const row = rows.find(item => item.id === S.selected[S.active]) || rows.find(item => item.kind === "activity") || rows.at(-1) || null;
  return { agent, rows, row, metric: metricView(rows) };
}

function metricView(rows) {
  const attempt = rows.find(row => row.kind === "attempt" && row.metric_label);
  return {
    label: attempt?.metric_label || "Score",
    axis: attempt?.axis_label || "score",
    attempts: rows.filter(row => row.kind === "attempt"),
  };
}

function render(options = {}) {
  S.chartWidth = Math.max(760, Math.floor(UI.chart.clientWidth || S.chartWidth));
  UI.session.textContent = S.task;
  const st = selectedView();
  UI.agents.innerHTML = agentNav(S.agents);

  if (!S.agents.length) {
    UI.empty.hidden = false;
    UI.empty.innerHTML = waiting("Waiting for an agent", "Runs will appear here as soon as an agent creates a log directory.");
    UI.viewBar.hidden = true;
    UI.viewBar.innerHTML = UI.chart.innerHTML = UI.table.innerHTML = UI.detail.innerHTML = "";
    return stopStream();
  }

  UI.empty.hidden = true;
  UI.viewBar.hidden = false;
  UI.viewBar.innerHTML = modeToggle();
  UI.chart.innerHTML = chart(st.agent, st.rows, st.metric);
  UI.table.innerHTML = table(st.rows, st.row, st.metric);

  if (st.row) {
    const tab = selectedTab(st.row);
    const current = UI.detail.querySelector("#experiment-detail");
    if (current?.dataset.row !== st.row.id || current?.dataset.tab !== tab) {
      UI.detail.innerHTML = detail(st.row, tab);
    } else {
      current.querySelector(".panel-meta").textContent = st.row.meta || st.row.label || st.row.id;
    }
    renderViewer(st.row, tab);
  } else {
    UI.detail.innerHTML = "";
    stopStream();
  }

  if (options.scrollToDetail) {
    requestAnimationFrame(() => document.querySelector("#experiment-detail")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }
}

function modeToggle() {
  const label = S.mode === "improvements" ? "All experiments" : "Improvements only";
  return `<button class="view-toggle" data-action="mode-toggle" title="Show ${H(label.toLowerCase())}">${label}</button>`;
}

function agentNav(agents) {
  return agents.map(agent => `
    <button class="${agent.id === S.active ? "active" : ""}" data-action="active" data-id="${H(agent.id)}">
      <span class="agent-status ${agent.live ? "running" : ""}" aria-hidden="true"></span>
      <span class="nav-copy"><strong>${H(agent.label)}</strong><span class="muted">${H(agent.meta)}</span></span>
    </button>`).join("");
}

function chart(agent, rows, metric) {
  const attempts = metric.attempts;
  const pts = attempts.filter(row => row.score != null);
  if (!pts.length) return `<section class="chart-panel waiting"><div class="panel-head"><h2>${H(agent?.label || "Run")} waiting</h2><span class="status-pill ${agent?.status === "running" ? "running" : ""}">${H(agent?.status || "")}</span></div><p class="muted">Waiting for the first metric.</p></section>`;
  const w = S.chartWidth, h = 230, m = { l: 92, r: 16, t: 22, b: 54 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const count = Math.max(attempts.length, ...pts.map(row => row.number || 1));
  const attemptTicks = ticksForAttempts(count);
  const x = (n) => m.l + pw * (count < 2 ? .5 : (n - 1) / (count - 1));
  const scale = yScale(pts.map(row => row.score));
  const xy = pts.map(row => [x(row.number || 1), m.t + ph - scale.norm(row.score) * ph, row]);
  const line = (x1, y1, x2, y2, c) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${c}"/>`;
  const text = (x, y, s, a = "middle", c = "axis-tick", extra = "") => `<text x="${x}" y="${y}" text-anchor="${a}" class="${c}" ${extra}>${H(s)}</text>`;
  return `<section class="chart-panel"><div class="chart"><svg width="${w}" height="${h}">
    ${scale.ticks.map(v => line(m.l, m.t + ph - scale.norm(v) * ph, m.l + pw, m.t + ph - scale.norm(v) * ph, "grid-line")).join("")}
    ${attemptTicks.map(n => line(x(n), m.t, x(n), m.t + ph, "grid-line")).join("")}
    ${line(m.l, m.t, m.l, m.t + ph, "axis-line")}${line(m.l, m.t + ph, m.l + pw, m.t + ph, "axis-line")}
    ${scale.ticks.map(v => text(m.l - 16, m.t + ph - scale.norm(v) * ph + 4, fmtAxis(v, scale.step), "end")).join("")}
    <polyline points="${xy.map(([x, y]) => `${x},${y}`).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    ${attemptTicks.map(n => text(x(n), m.t + ph + 24, `A${n}`)).join("")}
    ${xy.map(([x, y, row]) => `<circle class="chart-point" cx="${x}" cy="${y}" r="5" data-action="select" data-run="${H(row.id)}"><title>${H(row.label)} · ${H(metric.label)} ${fmt(row.score)}</title></circle>`).join("")}
    ${text(12, m.t + ph / 2, metric.axis, "middle", "axis-title", `transform="rotate(-90 12 ${m.t + ph / 2})"`)}
    ${text(m.l + pw / 2, h - 8, "attempt", "middle", "axis-title")}
  </svg></div></section>`;
}

function table(rows, selected, metric) {
  const body = rows.length ? rows.map(row => {
    const selectedClass = selected?.id === row.id ? "selected" : "";
    const number = row.kind === "activity"
      ? (row.live ? `<span class="live-pulse" title="Live run"></span>` : "log")
      : H(row.number);
    return `<tr class="${selectedClass}" data-action="select" data-run="${H(row.id)}">
      <td class="experiment-id attempt-number">${number}</td>
      <td>${row.kind === "activity" ? "" : fmt(row.score)}</td>
      <td class="description-cell">${H(row.description || "")}</td>
    </tr>`;
  }).join("") : `<tr><td class="empty-row" colspan="3">No visible experiments. Toggle attempts to show more.</td></tr>`;
  return `<section class="experiment-index"><div class="table-wrap"><table><thead><tr>${["#", metric.label, "What changed"].map(h => `<th>${H(h)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div></section>`;
}

function detail(row, selected = selectedTab(row)) {
  return `<section class="experiment-panel detail-panel" id="experiment-detail" data-row="${H(row.id)}" data-tab="${H(selected)}">
    <div class="experiment-viewer"><div class="tabs">${row.tab_order.map(key => `<button class="${key === selected ? "active" : ""}" data-action="tab" data-tab="${key}">${H(tabLabel(row, key))}</button>`).join("")}</div><div class="viewer" id="viewer"></div></div>
    <div class="panel-meta">${H(row.meta || row.label || row.id)}</div>
  </section>`;
}

function selectedTab(row) {
  const saved = S.tabs.__global;
  if (row.tab_order.includes(saved)) return saved;
  return row.tab_order.includes("agent") ? "agent" : row.tab_order[0];
}

function tabLabel(row, tab) {
  if (tab === "agent") return "Agent";
  if (tab === "notes") return "Notes";
  return row.artifacts?.[tab]?.label || tab;
}

function renderViewer(row, tab) {
  const v = document.querySelector("#viewer");
  if (!v) return stopStream();
  const key = `${row.id}:${tab}`;
  const selector = tab === "notes" ? ".markdown-panel" : tab === "diff" ? ".diff-panel" : "pre.log";
  const existing = v.querySelector(selector);
  if (existing?.dataset.key === key) {
    if (tab === "diff") fetchDiff(row, existing, key);
    else updateViewerNode(existing, row, tab);
    return;
  }
  stopStream();
  const panel = viewerPanel(row, tab, key);
  v.replaceChildren(panel);
  if (tab === "diff") {
    fetchDiff(row, panel, key);
    return;
  }
  updateViewerNode(v.querySelector(selector), row, tab);
}

function viewerPanel(row, tab, key) {
  if (tab === "diff") return diffPanel(row, key);
  const panel = document.createElement("div");
  panel.className = "log-panel";
  panel.innerHTML = `<div class="log-toolbar"><span class="log-status ${tabLive(row, tab) ? "live" : ""}">${H(statusText(row, tab))}</span></div>`;
  const node = document.createElement(tab === "notes" ? "div" : "pre");
  node.className = tab === "notes" ? "markdown-panel" : "log";
  node.dataset.key = key;
  bindScroll(node, key);
  panel.append(node);
  return panel;
}

function updateViewerNode(node, row, tab) {
  if (!node) return;
  const key = `${row.id}:${tab}`;
  if (tab === "agent") {
    if (row.artifacts?.agent) {
      hydrateArtifact(row, tab, node);
      return;
    }
    replaceText(node, key, "agent", esc(row.agent || ""));
    updateStatus(node, row, tab);
    return;
  }
  if (tab === "notes") {
    replaceText(node, key, "markdown", row.notes || "");
    updateStatus(node, row, tab);
    return;
  }
  if (tab === "logs") {
    hydrateArtifact(row, tab, node);
  }
}

function statusText(row, tab) {
  if (tab === "agent") {
    const state = textState(`${row.id}:${tab}`);
    if (state.error) return state.error;
    return `agent: ${row.artifacts?.agent && !state.loaded ? "loading" : tabLive(row, tab) ? "live" : "full"}`;
  }
  if (tab === "notes") return "notes";
  const state = textState(`${row.id}:${tab}`);
  if (state.error) return state.error;
  return `${tab}: ${state.loaded ? tabLive(row, tab) ? "live" : "full" : "loading"}`;
}

function tabLive(row, tab) {
  return Boolean(row.artifacts?.[tab]?.live || (tab === "agent" && !row.artifacts?.agent && row.live));
}

function updateStatus(node, row, tab, extra = "") {
  const status = node.closest(".log-panel")?.querySelector(".log-status");
  if (!status) return;
  status.textContent = extra || statusText(row, tab);
  status.className = `log-status ${tabLive(row, tab) ? "live" : ""}`;
}

async function hydrateArtifact(row, tab, node) {
  const key = `${row.id}:${tab}`;
  const state = textState(key);
  const artifact = row.artifacts?.[tab];
  if (!artifact) return;
  if (state.loaded && state.partial && !artifact.live) {
    state.text = "";
    state.offset = 0;
    state.loaded = false;
    state.partial = false;
  }
  if (!state.loaded && !state.pending) {
    state.error = "";
    const url = artifact.live ? `${artifact.url}&tail_bytes=${LIVE_TAIL_BYTES}` : artifact.url;
    state.pending = fetch(url).then(async response => {
      if (!response.ok) throw new Error(`fetch failed: ${response.status}`);
      state.offset = Number(response.headers.get("X-OpenRL-End") || 0);
      state.text = esc(await response.text());
      state.loaded = true;
      state.partial = artifact.live;
    }).catch(err => {
      state.error = err.message;
    }).finally(() => {
      state.pending = null;
    });
  }
  if (state.pending) await state.pending;
  if (!node.isConnected) return;
  if (state.error) {
    updateStatus(node, row, tab);
    return;
  }
  replaceText(node, key, "logs", state.text);
  updateStatus(node, row, tab);
  if (row.artifacts?.[tab]?.live) ensureStream(row, tab, node);
}

function ensureStream(row, tab, node) {
  const artifact = row.artifacts?.[tab];
  if (!artifact?.tail_url) return;
  const key = `${row.id}:${tab}`;
  if (S.stream?.key === key) {
    S.stream.node = node;
    return;
  }
  stopStream();
  const state = textState(key);
  const source = new EventSource(`${artifact.tail_url}&offset=${state.offset || 0}`);
  S.stream = { key, source, node };
  source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const text = esc(data.text || "");
    state.text += text;
    state.offset = data.end;
    appendText(S.stream?.node, key, "logs", text);
    updateStatus(S.stream?.node, row, tab, `${tab}: live`);
  };
  source.onerror = () => {
    if (S.stream?.source !== source) return;
    updateStatus(S.stream?.node, row, tab, "reconnecting");
    stopStream();
    setTimeout(() => {
      if (node.isConnected) ensureStream(row, tab, node);
    }, 1000);
  };
}

function stopStream() {
  S.stream?.source.close();
  S.stream = null;
}

function diffPanel(row, key) {
  const panel = document.createElement("div");
  panel.className = "diff-panel";
  panel.dataset.key = key;
  const state = textState(key);
  if (state.loaded) {
    const box = document.createElement("div");
    box.className = "diffbox";
    box.innerHTML = diffHtml(state.text);
    panel.append(box);
    return panel;
  }
  panel.append(fragment(`<div class="waiting-panel">${waiting("Diff", row.artifacts?.diff ? "Loading captured code diff." : "No code diff.")}</div>`));
  return panel;
}

async function fetchDiff(row, panel, key) {
  const artifact = row.artifacts?.diff;
  if (!artifact) return;
  const state = textState(key);
  if (state.loaded) return;
  if (!state.pending) {
    state.error = "";
    state.pending = fetch(artifact.url).then(async response => {
      if (!response.ok) throw new Error(`fetch failed: ${response.status}`);
      state.text = await response.text();
      state.loaded = true;
    }).catch(err => {
      state.error = err.message;
    }).finally(() => {
      state.pending = null;
    });
  }
  await state.pending;
  if (!panel.isConnected) return;
  if (state.error) {
    panel.replaceChildren(fragment(`<div class="waiting-panel">${waiting("Diff", state.error)}</div>`));
    return;
  }
  if (!state.loaded) return;
  const rendered = diffPanel(row, key);
  panel.replaceChildren(...rendered.childNodes);
}

function renderText(node, format, text) {
  if (format === "markdown") node.innerHTML = markdown(text);
  else colorize(node, text);
}

function replaceText(node, key, format, text) {
  withScroll(node, key, () => renderText(node, format, text));
}

function appendText(node, key, format, text) {
  if (!node?.isConnected) return;
  withScroll(node, key, () => {
    if (format === "markdown") renderText(node, format, text);
    else if (needsRichLogRender(text)) node.insertAdjacentHTML("beforeend", text.split("\n").map(logLine).join("\n"));
    else node.append(document.createTextNode(text));
  });
}

function withScroll(node, key, mutate) {
  const scroll = savedOrCurrentScroll(node, key);
  mutate();
  restoreScroll(node, key, scroll);
}

function bindScroll(node, key) {
  node.tabIndex = 0;
  const unpin = () => { textState(key).scroll = { ...logScroll(node), pinned: false }; };
  node.addEventListener("scroll", () => { textState(key).scroll = logScroll(node); }, { passive: true });
  node.addEventListener("wheel", event => { if (event.deltaY < 0) unpin(); }, { passive: true });
  node.addEventListener("touchstart", unpin, { passive: true });
  node.addEventListener("keydown", event => { if (["ArrowUp", "PageUp", "Home"].includes(event.key)) unpin(); });
}

function savedOrCurrentScroll(node, key) {
  const saved = textState(key).scroll;
  return saved && !saved.pinned ? saved : logScroll(node);
}

function rememberScroll() {
  document.querySelectorAll("pre.log[data-key],.markdown-panel[data-key]").forEach(node => { textState(node.dataset.key).scroll = logScroll(node); });
}

function logScroll(node) {
  const bottomGap = Math.max(0, node.scrollHeight - node.clientHeight - node.scrollTop);
  return { top: node.scrollTop, pinned: bottomGap <= 8 };
}

function restoreScroll(node, key, scroll = textState(key).scroll) {
  const apply = () => { node.scrollTop = !scroll || scroll.pinned ? node.scrollHeight : scroll.top; };
  apply();
  requestAnimationFrame(apply);
}

function colorize(node, text) {
  if (!needsRichLogRender(text)) {
    node.textContent = text;
    return;
  }
  node.innerHTML = text.split("\n").map(logLine).join("\n");
}

function needsRichLogRender(text) {
  return /[\x1b{}┏┡└┗┣┠┯┷─━│┃]/.test(text);
}

function logLine(line) {
  const event = agentEventLine(line);
  if (event) return event;
  if (line.includes("\x1b[")) return ansiLine(line);
  const plain = line.replace(/\x1b\[[0-9;]*m/g, "");
  if (!/^[┏┡└┗┣┠┯┷─━│┃]/.test(plain)) return H(line);
  if (plain.startsWith("│ ")) {
    const parts = plain.split("│");
    if (parts.length >= 4) return `<span class="metric-row">│<span class="metric-key">${H(parts[1])}</span>│<span class="metric-value">${H(parts[2])}</span>│</span>`;
  }
  return `<span class="metric-border">${H(plain)}</span>`;
}

const EVENT_RENDERERS = {
  init: (e) => eventHtml("init", e.model || "agent", e.session_id ? `session ${e.session_id}` : ""),
  message: messageEvent,
  tool_use: (e) => eventHtml("tool", e.tool_name || "tool", toolDetail(e.tool_name, e.parameters)),
  tool_result: (e) => eventHtml("result", e.status || "done", compact(e.output || e.result || e.content || "")),
};

function agentEventLine(line) {
  const text = line.trim();
  if (!text.startsWith("{") || !text.endsWith("}")) return "";
  let event;
  try { event = JSON.parse(text); } catch { return ""; }
  if (!event?.type) return "";
  const render = EVENT_RENDERERS[event.type];
  if (render) return render(event);
  return eventHtml(event.type.replaceAll("_", " "), event.status || event.role || "event", compact(event.content || event.message || ""));
}

function messageEvent(event) {
  const role = event.role || "agent", content = eventText(event.content);
  if (role === "user" && content.length > 2000) return eventHtml("prompt", "loaded", `${content.length.toLocaleString()} chars`);
  return eventHtml(role, content ? compact(content, 1800) : "message");
}

function toolDetail(tool, params = {}) {
  if (tool === "run_shell_command" && params.command) return `$ ${compact(params.command, 1200)}`;
  if (tool === "read_file" && params.file_path) return params.file_path;
  if (tool === "replace") return [params.file_path, params.instruction].filter(Boolean).join(" · ");
  if (tool === "update_topic") return [params.title, params.summary || params.strategic_intent].filter(Boolean).join("\n");
  return compact(Object.entries(params || {}).map(([k, v]) => `${k}=${eventText(v)}`).join(" "), 1200);
}

function eventHtml(kind, title, detail = "") {
  return `<span class="agent-event"><span class="event-kind">${H(kind)}</span><span class="event-title">${H(title)}</span>${detail ? `<span class="event-detail">${H(detail)}</span>` : ""}</span>`;
}

function eventText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(eventText).filter(Boolean).join("\n");
  if (typeof value === "object") return value.text || value.content || value.output || JSON.stringify(value);
  return String(value);
}

function compact(value, limit = 1000) {
  const text = eventText(value).replace(/\r\n/g, "\n").trim();
  return text.length > limit ? `${text.slice(0, limit)} ...` : text;
}

function ansiLine(line) {
  let out = "", index = 0, open = false;
  for (const match of line.matchAll(/\x1b\[([0-9;]*)m/g)) {
    out += H(line.slice(index, match.index));
    if (open) out += "</span>";
    const codes = (match[1] || "0").split(";").filter(Boolean).map(Number);
    const color = [...codes].reverse().find(c => ansi.has(c));
    open = !codes.includes(0) && !!color;
    if (open) out += `<span style="color:${ansi.get(color)};${codes.includes(1) ? "font-weight:700" : ""}">`;
    index = match.index + match[0].length;
  }
  return out + H(line.slice(index)) + (open ? "</span>" : "");
}

function markdown(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let out = "", list = false, code = false, buf = [];
  const inline = (s) => H(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const closeList = () => { if (list) { out += "</ul>"; list = false; } };
  const flushCode = () => { out += `<pre>${H(buf.join("\n"))}</pre>`; buf = []; };
  for (const line of lines) {
    if (line.startsWith("```")) { code ? flushCode() : closeList(); code = !code; continue; }
    if (code) { buf.push(line); continue; }
    if (!line.trim()) { closeList(); continue; }
    const h = /^(#{2,4})\s+(.+)$/.exec(line);
    if (h) { closeList(); out += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; continue; }
    const li = /^[-*]\s+(.+)$/.exec(line);
    if (li) { if (!list) { out += "<ul>"; list = true; } out += `<li>${inline(li[1])}</li>`; continue; }
    closeList(); out += `<p>${inline(line)}</p>`;
  }
  if (code) flushCode();
  closeList();
  return out || `<p class="muted">No notes yet.</p>`;
}

function improvedAttempts(attempts) {
  let best = -Infinity;
  const kept = [];
  for (const row of attempts) {
    const rank = row.score == null ? -Infinity : row.score_mode === "min" ? -row.score : row.score;
    if (rank === -Infinity) {
      if (row.status === "running") kept.push(row);
      continue;
    }
    if (rank > best) {
      kept.push(row);
      best = rank;
    }
  }
  return kept;
}

function ticksForAttempts(attempts) {
  if (attempts <= 12) return Array.from({ length: attempts }, (_, i) => i + 1);
  const step = Math.ceil(attempts / 12);
  const ticks = Array.from({ length: Math.ceil(attempts / step) }, (_, i) => 1 + i * step);
  if (ticks.at(-1) !== attempts) ticks.push(attempts);
  return ticks;
}

function yScale(values) {
  let lo = Math.min(...values), hi = Math.max(...values);
  if (lo === hi) {
    const pad = Math.max(Math.abs(lo) * .1, .01);
    lo -= pad; hi += pad;
  }
  const pad = (hi - lo) * .08;
  lo -= pad; hi += pad;
  const step = niceStep(hi - lo);
  const min = Math.floor(lo / step) * step;
  const max = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = min; v <= max + step / 2; v += step) ticks.push(Math.abs(v) < step / 1000 ? 0 : v);
  return { ticks, step, norm: (v) => (v - min) / (max - min || 1) };
}

function niceStep(span) {
  const raw = span / 4;
  const pow = 10 ** Math.floor(Math.log10(raw || 1));
  const unit = raw / pow;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * pow;
}

function fmtAxis(value, step) {
  if (Math.abs(value) < step / 1000) value = 0;
  const decimals = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step)) + 1));
  return Number(value.toFixed(decimals)).toString();
}

function fmt(v) {
  return v == null ? "" : Number.isFinite(v) ? Number(v.toPrecision(4)).toString() : String(v);
}

function waiting(title, text) {
  return `<div class="waiting-row"><span class="live-dot"></span><div><strong>${H(title)}</strong><p class="muted">${H(text)}</p></div></div>`;
}

function fragment(html) {
  const t = document.createElement("template");
  t.innerHTML = html;
  return t.content;
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;

  switch (target.dataset.action) {
    case "active":
      updateUiSelection({ active: target.dataset.id });
      break;
    case "mode-toggle":
      updateUiSelection({ mode: S.mode === "improvements" ? "all" : "improvements" });
      break;
    case "select":
      updateUiSelection({ selected: { ...S.selected, [S.active]: target.dataset.run } }, { scrollToDetail: true });
      break;
    case "tab": {
      const row = selectedView().row;
      if (row) updateUiSelection({ tabs: { ...S.tabs, __global: target.dataset.tab } });
      break;
    }
    case "collapse": {
      const file = target.closest(".diff-file");
      const collapsed = file.classList.toggle("collapsed");
      target.classList.toggle("collapsed", collapsed);
      target.title = collapsed ? "Expand file" : "Collapse file";
      break;
    }
  }
});

function connect() {
  S.events?.close();
  S.events = new EventSource("events");
  S.events.addEventListener("snapshot", event => applySnapshot(event.data));
  S.events.addEventListener("row", event => applyRow(event.data));
  S.events.onerror = () => {
    UI.empty.hidden = false;
    UI.empty.innerHTML = waiting("UI error", "Lost connection to observer.");
  };
}

addEventListener("beforeunload", () => {
  stopStream();
  S.events?.close();
});
addEventListener("resize", () => render());
connect();
