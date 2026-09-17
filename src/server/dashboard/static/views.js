// Page renderers return markup; DOM updates stay in app.js.

import { escape, encode, empty, runStatus, elapsedTime, duration, shortNodeName } from "./ui.js";
import { chart, chartNumber } from "./charts.js";
import { route } from "./store.js";

export function runs(state) {
  const error = state.store_error ? `<p class="source-error" role="status">${escape(state.store_error)}</p>` : "";
  if (error && !state.runs.length) return `<h1 class="heading">Overview</h1>${error}`;
  const active = (r) => ["active", "running"].includes(String(r.status || "").toLowerCase());
  const count = (test) => state.runs.filter(test).length;
  const summary = [
    [count(active), "Active"],
    [count((r) => r.status === "failed"), "Failed"],
    [count((r) => ["completed", "ended"].includes(r.status)), "Finished"],
  ]
    .map(([n, label]) => `<span><strong>${n}</strong> ${label}</span>`)
    .join("");
  const rows = [...state.runs]
    .sort((a, b) => Number(active(b)) - Number(active(a)))
    .map((r) => {
      const label = [r.display_name, r.recipe_name].filter((v, i, a) => v && a.indexOf(v) === i).join(" · ");
      return `<a class="job-list-row" data-key="${escape(r.run_id)}" href="#run/${encode(r.run_id)}/activity"><span class="job-identity"><span>${escape((r.model || "Run").split("/").at(-1))} · <span class="mono">${escape(r.run_id.slice(0, 8))}</span></span>${label ? `<span class="muted micro">${escape(label)}</span>` : ""}</span><span>${runStatus(r.display_status || r.status)}</span><span>${escape({ lora: "LoRA", full: "FFT", fft: "FFT" }[r.fine_tuning_type] || r.fine_tuning_type || "—")}</span><span>${escape(r.steps ?? "—")}</span><span>${escape(elapsedTime(r, state.observed_at))}</span></a>`;
    })
    .join("");
  return `<h1 class="heading">Overview</h1><div class="overview-summary">${summary}</div>${error}
    <div class="job-list"><div class="job-list-head"><span>Job</span><span>Status</span><span>Training kind</span><span>Completed steps</span><span>Elapsed</span></div>${rows}</div>${!state.runs.length ? empty("No runs recorded") : ""}`;
}

export function scheduler(state) {
  const data = state.cluster.scheduler || {};
  if (!data.available) return `<h1 class="heading">Scheduler</h1>${empty(data.error || (data.installed === false ? "Scheduler is not installed" : "Scheduler data is unavailable"))}`;
  const workloads = data.workloads || [];
  const pending = workloads.filter((w) => !w.node_name && !["Completed", "Failed"].includes(w.phase));
  const runFor = (w) => state.runs.find((r) => (r.workloads || []).some((item) => item.uid === w.uid));
  const label = (w) => {
    const run = runFor(w);
    return run ? `<a href="#run/${encode(run.run_id)}/activity">${escape(run.name)} ↗</a>` : escape(w.model_id || w.name);
  };
  const role = (w) => ({ trainer: "Trainer", sampler: "Sampler" })[w.role] || "Unknown process";
  const rows = pending
    .map(
      (w) =>
        `<tr><td>${label(w)}<div class="muted">${role(w)}${w.owner_id ? ` · Owner ID: ${escape(w.owner_id)}` : ""}</div></td><td>${escape(w.requested_memory || "Not reported")}${w.exclusive ? " · Exclusive" : ""}</td><td>${escape(w.phase)}</td><td>${escape(w.placed_message || w.placed_reason || w.reason || "No placement reason reported")}</td></tr>`,
    )
    .join("");
  const reservations = (data.ledgers || [])
    .map((ledger) => {
      const seats = ledger.seats
        .map((seat) => {
          const w = workloads.find((item) => item.uid === seat.workload_uid);
          const placement = w && state.placements.find((item) => item.id === w.uid);
          return `<div class="scheduler-seat"><span>${w ? label(w) : escape(seat.workload)}</span><span class="muted">${w ? role(w) + " · " : ""}${seat.exclusive ? "Exclusive" : "Shared"}</span>${seat.owner ? `<span class="muted">Owner ID: ${escape(seat.owner)}</span>` : ""}${placement ? `<a href="#nodes" data-scheduler-placement="${escape(placement.id)}" title="${escape(placement.node)}">Node ${escape(shortNodeName(placement.node, state.cluster.nodes))} ↗</a>` : ""}</div>`;
        })
        .join("");
      return `<div class="scheduler-reservation" data-key="${escape(ledger.name || ledger.claim_name)}"><div>${escape(ledger.claim_name || ledger.name)}<div class="muted">${ledger.seats.length} reservation${ledger.seats.length === 1 ? "" : "s"}</div></div><div class="scheduler-seat-list">${seats}</div></div>`;
    })
    .join("");
  return `<h1 class="heading">Scheduler</h1>
    <div class="overview-summary"><span><strong>${pending.length}</strong> Pending</span><span><strong>${workloads.filter((w) => w.node_name).length}</strong> Assigned workloads</span></div>
    <h2 class="scheduler-title">Waiting for placement</h2>
    ${pending.length ? `<div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Workload</th><th>Request</th><th>State</th><th>Placement reason</th></tr></thead><tbody>${rows}</tbody></table></div>` : empty("Nothing is waiting for placement")}
    <h2 class="scheduler-title">Reservations</h2>${reservations || empty("No claim reservations reported")}
    <p class="muted scheduler-title"><a href="/api/v1/dashboard/snapshot">Inspect scheduler JSON ↗</a></p>`;
}

const FAILED_WAITING = new Set(["CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError", "CreateContainerError", "RunContainerError", "InvalidImageName", "ContainerCannotRun", "StartError"]);

function podTone(pod) {
  if (pod.phase === "Failed") return "error";
  const failing = (pod.containers || []).some(
    (c) => (c.state === "waiting" && FAILED_WAITING.has(c.reason)) || (c.state === "terminated" && ((c.exit_code != null && c.exit_code !== 0) || (c.reason && c.reason !== "Completed"))),
  );
  if (failing) return "error";
  const reason = String(pod.problem || "").split(":", 1)[0];
  return reason === "Failed" || FAILED_WAITING.has(reason) ? "error" : "warning";
}

const healthStatus = (label, tone) => `<span class="health-status health-${tone}"><span class="health-dot" aria-hidden="true"></span>${escape(label)}</span>`;

export function health(state) {
  const cluster = state.cluster;
  const errors = [...new Set([state.store_error, state.history_error, cluster.error, cluster.nodes_error, cluster.events_error, cluster.devices?.error, cluster.scheduler?.error].filter(Boolean))];
  const issues = [];
  for (const pod of cluster.pods || []) {
    if (!pod.problem) continue;
    const run = state.runs.find((r) => r.pods.some((p) => p.uid === pod.uid));
    const link = run ? `<a href="#run/${encode(run.run_id)}/logs">Logs ↗</a>` : `<a href="/api/v1/dashboard/pods/${encode(pod.name)}/logs">Logs ↗</a>`;
    issues.push([pod.problem, pod.name, `${pod.restarts || 0} restarts`, link, podTone(pod)]);
  }
  for (const node of cluster.nodes || []) if (node.ready !== true) issues.push(["Node not ready", node.name, "Ready condition is false or unknown", '<a href="#nodes">Nodes ↗</a>', "error"]);
  const rows = issues.map(([issue, resource, evidence, link, tone]) => `<tr><td>${healthStatus(issue, tone)}</td><td>${escape(resource)}</td><td>${escape(evidence)}</td><td>${link}</td></tr>`).join("");
  const complete = cluster.available === true && errors.length === 0;
  return `<h1 class="heading">Health</h1>${errors.map((error) => `<p class="health-message" role="status">${healthStatus("Source unavailable", "warning")}<span>${escape(error)}</span></p>`).join("")}
    ${issues.length ? `<div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Issue</th><th>Resource</th><th>Evidence</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` : complete ? `<p class="health-message">${healthStatus("Healthy", "success")}<span>No pod problems, node problems or source errors.</span></p>` : ""}
    <p class="run-json-link"><a href="/api/v1/dashboard/snapshot">Diagnostic JSON ↗</a> · <a href="/docs">API reference ↗</a></p>`;
}

// The recipe config carries a lora_rank even for full fine-tuning runs; the
// run directory name is the reliable signal the sweep scripts leave behind.
const kindLabel = (run) => (/(^|[-_])fft([-_]|$)/.test(run.name) || run.config.lora_rank == null ? "FFT" : `LoRA r${run.config.lora_rank}`);
const pct = (value) => (Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—");
const shortName = (name) => name.replace(/^gsm8k_rl_(mega|rank_sweep)_/, "");

function experimentCharts(run) {
  return ["reward", "correct"].map((key) => {
    const points = run.series[key] || [], percent = key === "correct";
    return chart({
      title: percent ? "Correctness" : "Reward",
      points: percent ? points.map(([step, value]) => [step, Number.isFinite(value) ? value * 100 : value]) : points,
      start: points[0]?.[0] ?? 0, end: points.at(-1)?.[0] ?? 0,
      unit: percent ? "%" : "", ...(percent ? { min: 0, max: 100 } : {}), tone: "accent", xFormat: "step",
    });
  }).join("");
}

// Training curves read from each run's metrics.jsonl on the shared volume,
// grouped by the sweep directory they were written under.
export function experiments(entry) {
  const data = entry.data;
  if (!data) return `<h1 class="heading">Experiments</h1>${empty(entry.error || "Loading run metrics…")}`;
  if (data.error) return `<h1 class="heading">Experiments</h1>${empty(data.error)}`;
  const ordered = [...data.runs].sort((a, b) => b.updated_at - a.updated_at);
  const selected = ordered.find((run) => run.path === route()[1]) || ordered.find((run) => ["reward", "correct"].some((key) => run.series[key]?.filter(([, value]) => Number.isFinite(value)).length > 1)) || ordered[0];
  const sweeps = new Map();
  for (const run of ordered) sweeps.set(run.sweep, [...(sweeps.get(run.sweep) || []), run]);
  const now = Date.now() / 1000;
  const sections = [...sweeps.entries()]
    .map(([sweep, members]) => {
      const rows = members
        .map((run) => {
          const charts = run === selected ? `<div class="chart-grid" data-key="charts:${escape(run.path)}">${experimentCharts(run)}</div>` : "";
          return `<a class="job-list-row experiment-row" data-key="${escape(run.path)}" href="#experiments/${encode(run.path)}"${run === selected ? ' aria-current="true"' : ""}><span class="job-identity"><span class="mono">${escape(shortName(run.name))}</span></span><span>${escape((run.config.model_name || "").split("/").at(-1))}</span><span>${escape(kindLabel(run))}</span><span>${run.step}${run.config.max_steps ? ` / ${run.config.max_steps}` : ""}</span><span>${Number.isFinite(run.last.reward) ? escape(chartNumber(run.last.reward)) : "—"}</span><span>${escape(pct(run.last.correct))}</span><span>${escape(pct(run.last.format))}</span></a>${charts}`;
        })
        .join("");
      return `<section class="experiment-sweep"><h2>${escape(sweep || "runs")} <span class="muted micro">${members.length} run${members.length === 1 ? "" : "s"} · updated ${duration(now - Math.max(...members.map((r) => r.updated_at)))} ago</span></h2>
        <div class="job-list"><div class="job-list-head experiment-head"><span>Run</span><span>Model</span><span>Kind</span><span>Step</span><span>Reward</span><span>Correct</span><span>Format</span></div>${rows}</div></section>`;
    })
    .join("");
  return `<h1 class="heading">Experiments</h1><p class="muted">Select a run to inspect reward and correctness.</p>${entry.error ? empty(`${entry.error} · Showing previously fetched metrics`) : ""}${sections}${!data.runs.length ? empty("No run metrics found under the runs directory") : ""}`;
}
