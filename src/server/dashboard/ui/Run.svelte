<script>
  import { createQuery, inspectUrl, now, rangeOf, viewLink, saved, MAX_WINDOW } from "./state.svelte.js";
  import { nodeTime } from "./node-time.js";
  import { clipPlacements } from "./visualization.js";
  import TimeControl from "./TimeControl.svelte";
  import Status from "./Status.svelte";
  import Metrics from "./Metrics.svelte";
  import Activity from "./Activity.svelte";
  import Logs from "./Logs.svelte";
  import CopyLink from "./CopyLink.svelte";
  import { KINDS, runDescription, isFinished, finishedAt, runLabel } from "./format.js";
  let { inventory, view } = $props();
  // Draw gestures immediately, but fetch only once the user settles on a window.
  let navigating = $state(false);
  const observed = $derived(now(inventory));
  const range = $derived(rangeOf(inventory, view));
  const validEvent = $derived(view.event === null || (view.event >= range.start && view.event <= range.now));
  $effect(() => { if (!validEvent) view.event = null; });
  const bounds = $derived({ since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString(), run_id: view.id });
  const url = $derived(inspectUrl("activity", bounds));
  const selection = () => !navigating && `${view.id} ${view.duration} ${view.end}`;
  const tab = $derived(["metrics", "logs"].includes(view.tab) ? view.tab : "activity");
  // Only the Activity tab draws from the run's activity.
  const detail = createQuery(() => url, { poll: 15000, remember: () => view.end !== null, pollWhen: () => view.end === null && tab === "activity", key: () => tab === "activity" && selection() });
  const run = $derived(inventory.runs.find((r) => r.run_id === view.id));
  const error = $derived(detail.error);
  const title = $derived(run ? [(run.model || "Run").split("/").at(-1), run.run_id.slice(0, 8), KINDS[run.fine_tuning_type]].filter(Boolean).join(" · ") : "Run");
  // The saved settings worth reading, after the recipe; the title already names the training type.
  function setupOf(s) {
    const gpus = s["openrl.trainer_gpus"], cp = s["openrl.trainer_cp"], samplers = s["openrl.sampler_replicas"];
    return [
      { automodel: "Automodel", pytorch: "PyTorch" }[s["openrl.trainer_backend"]] || s["openrl.trainer_backend"],
      gpus != null && `${gpus} trainer GPU${gpus === 1 ? "" : "s"}${cp > 1 ? ` (CP ${cp})` : ""}`,
      samplers != null && `${samplers} sampler${samplers === 1 ? "" : "s"}`,
      s["openrl.exclusive"] && "exclusive GPUs",
      // A LoRA run always ships its adapter; the weight sync strategy only applies to full fine-tuning.
      s["openrl.fine_tuning_type"] === "full" && s["openrl.weight_sync"] && `${s["openrl.weight_sync"]} weight sync`,
    ];
  }
  const description = $derived(run ? [runDescription(run, title), ...setupOf(run.settings || {})].filter(Boolean).join(" · ") : "");
  // Duty is one number per GPU over the window, keyed by UUID.
  const busy = createQuery(() => inspectUrl("query", { name: "gpu_busy", ...bounds, points: 1 }), { scope: () => view.id, poll: 15000, pollWhen: () => view.end === null, remember: () => view.end !== null, key: () => tab === "activity" && selection() });
  const duty = $derived(Object.fromEntries((busy.data?.series || []).filter((s) => s.points.length).map((s) => [s.labels.UUID, Math.round(s.points.at(-1)[1])])));
  // DCGM's slot index names a GPU whose node is gone.
  const slots = $derived(Object.fromEntries((busy.data?.series || []).filter((s) => s.labels.gpu != null).map((s) => [s.labels.UUID, s.labels.gpu])));
  // A finished run opens on its own lifetime rather than the last 30 minutes, unless the link names a window.
  const finished = $derived(isFinished(run));
  let opened = false;
  $effect(() => {
    if (!run || opened) return;
    opened = true;
    const stop = finishedAt(run);
    if (!finished || view.explicitWindow || view.end !== null || view.duration !== 1800 || !Number.isFinite(stop)) return;
    const span = Math.max(0, stop - (run.created_at || stop));
    view.duration = Math.max(1800, Math.min(MAX_WINDOW, Math.round(span * 1.05)));
    view.end = Math.min(observed, Math.round(stop + Math.max(60, span * 0.025)));
  });
  // Clipped once per response or window change, not on every render of a long run's lanes.
  const clipped = $derived(clipPlacements(detail.data?.placements || [], range, detail.error, detail.status));
  const link = (tab, changes = {}) => viewLink(`run/${encodeURIComponent(view.id)}/${tab}`, { duration: view.duration, end: view.end, back: view.back, event: view.event, q: view.q, pod: view.pod, ...changes });
</script>

<p class="overview-back"><a href={view.back || saved.overview}>← {view.back ? "Nodes" : "Overview"}</a></p>
{#if !run}
  <h1 class="heading">Run</h1>
  {#if error}<p class="source-error" role="status">{error}</p>{:else}<p class="empty-state">{view.id ? "Loading run…" : "Choose a run from Overview"}</p>{/if}
  {#if detail.error}<button type="button" class="chip" onclick={detail.retry}>Retry</button>{/if}
{:else}
  <div class="run-heading"><h1 class="heading" title={run.run_id}>{title}</h1><Status value={runLabel(run)} /></div>
  {#if error}<p class="source-error" role="status">{error}</p>{/if}
  {#if description}<p class="run-description">{description}</p>{/if}
  <section class="run-state" aria-label="Run state">
    {#each (run.pods || []).filter((p) => p.problem) as pod}<p class="source-error" role="status">{pod.name}: {pod.problem}</p>{/each}
    {#if !finished && run.shared_runtime}<p class="muted">Shared runtime: its workers also serve other runs</p>{/if}
  </section>
  <div use:nodeTime={{ view, getRange: () => range, getNow: () => observed, onGesture: (value) => navigating = !!value }}>
  <div class="run-toolbar">
    <nav class="workspace-tabs" aria-label="Run views">
      <a href={link("activity")} aria-current={tab === "activity" ? "page" : undefined}>Activity</a>
      <a href={link("metrics")} aria-current={tab === "metrics" ? "page" : undefined}>Metrics</a>
      <a href={link("logs")} aria-current={tab === "logs" ? "page" : undefined}>Logs</a>
    </nav>
    <div class="time-control run-time-controls" aria-label="Run time range"><TimeControl {view} {range} {observed} /><CopyLink /><a href={url}>Agent JSON ↗</a></div>
  </div>
  <div id="run-panel">
    {#if tab === "logs"}
      {#if validEvent}<Logs {run} {view} {range} {navigating} allLogs={link("logs", { event: null, end: null })} />{/if}
    {:else if tab === "metrics"}<Metrics {run} {view} {range} {bounds} {selection} {finished} {navigating} />
    {:else}
      <div class="run-activity"><Activity placements={clipped} {range} {inventory} acrossNodes={true} {duty} {slots} error={detail.error} pending={detail.pending} {navigating} onpick={(at) => location.hash = link("logs", { event: at, end: range.now })} /></div>
    {/if}
  </div>
  </div>
{/if}
