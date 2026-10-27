<script>
  import Chart from "./Chart.svelte";
  import QueryChart from "./QueryChart.svelte";
  import { createQuery, inspectUrl, viewLink } from "./state.svelte.js";
  import { opLabel, opRank } from "./visualization.js";
  let { run, view, range, bounds, selection, finished, navigating = false } = $props();
  // Every recorded step of the run, charted by step rather than by the time window.
  const steps = createQuery(() => inspectUrl(`runs/${encodeURIComponent(view.id)}/steps`, { last: 1000, fields: "*" }), { poll: 30000, pollWhen: () => !finished });
  const rows = $derived(steps.data?.steps || []);
  // Until the first steps arrive, hold each section's space with empty charts so nothing below jumps when they land.
  const loading = $derived(!steps.data && !steps.error);
  const stepRange = $derived(rows.length ? { start: rows[0].step, end: rows.at(-1).step } : null);
  const byStep = (value) => rows.map((row) => [row.step, value(row)]).filter(([, v]) => Number.isFinite(v));
  // Each op's metrics arrive as min, max and mean over the step's calls; the means are charted, of per-call means such as
  // loss:mean and of plain values such as time/optimizer_step.
  const trainerCharts = $derived([...new Map(rows.flatMap((row) => Object.entries(row.ops).flatMap(([op, entry]) => Object.keys(entry.metrics).filter((name) => !name.includes(":") || name.endsWith(":mean")).map((name) => [`${op} ${name}`, [op, name]])))).values()]
    .sort(([a, x], [b, y]) => opRank(a) - opRank(b) || x.localeCompare(y)));
  const FIRST = [["env/all/reward/total", "Reward"], ["test/env/all/reward/total", "Held-out reward"], ["env/all/lab/criteria_pass_fraction", "Criteria pass rate"],
    ["test/env/all/lab/criteria_pass_fraction", "Held-out criteria pass rate"], ["optim/kl_sample_train_v2", "Sampler-trainer KL"], ["optim/entropy", "Entropy"], ["time/total", "Step time", "s"]];
  const allClient = $derived.by(() => {
    const names = new Set(rows.flatMap((row) => Object.keys(row.client || {})));
    return [...FIRST.filter(([name]) => names.has(name)), ...[...names].filter((name) => !FIRST.some(([first]) => first === name)).sort().map((name) => [name, name])];
  });
  // A client can report a hundred metrics; show the headline ones first and let the rest be picked by group or name.
  const groupOf = (name) => (name.includes("/") ? name.split("/")[0] : "other");
  const groups = $derived([...new Set(allClient.map(([name]) => groupOf(name)))].sort());
  let group = $state("key");
  let search = $state("");
  const clientCharts = $derived.by(() => {
    const terms = search.toLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length) return allClient.filter(([name, title]) => terms.every((term) => `${name} ${title}`.toLowerCase().includes(term)));
    if (group === "key") return allClient.filter(([name]) => FIRST.some(([first]) => first === name));
    return group === "all" ? allClient : allClient.filter(([name]) => groupOf(name) === group);
  });
  // Each op's mean duration per step, from the same step rows.
  const opNames = $derived([...new Set(rows.flatMap((row) => Object.keys(row.ops)))].sort((a, b) => opRank(a) - opRank(b) || a.localeCompare(b)));
  // The vLLM series of a run's samplers, as vLLM's and llm-d's dashboards chart them.
  const SAMPLER_CHARTS = [["generation_tokens_per_second", "Generated tokens", "tok/s"], ["prompt_tokens_per_second", "Prompt tokens", "tok/s"], ["requests_running", "Running requests", ""], ["requests_waiting", "Waiting requests", ""],
    ["kv_cache_usage", "KV cache use", "%", 100], ["preemptions_per_second", "Preemptions", "/s"], ["time_to_first_token_p99_seconds", "Time to first token p99", "s"], ["request_latency_p99_seconds", "Request latency p99", "s"]];
</script>

{#snippet placeholders(count)}<div class="chart-grid">{#each Array.from({ length: count }) as _, i (i)}<Chart title="" points={[]} empty="Loading…" />{/each}</div>{/snippet}

<div class="run-metrics" data-run-metrics={view.id}>
  {#if steps.error}<p class="source-error" role="status">{steps.error}{rows.length ? " · Showing previously fetched steps" : ""}</p>{/if}
  <h2 class="scheduler-title">Client metrics</h2>
  {#if allClient.length}<div class="metric-filters">
    <input type="search" placeholder={`Filter ${allClient.length} metrics by name`} aria-label="Filter client metrics" bind:value={search} />
    <div class="run-filters" role="group" aria-label="Client metric group">
      {#each ["key", ...groups, "all"] as key (key)}<button type="button" class="chip" aria-pressed={!search && group === key} onclick={() => { group = key; search = ""; }}>{{ key: "Key", all: "All" }[key] || key} <span class="run-filter-count">{key === "key" ? allClient.filter(([name]) => FIRST.some(([first]) => first === name)).length : key === "all" ? allClient.length : allClient.filter(([name]) => groupOf(name) === key).length}</span></button>{/each}
    </div>
  </div>{/if}
  {#if clientCharts.length}<div class="chart-grid">
    {#each clientCharts as [key, title, unit = ""] (key)}<Chart {title} {unit} steps points={byStep((row) => row.client?.[key])} start={stepRange.start} end={stepRange.end} empty="No values" />{/each}
  </div>{:else if allClient.length}<p class="muted">No client metric matches</p>{:else if loading}{@render placeholders(FIRST.length)}{:else if steps.data}<p class="muted">0 metrics</p>{/if}
  <h2 class="scheduler-title">Trainer metrics</h2>
  {#if trainerCharts.length}<div class="chart-grid">
    {#each trainerCharts as [op, name] (`${op} ${name}`)}<Chart title={`${name} · ${opLabel(op)}`} steps points={byStep((row) => row.ops[op]?.metrics[name]?.mean)} start={stepRange.start} end={stepRange.end} tone="accent" empty="No values" />{/each}
  </div>{:else if loading}{@render placeholders(3)}{:else if steps.data}<p class="muted">0 metrics</p>{/if}
  <h2 class="scheduler-title">Operations</h2>
  <div class="run-metric-summary"><span>Completed steps <strong>{run.steps ?? "—"}</strong></span></div>
  {#if loading}{@render placeholders(3)}{/if}
  <div class="chart-grid">
    {#each opNames as op (op)}<Chart title={`${opLabel(op)} duration`} unit="s" steps points={byStep((row) => row.ops[op] && row.ops[op].seconds / row.ops[op].count)} start={stepRange.start} end={stepRange.end} min={0} tone="accent" empty="No values" />{/each}
    {#if steps.data && !rows.length}<p class="empty-state">No operations recorded</p>{/if}
  </div>
  <h2 class="scheduler-title">Samplers</h2>
  <!-- A shared pool's load is every run's, so it is charted on Nodes rather than here. -->
  {#if run.shared_runtime}<p class="muted">The pool's samplers serve every run on this base model; see their load on <a href={viewLink("nodes")}>Nodes</a>.</p>
  {:else}<div class="chart-grid">
    {#each SAMPLER_CHARTS as [key, title, unit, scale = 1] (key)}<QueryChart name={`vllm_${key}`} params={bounds} {view} {range} {selection} {title} {unit} {scale} {navigating} />{/each}
  </div>{/if}
</div>
