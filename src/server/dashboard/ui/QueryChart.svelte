<script>
  import Chart from "./Chart.svelte";
  import { createQuery, inspectUrl } from "./state.svelte.js";
  // One named query drawn as one chart, at about one point per pixel of the chart's width.
  let { name, params, view, range, selection, title, unit = "", scale = 1, navigating = false } = $props();
  let width = $state(0);
  const points = $derived(Math.max(30, Math.min(1000, Math.round(width / 50) * 50)));
  const result = createQuery(() => inspectUrl("query", { name, ...params, points }), { scope: () => params.run_id, poll: 15000, pollWhen: () => view.end === null, remember: () => view.end !== null, key: () => width > 0 && selection() && `${selection()} ${points}` });
  const series = $derived(result.data?.series?.[0]?.points || []);
</script>

<div bind:clientWidth={width}>
  <Chart {title} {unit} points={scale === 1 ? series : series.map(([at, value]) => [at, value * scale])} start={range.start} end={range.now} min={0} {navigating} empty={result.error || "No samples in this time range"} note={result.error && series.length ? `${result.error} · Showing previously fetched samples` : ""} />
</div>
