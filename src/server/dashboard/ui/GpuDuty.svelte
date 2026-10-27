<script>
  import Chart from "./Chart.svelte";
  import { createQuery, inspectUrl } from "./state.svelte.js";
  import { DCGM_READING_SECONDS, dutyBands } from "./visualization.js";
  // One GPU's duty cycle, filled with the color of whichever process held it, from every run on its node.
  let { node, device, range } = $props();
  const bounds = () => ({ node_id: node, since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() });
  const options = { poll: 15000, pollWhen: () => range.live, remember: () => !range.live, key: () => `${node} ${device.id} ${range.live ? "live" : range.now} ${Math.round(range.now - range.start)}` };
  const utilization = createQuery(() => inspectUrl("query", { name: "gpu_util", ...bounds() }), options);
  const holders = createQuery(() => inspectUrl("activity", bounds()), options);
  const points = $derived(utilization.data?.series?.find((s) => s.labels.UUID === device.id)?.points || []);
</script>

<Chart title={`Duty cycle · ${device.label}`} unit="%" {points} start={range.start} end={range.now} min={0} max={100} readingSeconds={DCGM_READING_SECONDS} bands={dutyBands(holders.data?.placements || [], device.id)} empty={utilization.error || (!utilization.data && "Loading…") || "No samples for this GPU"} />
