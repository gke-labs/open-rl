<script>
  import { tick, onDestroy } from "svelte";
  import Chart from "./Chart.svelte";
  import Activity from "./Activity.svelte";
  import CopyLink from "./CopyLink.svelte";
  import TimeControl from "./TimeControl.svelte";
  import { createQuery, inspectUrl, now, rangeOf, viewLink, crosshair, crosshairOffset, timeAt } from "./state.svelte.js";
  import { nodeTime } from "./node-time.js";
  import { axisTime, deviceLabel, acceleratorLabel, shortNodeName, placementColor, clipPlacements, claimLabel, capacityBars, activitySource, deviceSeries, gapFor, lastValue, DCGM_READING_SECONDS, dutyBands } from "./visualization.js";

  let { inventory, view } = $props();
  let element, navigating = $state(false);
  const observed = $derived(now(inventory));
  const range = $derived(rangeOf(inventory, view));
  const bounds = $derived({ since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() });
  const selection = () => !navigating && `${view.duration} ${view.end}`;
  const history = createQuery(() => inspectUrl("activity", bounds), { poll: 15000, pollWhen: () => view.end === null, remember: () => view.end !== null, key: selection });
  // Duty cycle: mean DCGM GPU utilization, the share of time a kernel was running, over each node's GPUs.
  const fleet = createQuery(() => inspectUrl("query", { name: "gpu_busy", points: 1, ...bounds }), { poll: 15000, pollWhen: () => view.end === null, remember: () => view.end !== null, key: selection });
  const busyByGpu = $derived(Object.fromEntries((fleet.data?.series || []).filter((s) => s.points.length).map((s) => [s.labels.UUID, s.points.at(-1)[1]])));
  function dutyCycle(node) {
    const values = (node.devices || []).map((d) => busyByGpu[d.id]).filter(Number.isFinite);
    return values.length ? `${Math.round(values.reduce((a, b) => a + b, 0) / values.length)}%` : "—";
  }
  // Lanes wait for the first history response so they render once, complete, instead of filling in and shifting.
  const loading = $derived(!history.data && !history.error);
  const all = $derived(clipPlacements(history.data?.placements || [], range, history.error, history.status));
  const node = $derived(inventory.cluster.nodes.find((n) => n.name === view.node));
  // The selected node's GPU and worker series, one named query each, joined into one row per GPU and per pod.
  const nodeQuery = (name) => createQuery(() => node ? inspectUrl("query", { name, node_id: node.name, ...bounds }) : null, { scope: () => node?.name ?? null, poll: 15000, pollWhen: () => view.end === null, remember: () => view.end !== null, key: () => node && !navigating && `${node.name} ${selection()}` });
  const queries = { utilization: nodeQuery("gpu_util"), memory_mib: nodeQuery("gpu_memory_mib") };
  function rows(fields, label) {
    const found = new Map();
    for (const field of fields) for (const s of queries[field].data?.series || []) found.set(s.labels[label], { id: s.labels[label], ...found.get(s.labels[label]), [field]: s.points });
    return [...found.values()];
  }
  const loaded = $derived(Object.values(queries).some((q) => q.data || q.error));
  const metrics = $derived({ devices: rows(["utilization", "memory_mib"], "UUID") });
  const selectedDevices = $derived(metrics.devices.filter((d) => view.gpu === "all" || view.gpu === d.id));
  const gpuPoints = $derived(deviceSeries(selectedDevices, "utilization"));
  const gpuMemory = $derived(lastValue(deviceSeries(selectedDevices, "memory_mib"), range));
  const failure = (q) => q.error || q.data?.error || "";
  const gpuError = $derived(failure(queries.utilization));
  const gpuMemoryError = $derived(failure(queries.memory_mib));
  const visible = $derived(all.filter((p) => p.node === node?.name && (view.gpu === "all" || p.devices.includes(view.gpu)) && p.start <= range.now && p.end >= range.start));
  const errors = $derived([...new Set([...inventory.errors, history.error].filter(Boolean))]);
  // Only accelerator nodes are tracked; busy ones first.
  const busy = (n) => Number(inventory.placements.some((p) => p.node === n.name));
  const nodes = $derived(inventory.cluster.nodes.filter((n) => n.gpu_capacity > 0 || n.devices?.length).sort((a, b) => busy(b) - busy(a)));
  const unavailable = $derived(!node && (view.node || (view.placement && history.data)));
  const duration = $derived(range.now - range.start);
  const back = $derived(viewLink("nodes", { duration, end: range.now, node: view.node, placement: view.placement, gpu: view.node && view.gpu !== "all" ? view.gpu : null }));
  const cursor = $derived(crosshairOffset(range));
  $effect(() => {
    if (view.duration !== duration) view.duration = duration;
    if (view.end !== null && view.end !== range.now) view.end = range.now;
    if (view.placement && !all.some((p) => p.id === view.placement)) {
      const alias = all.find((p) => p.allocation_id === view.placement);
      if (alias) view.placement = alias.id;
    }
    const selected = all.find((p) => p.id === view.placement);
    if (!view.node && selected) view.node = selected.node;
    if (node && selected && selected.node !== node.name) view.placement = null;
    if (node && (node.devices.length === 1 || (view.gpu !== "all" && !node.devices.some((d) => d.id === view.gpu)))) view.gpu = "all";
  });
  let scrollSelection = $state(true);
  $effect(() => {
    if (!view.node && !view.placement) scrollSelection = false;
    if (node && !loading && scrollSelection) { scrollSelection = false; tick().then(() => element?.querySelector("#placement-detail")?.scrollIntoView({ block: "start" })); }
  });
  function keepViewport() { element?.closest("#app")?.style.setProperty("min-height", `${window.scrollY + innerHeight}px`); }
  function select(p, reset = false) {
    if (view.node !== p.node) element?.closest("#app")?.style.removeProperty("min-height");
    if (reset) view.gpu = "all";
    view.node = p.node;
    view.placement = p.id;
  }
  async function close() {
    const previous = view.node;
    view.node = view.placement = null;
    element?.closest("#app")?.style.removeProperty("min-height");
    await tick();
    element.querySelector(`[data-node-details="${CSS.escape(previous || "")}"]`)?.focus({ preventScroll: true });
  }
  function openNode(name) {
    if (view.node === name) return close();
    element?.closest("#app")?.style.removeProperty("min-height");
    view.node = name; view.placement = null; view.gpu = "all";
  }
  function escapeKey(event) {
    if (event.key !== "Escape" || event.defaultPrevented) return;
    if (view.node || view.placement) close();
  }
  onDestroy(() => element?.closest("#app")?.style.removeProperty("min-height"));
</script>
<svelte:window onhashchange={() => scrollSelection = true} onkeydown={escapeKey} />
<div bind:this={element} use:nodeTime={{ view, getRange: () => range, getNow: () => observed, onGesture: (value) => navigating = !!value }} data-crosshair={cursor !== null ? "" : undefined} style:--crosshair={`${cursor ?? 0}%`}>
  <div class="nodes-heading"><h1 class="heading">Accelerator nodes</h1>
    <div class="time-control" aria-label="Node time range">
      <TimeControl {view} {range} {observed} /><CopyLink />
    </div>
  </div>
  {#each errors as error}<p class="source-error" role="status">{error}</p>{/each}
  {#if history.error}<button type="button" class="chip" data-retry-data="activity:nodes" onclick={history.retry}>Retry activity</button>{/if}
  {#if !inventory.cluster.available && !errors.length}<p class="empty-state">Kubernetes unavailable</p>{/if}
  {#if unavailable}<div class="unavailable-selection" role="status"><span>{view.node ? "The selected node is no longer reported." : "The selected placement is not in retained history."}</span><button type="button" class="detail-close" data-close-details aria-label="Clear unavailable selection" title="Clear selection" onclick={close}>×</button></div>{/if}
  <div class="node-time-header"><span>Node</span><div class="node-axis">{#each [0, 1, 2, 3] as step}<span>{axisTime(range.start + (range.now - range.start) * step / 3, range)}</span>{/each}</div><span class="node-duty" title="Duty cycle: mean GPU utilization (share of time a kernel was running) over this window, from DCGM">Duty</span></div>
  {#each loading ? [] : nodes as item (item.name)}
    {@const devices = item.devices || []}
    {@const height = devices.length === 1 ? 30 : 22}
    {@const placements = all.filter((p) => p.node === item.name && p.start <= range.now && p.end >= range.start)}
    {@const current = range.live ? inventory.placements.filter((p) => p.node === item.name) : placements.filter((p) => p.start <= range.now && p.end >= range.now)}
    {@const unmapped = placements.filter((p) => !p.devices.length || p.devices.some((id) => !devices.some((d) => d.id === id)))}
    <div class="node-placement-group" data-key={item.name}>
      <div class="node-lane"><button type="button" class="node-lane-label" data-node-details={item.name} aria-expanded={view.node === item.name} title={item.name} onclick={() => openNode(item.name)}><span class="node-accelerator">{acceleratorLabel(item)} <span class="muted">· {shortNodeName(item.name, inventory.cluster.nodes)}</span></span>{#if item.gpu_capacity}<span class="claim-label">{claimLabel(item, current, range.live)}</span>{/if}</button>
        <div>{#if devices.length}<div class="gpu-capacity"><div class="gpu-lane-ids" style:grid-auto-rows={`${height}px`}>{#each devices as device (device.id)}<span title={device.id}>{deviceLabel(device.name)}</span>{/each}</div><!-- svelte-ignore a11y_no_static_element_interactions (Pointer location follows the shared time axis.) --><div class="capacity-track" style:height={`${devices.length * height}px`} style:--gpu-lane-height={`${height}px`} onpointermove={(event) => crosshair.at = event.buttons ? null : timeAt(event, range)} onpointerleave={() => crosshair.at = null}>
          {#each capacityBars(devices, placements, range, view.placement) as bar (bar.key)}
            <button type="button" class="capacity-allocation held" class:selected={bar.selected} data-key={bar.key} data-placement={bar.placement.id} aria-expanded={bar.selected} aria-label={bar.title} title={bar.title} style={bar.style} onclick={(event) => select(placements.find((p) => p.id === event.target.closest("[data-placement]")?.dataset.placement) || bar.placement, true)}>
              {#each bar.holds as hold}<span class={`hold ${placementColor(hold.placement)}`} class:selected={hold.placement.id === view.placement} data-placement={hold.placement.id} style={hold.style} title={`${hold.placement.label} · ${hold.placement.role || ""}`}></span>{/each}
            </button>
          {/each}
        </div></div>{/if}
        {#each unmapped as p (p.id)}<button type="button" class={`capacity-allocation unknown-mapping ${placementColor(p)}`} data-key={`unmapped:${p.id}`} data-placement={p.id} aria-expanded={p.id === view.placement} onclick={() => select(p, true)}><span class="allocation-name">{p.label}</span><span class="allocation-count">{p.device_count} GPU{p.device_count === 1 ? "" : "s"}</span></button>{/each}
        {#if unmapped.length}<span class="muted micro">Device mapping unavailable</span>{:else if !devices.length && item.gpu_capacity}<p class="empty-state">GPUs without DRA devices</p>{/if}</div>
        <span class="node-duty">{dutyCycle(item)}</span>
      </div>
      {#if item.name === view.node}
        {@const label = `${acceleratorLabel(item)} · ${shortNodeName(item.name, inventory.cluster.nodes)}`}
        <section class="allocation-expansion" id="placement-detail"><div class="allocation-detail-head inspector-heading"><h2 title={item.name}>{label}</h2><button type="button" class="detail-close" data-close-details aria-label="Close node details" title="Close (Esc)" onclick={close}>×</button></div>
          {#if devices.length > 1}<div class="allocation-device-picker" aria-label="GPU selection">
            <button type="button" class="chip" data-device="all" aria-pressed={view.gpu === "all"} onclick={() => { keepViewport(); view.gpu = "all"; }}>All GPUs</button>
            <!-- GPU labels wait for the first metrics response, so the centered numbers don't slide when utilization arrives. -->
            {#each devices as device (device.id)}{@const value = lastValue(deviceSeries(metrics.devices.filter((d) => d.id === device.id), "utilization"), range)}<button type="button" class="chip" data-device={device.id} title={`GPU ${deviceLabel(device.name)}`} aria-pressed={view.gpu === device.id} onclick={() => { keepViewport(); view.gpu = device.id; }}>{#if loaded}{deviceLabel(device.name)}{Number.isFinite(value) ? ` · ${Math.round(value)}%` : ""}{/if}</button>{/each}
          </div>{/if}
          <div class="node-activity" data-key="node-timeline"><Activity placements={visible} {range} {inventory} selected={view.placement} onselect={select} {back} activeOnly showSource={false} {navigating} /></div>
          {#if item.gpu_capacity}<div class="node-gpu-detail" data-key="node-gpu-detail"><div><Chart title={`Duty cycle · ${label}`} unit="%" points={gpuPoints} start={range.start} end={range.now} min={0} max={100} readingSeconds={DCGM_READING_SECONDS} bands={dutyBands(visible)} gapSeconds={gapFor(gpuPoints, 60)} empty={gpuError || (!loaded && "Loading…") || "No samples for this GPU"} note={gpuError ? `${gpuError} · Showing previously fetched samples` : ""} {navigating} />
            {#if queries.utilization.error}<button type="button" class="chip" onclick={queries.utilization.retry}>Retry GPU metrics</button>{/if}
          </div><div class="allocation-detail-footer"><span>{activitySource(visible)}</span><span class="allocation-memory">GPU memory <strong>{Number.isFinite(gpuMemory) ? `${(gpuMemory / 1024).toFixed(1)} GiB` : "—"}</strong>{#if gpuMemoryError}<span class="source-error" role="status"> · {Number.isFinite(gpuMemory) ? "Stale" : "Unavailable"}: {gpuMemoryError}</span>{/if}</span></div></div>{/if}
        </section>
      {/if}
    </div>
  {:else}{#if loading}<p class="empty-state">Loading activity…</p>{:else if !errors.length && inventory.cluster.available}<p class="empty-state">No accelerator nodes reported by Kubernetes</p>{/if}{/each}
</div>
