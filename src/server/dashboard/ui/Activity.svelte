<script>
  import { nodeLink, viewLink, crosshair, crosshairOffset, timeAt } from "./state.svelte.js";
  import GpuDuty from "./GpuDuty.svelte";
  import { axisTime, chartTime, valueText, placementColor, processLabel, shortNodeName, opRank, opLabel, activitySource, deviceLabel } from "./visualization.js";
  let { placements = [], range, inventory, acrossNodes = false, activeOnly = false, selected = null, onselect, error = null, pending = false, showSource = true, back = null, navigating = false, onpick = null, duty = null, slots = null } = $props();
  let tooltip = $state(null), tip, open = $state({}), mode = $state("lanes");
  const cursor = $derived(crosshairOffset(range));
  const activities = $derived(placements.toSorted((a, b) => (acrossNodes ? (({ trainer: 0, sampler: 1 }[a.role] ?? 2) - ({ trainer: 0, sampler: 1 }[b.role] ?? 2) || String(a.node || "").localeCompare(String(b.node || ""))) : 0) || a.start - b.start || a.id.localeCompare(b.id)));
  const shown = $derived(activeOnly ? activities.filter((p) => p.intervals.length || p.error) : activities);
  // The run view draws one row per GPU; a multi-GPU process runs each op on all its GPUs, so its rows repeat the lane.
  const rows = $derived(shown.flatMap((p) => (acrossNodes && p.devices?.length ? p.devices : [null]).map((id, i) => ({ key: `${p.id}:${id ?? "none"}`, lane: p, device: id && gpuOf(p, id), first: i === 0, lead: i === 0 && p.devices.length > 1 }))));
  const warnings = $derived([...new Set(activities.map((p) => p.errorStatus !== 404 && p.error).filter(Boolean))]);
  const ops = $derived([...new Set(shown.flatMap((p) => Object.keys(p.ops || {})))].sort((a, b) => opRank(a) - opRank(b) || a.localeCompare(b)));
  const empty = $derived(error ? `Activity unavailable: ${error}` : pending ? "Loading activity…" : acrossNodes && !activities.length ? "No worker processes for this run in this window" : "No recorded activity in this window");
  const href = (p) => acrossNodes ? inventory.cluster.nodes.some((n) => n.name === p.node) ? nodeLink(p.id, range) : null : inventory.runs.some((r) => r.run_id === p.run_ids?.[0]) ? viewLink(`run/${encodeURIComponent(p.run_ids[0])}/activity`, { duration: range.now - range.start, end: range.now, back }) : null;
  const blocks = (spans) => spans.map(([a, b]) => `M${1000 * (a - range.start) / (range.now - range.start)},0 H${1000 * (b - range.start) / (range.now - range.start)} V24 H${1000 * (a - range.start) / (range.now - range.start)} Z`).join(" ");
  const rowLabel = (p) => acrossNodes ? `${p.label} · ${processLabel(p, inventory.cluster.nodes)} · ${p.node}` : p.label;
  // A lane names its GPUs by UUID; a node no longer reported leaves DCGM's slot as the name.
  function gpuOf(p, id) {
    const device = inventory.cluster.nodes.find((n) => n.name === p.node)?.devices.find((d) => d.id === id);
    return { id, label: device ? `GPU ${deviceLabel(device.name)}` : slots?.[id] != null ? `GPU ${slots[id]}` : id.slice(0, 12) };
  }
  const dutyText = (device) => { const value = device && duty?.[device.id]; return Number.isFinite(value) ? `${value}%` : "—"; };
  $effect(() => { placements; range; navigating; tooltip = null; });
  function trackMove(event) {
    if (navigating || event.buttons) { tooltip = crosshair.at = null; return; }
    crosshair.at = timeAt(event, range);
    if (!event.target.closest(".activity-block")) tooltip = null;
  }
  function inspect(event, spans, label, source) {
    if (navigating || event.buttons) return;
    const box = event.currentTarget.ownerSVGElement.getBoundingClientRect();
    const at = range.start + (event.clientX - box.left) / box.width * (range.now - range.start);
    const interval = spans.find(([a, b]) => at >= a && at <= b);
    if (!interval) { tooltip = null; return; }
    const [a, b] = interval, date = (value) => new Date(value * 1000).toISOString().slice(0, 10);
    tooltip = { label, time: `${date(a)} · ${chartTime(a, true)} – ${date(a) === date(b) ? "" : date(b) + " · "}${chartTime(b, true)}`, detail: `Visible interval · ${valueText(b - a, "s")}${source ? ` · ${source}` : ""}${onpick ? " · Click for logs" : ""}`, left: Math.max(8, Math.min(event.clientX + 12, innerWidth - (tip?.offsetWidth || 300) - 8)), top: Math.max(8, Math.min(event.clientY + 12, innerHeight - (tip?.offsetHeight || 85) - 8)) };
  }
  function select(event, placement) {
    if (onselect && (!event.key || ["Enter", " "].includes(event.key))) { event.preventDefault(); onselect(placement); }
  }
</script>
<svelte:window onscroll={() => tooltip = crosshair.at = null} onresize={() => tooltip = crosshair.at = null} />
<!-- svelte-ignore a11y_no_static_element_interactions (The timeline delegates pointer inspection to its tracks.) -->
<section class:cross-node-activity={acrossNodes} class:run-activity={acrossNodes} aria-label={acrossNodes ? "Run process activity" : "Node process activity"} data-crosshair={cursor !== null ? "" : undefined} style:--crosshair={`${cursor ?? 0}%`} onpointerleave={() => tooltip = crosshair.at = null} onpointerdown={() => tooltip = crosshair.at = null}>
  {#if acrossNodes}<h2 class="scheduler-title activity-title">GPU activity</h2><p class="muted activity-intro">One row per accelerator. Colored bars are ops the process on that GPU ran; grey is GPU time held between ops. Duty is the share of time a kernel ran on the GPU, measured by DCGM.</p>
    <div class="activity-mode" role="group" aria-label="Lanes or duty cycle"><button type="button" class="chip" aria-pressed={mode === "lanes"} onclick={() => mode = "lanes"}>Lanes</button><button type="button" class="chip" aria-pressed={mode === "duty"} onclick={() => mode = "duty"}>Duty cycle</button></div>{/if}
  {#if !shown.length}<p class="empty-state">{empty}</p>
  {:else}
    <div class="activity-timeline" aria-label="Recorded run activity">
      <div class="activity-header"><h3>{acrossNodes ? "Process" : "Run"}</h3>{#if acrossNodes}<h3>GPU</h3>{/if}<div class="activity-axis">{#each [0, 1, 2, 3, 4] as tick}<span>{axisTime(range.start + (range.now - range.start) * tick / 4, range)}</span>{/each}</div>{#if acrossNodes}<span class="node-duty" title="Duty cycle: mean GPU utilization (share of time a kernel was running) over this window, from DCGM">Duty</span>{/if}</div>
      {#each rows as { key, lane: p, device, first, lead } (key)}
        {@const name = acrossNodes ? processLabel(p, inventory.cluster.nodes, { gpus: false }) : p.label}
        {@const meta = acrossNodes ? p.node ? shortNodeName(p.node, inventory.cluster.nodes) : p.ended && p.pod ? p.pod : "Node unassigned" : p.role || "process"}
        {@const link = href(p)}
        {@const status = p.error ? "Ops unavailable" : !p.intervals.length ? "No recorded activity" : ""}
        <div class={`activity-row ${placementColor(p)}`} data-key={p.id} data-selected={!acrossNodes && p.id === selected} data-continued={first ? undefined : ""} data-lead={lead ? "" : undefined}>
          {#if first}<svelte:element this={link ? "a" : "span"} class={`activity-label ${placementColor(p)}`} data-key={`label:${p.id}`} href={link || undefined} aria-disabled={!link ? "true" : undefined} title={acrossNodes ? `${name} · ${p.node || "Node unassigned"}${p.node && !link ? " · Node no longer reported" : ""}` : p.label}>
            <span class="activity-swatch"></span><span class="activity-label-text"><span class="activity-name">{name}</span><span class="activity-meta">{meta}{p.ended ? " · Process ended" : p.phase && p.phase !== "ready" ? ` · ${p.phase.replace("_", " ")}` : ""}{!link ? acrossNodes ? p.node ? " · Node no longer reported" : "" : p.run_ids?.length ? " · Run record unavailable" : " · No linked run" : ""}</span></span>
          </svelte:element>{:else}<span></span>{/if}
          {#if acrossNodes}{#if device && p.node}<button type="button" class="activity-gpu activity-gpu-toggle" aria-expanded={!!open[key]} title={`${device.id} · ${open[key] ? "Hide" : "Plot"} this GPU's duty cycle`} onclick={() => open[key] = !open[key]}><span class="activity-chevron" aria-hidden="true">{open[key] ? "▾" : "▸"}</span>{device.label}</button>{:else}<span class="activity-gpu" title={device ? device.id : "This process recorded no GPUs"}>{device?.label ?? "GPUs not recorded"}</span>{/if}{/if}
          <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events (Pointer location follows the shared time axis. Incident links and the Logs tab reach the same logs by keyboard.) -->
          <div class="activity-track" onpointermove={trackMove} onclick={onpick && ((event) => onpick(Math.round(timeAt(event, range))))}>
            {#if mode === "lanes"}<svg viewBox="0 0 1000 24" preserveAspectRatio="none" role="img" aria-label="Recorded activity">
              {#if p.waits?.length}<path class="activity-block wait" d={blocks(p.waits)} vector-effect="non-scaling-stroke" onpointermove={(event) => inspect(event, p.waits, `Waiting for the GPU · ${rowLabel(p)}`, "From asking the time slicer to the grant")} />{/if}
              <!-- svelte-ignore a11y_no_static_element_interactions, a11y_no_noninteractive_tabindex (Keyboard activation selects this placement.) -->
              <path class={`activity-block ${placementColor(p)}`} role={onselect ? "button" : undefined} tabindex={onselect ? 0 : undefined} aria-label={rowLabel(p)} data-placement={!acrossNodes ? p.id : undefined} data-selected={!acrossNodes && p.id === selected} d={blocks(p.intervals)} vector-effect="non-scaling-stroke" onclick={(event) => select(event, p)} onkeydown={(event) => select(event, p)} onpointermove={(event) => inspect(event, p.intervals, rowLabel(p), p.exact ? "GPU turns" : "Recorded operations")} />
              {#each Object.entries(p.ops).sort(([a], [b]) => opRank(a) - opRank(b)) as [op, spans] (op)}
                <!-- svelte-ignore a11y_no_static_element_interactions, a11y_no_noninteractive_tabindex (Keyboard activation selects this placement.) -->
                <path class="activity-block op" role={onselect ? "button" : undefined} tabindex={onselect ? 0 : undefined} aria-label={`${opLabel(op)} · ${rowLabel(p)}`} data-placement={!acrossNodes ? p.id : undefined} data-op={op} d={blocks(spans)} vector-effect="non-scaling-stroke" onclick={(event) => select(event, p)} onkeydown={(event) => select(event, p)} onpointermove={(event) => inspect(event, spans, `${opLabel(op)} · ${rowLabel(p)}`, "Worker-recorded op")} />
              {/each}
            </svg>{/if}
            {#if status && first}<span class="activity-status" class:unavailable={!!p.error} title={p.error || status}>{status}</span>{/if}
          </div>
          {#if acrossNodes}<span class="node-duty">{dutyText(device)}</span>{/if}
        </div>
        {#if (open[key] || mode === "duty") && device && p.node}<div class="activity-duty"><GpuDuty node={p.node} {device} {range} /></div>{/if}
      {/each}
      {#if ops.length && mode === "lanes"}<div class="op-legend" aria-label="GPU ops">{#each ops as op}<span data-op={op}><span class="op-swatch"></span>{opLabel(op)}</span>{/each}<span><span class="op-swatch turn"></span>held, unlabelled</span>{#if shown.some((p) => p.waits?.length)}<span><span class="op-swatch wait"></span>waiting for the GPU</span>{/if}</div>{/if}
    </div>
  {/if}
  {#if error && shown.length}<p class="source-error" role="status">Activity unavailable: {error}</p>{/if}
  {#if warnings.length}<p class="source-error" role="status">{warnings.join(" · ")}</p>{/if}
  {#if showSource && activities.length}<p class="activity-source">{activitySource(activities)}</p>{/if}
  <div class="activity-tooltip" bind:this={tip} role="tooltip" hidden={!tooltip} style:left={`${tooltip?.left || 0}px`} style:top={`${tooltip?.top || 0}px`}><strong class="activity-tooltip-label">{tooltip?.label || ""}</strong><div class="activity-tooltip-time">{tooltip?.time || ""}</div><div class="activity-tooltip-detail">{tooltip?.detail || ""}</div></div>
</section>
