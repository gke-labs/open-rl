<script>
  import { WINDOWS } from "./node-time.js";
  import { MAX_WINDOW } from "./state.svelte.js";
  import { chartTime } from "./visualization.js";
  // The time window shared by the Nodes and run pages: back and forward, a window and
  // end picker, and the W/S A/D keys that node-time.js handles on the page.
  let { view, range, observed } = $props();
  let picker;
  const duration = $derived(range.now - range.start);
  const custom = $derived([[Math.floor(duration / 3600), "h"], [Math.floor(duration / 60) % 60, "m"], [duration % 60, "s"]].filter(([n]) => n).map(([n, unit]) => `${n}${unit}`).join(" "));
  const options = $derived(WINDOWS.some(([n]) => n === duration) ? WINDOWS : [[duration, `${custom} (custom)`], ...WINDOWS]);
  const local = (at) => new Date(at * 1000).toISOString().slice(0, duration <= 120 ? 19 : 16);
  const timeLabel = $derived(range.live ? `Last ${WINDOWS.find(([n]) => n === duration)?.[1] || custom}` : `${local(range.start).slice(5, 10)} · ${chartTime(range.start, duration <= 120)} – ${local(range.start).slice(0, 10) === local(range.now).slice(0, 10) ? "" : `${local(range.now).slice(5, 10)} · `}${chartTime(range.now, duration <= 120)}`);
  function shift(sign) { const end = range.now + sign * duration; view.end = end >= observed ? null : end; }
  function until(event) {
    if (!event.currentTarget.validity.valid) { event.currentTarget.reportValidity(); return; }
    const end = event.currentTarget.value ? Date.parse(`${event.currentTarget.value}Z`) / 1000 : null;
    view.end = Number.isFinite(end) ? Math.min(end, observed) : null;
  }
  function escapeKey(event) {
    if (event.key === "Escape" && picker?.open) { picker.open = false; picker.querySelector("summary").focus(); event.preventDefault(); }
  }
</script>
<svelte:window onkeydown={escapeKey} onclick={(event) => { if (picker?.open && !picker.contains(event.target)) picker.open = false; }} />
<button type="button" class="time-shift" data-time-shift="-1" aria-label="Previous time window" disabled={range.start <= observed - MAX_WINDOW} onclick={() => shift(-1)}><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(90 8 8)" /></svg></button>
<details class="time-picker" data-key="node-time-picker" bind:this={picker}><summary class="time-summary" title={timeLabel}><span>{timeLabel}</span><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" /></svg></summary>
  <div class="time-popover"><label>Window <select data-time-duration value={duration} onchange={(event) => { view.duration = Number(event.currentTarget.value); if (view.end !== null) view.end = range.now; }}>{#each options as [n, text] (n)}<option value={n}>{text}</option>{/each}</select></label>
    <label>Until (UTC)<input type="datetime-local" data-time-end value={range.live ? "" : local(range.now)} min={local(observed - MAX_WINDOW + duration)} max={local(observed)} step="any" onchange={until}></label>
    {#if range.live}<span class="muted">Following current time</span>{:else}<button type="button" class="chip" data-time-live="true" onclick={() => view.end = null}>Return to live</button>{/if}
    <span class="muted">W/S zoom · A/D pan</span>
  </div>
</details>
<button type="button" class="time-shift" data-time-shift="1" aria-label="Next time window" disabled={range.now >= observed} onclick={() => shift(1)}><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(-90 8 8)" /></svg></button>
