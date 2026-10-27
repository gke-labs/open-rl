<script module>
  let charts = 0;
</script>
<script>
  import { axisTime, chartNumber, chartTime, valueText } from "./visualization.js";
  import { crosshair, crosshairOffset } from "./state.svelte.js";
  let { title, unit = "", points = [], start, end, min, max, tone = "neutral", gapSeconds = Infinity, empty = "No samples in this time range", note = "", navigating = false, steps = false, readingSeconds = 0, bands = [] } = $props();
  // Each band is a process and the spans it held the GPU; the area under the line takes its color there.
  // With readingSeconds, each point is a reading the source took that often, so it holds until the next one.
  // With steps, x is a training step rather than a time, so the chart stays out of the shared time crosshair.
  let plot = $state(null), svg = $state(null), tip = $state(null);
  let selectedAt = $state(null);
  const uid = `chart-${charts++}`;
  // A chart with bands opens on whether anything ran on the GPU: 100 while a process held it or ran an op, else 0, from the
  // records rather than DCGM's 10 s averages. Readings shows DCGM's own, and Duty (1 min) their rolling mean.
  const ROLLING_SECONDS = 60;
  let chosen = $state(null);
  const mode = $derived(chosen ?? (bands.length ? "running" : "readings"));
  const running = $derived.by(() => {
    const merged = [];
    for (const [a, b] of bands.flatMap((band) => band.spans).map(([a, b]) => [Math.max(start, a), Math.min(end, b)]).filter(([a, b]) => b > a).toSorted((p, q) => p[0] - q[0])) {
      if (merged.length && a <= merged.at(-1)[1]) merged.at(-1)[1] = Math.max(merged.at(-1)[1], b);
      else merged.push([a, b]);
    }
    return merged;
  });
  const shown = $derived.by(() => {
    if (mode === "running") return [[start, 0], ...running.flatMap(([a, b]) => [[a, 0], [a, 100], [b, 100], [b, 0]]), [end, 0]];
    if (mode !== "rolling") return points;
    const sorted = points.filter(([at, value]) => Number.isFinite(at) && Number.isFinite(value)).toSorted((a, b) => a[0] - b[0]);
    let first = 0, sum = 0;
    return sorted.map(([at, value], i) => {
      sum += value;
      while (sorted[first][0] <= at - ROLLING_SECONDS) sum -= sorted[first++][1];
      return [at, sum / (i - first + 1)];
    });
  });
  const samples = $derived(shown.filter(([at]) => Number.isFinite(at) && at >= start && at <= end).sort((a, b) => a[0] - b[0]));
  const data = $derived(samples.filter(([, value]) => Number.isFinite(value)));
  const values = $derived(data.map(([, value]) => value));
  const scale = $derived.by(() => {
    let low = Number.isFinite(min) ? min : Math.min(...values), high = Number.isFinite(max) ? max : Math.max(...values);
    if (!values.length) return { low: 0, high: 1, ticks: [] };
    if (high <= low) { const pad = Math.max(Math.abs(low) * 0.1, 0.1); if (!Number.isFinite(min)) low -= pad; high += pad; }
    let ticks = [high, (high + low) / 2, low];
    if (!Number.isFinite(min) && !Number.isFinite(max)) {
      const raw = (high - low) / 3, magnitude = 10 ** Math.floor(Math.log10(raw));
      const step = [1, 2, 5, 10].find((n) => n >= raw / magnitude) * magnitude;
      low = Math.floor(low / step) * step; high = Math.ceil(high / step) * step;
      ticks = Array.from({ length: Math.round((high - low) / step) + 1 }, (_, i) => Number((high - i * step).toPrecision(12)));
    }
    return { low, high, ticks };
  });
  const segments = $derived.by(() => {
    const result = [];
    let current;
    for (const sample of samples) {
      if (!Number.isFinite(sample[1])) { current = null; continue; }
      if (!current || (mode !== "running" && sample[0] - current.at(-1)[0] > gapSeconds)) result.push(current = []);
      current.push(sample);
    }
    return result;
  });
  const spans = $derived(segments.map((segment) => [segment[0][0], segment.at(-1)[0]]));
  const tickCount = $derived(end === start ? 0 : 4);
  const xAt = (at) => (Math.max(start, Math.min(end, at)) - start) / (end - start || 1) * 1000;
  const x = (at) => ((at - start) / (end - start || 1) * 1000).toFixed(1);
  const y = (value) => (1000 - (Math.max(scale.low, Math.min(scale.high, value)) - scale.low) / (scale.high - scale.low) * 1000).toFixed(1);
  const paths = $derived(segments.map((segment) => {
    const line = segment.map(([at, value], i) => !i ? `M${x(at)},${y(value)}` : readingSeconds && mode === "readings" ? `H${x(at)} V${y(value)}` : `L${x(at)},${y(value)}`).join(" ");
    return { line: segment.length === 1 ? `${line} h0.01` : line, area: segment.length > 1 ? `${line} L${x(segment.at(-1)[0])},1000 L${x(segment[0][0])},1000 Z` : null };
  }));
  const label = (at) => steps ? chartNumber(Math.round(at)) : axisTime(at, { start, now: end });
  const crosshairX = $derived(steps ? null : crosshairOffset({ start, now: end }));
  function indexAt(at) {
    let left = 0, right = data.length - 1;
    while (left < right) { const middle = Math.floor((left + right) / 2); if (data[middle][0] < at) left = middle + 1; else right = middle; }
    return left > 0 && Math.abs(data[left - 1][0] - at) < Math.abs(data[left][0] - at) ? left - 1 : left;
  }
  const hover = $derived.by(() => {
    if (selectedAt === null || navigating || !svg) return null;
    const index = indexAt(selectedAt);
    if (data[index]?.[0] !== selectedAt) return null;
    const [at, value] = data[index], box = svg.getBoundingClientRect();
    const left = (at - start) / (end - start || 1) * box.width;
    return { index, left: box.left - plot.getBoundingClientRect().left + left, top: Math.max(0, Math.min(100, (scale.high - value) / (scale.high - scale.low) * 100)), tipLeft: Math.max(-left, Math.min(8, box.width - left - (tip?.offsetWidth || 160))), text: `${steps ? `Step ${at}` : chartTime(at, true)} · ${valueText(value, unit)}` };
  });
  const inspect = (index) => { selectedAt = data[Math.max(0, Math.min(data.length - 1, index))]?.[0] ?? null; };
  function pointer(event) {
    const box = svg.getBoundingClientRect(), at = start + (event.clientX - box.left) / (box.width || 1) * (end - start);
    const outside = event.clientX < box.left || event.clientX > box.right;
    crosshair.at = steps || navigating || event.buttons || outside ? null : at;
    if (!data.length || navigating || event.buttons) { selectedAt = null; return; }
    const index = indexAt(at);
    const near = Math.abs(data[index][0] - at) / (end - start || 1) * box.width < 6;
    if (outside || (!near && !spans.some(([a, b]) => at >= a && at <= b))) selectedAt = null;
    else inspect(index);
  }
  function keyboard(event) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.key === "Escape") { selectedAt = null; plot.blur(); }
    else inspect(event.key === "Home" ? 0 : event.key === "End" ? data.length - 1 : (hover?.index ?? data.length - 1) + (event.key === "ArrowLeft" ? -1 : 1));
  }
</script>
<svelte:window onresize={() => selectedAt = null} />
<figure class="chart" data-key={title} data-tone={tone}>
  <figcaption class="chart-head"><h2>{title}</h2>{#if readingSeconds}<span class="chart-mode" role="group" aria-label="What to plot">{#if bands.length}<button type="button" class="chip" aria-pressed={mode === "running"} onclick={() => chosen = "running"}>Running</button>{/if}<button type="button" class="chip" aria-pressed={mode === "readings"} onclick={() => chosen = "readings"}>Readings</button><button type="button" class="chip" aria-pressed={mode === "rolling"} onclick={() => chosen = "rolling"}>Duty (1 min)</button></span>{/if}{#if data.length && mode !== "running"}<span class="chart-latest" title={steps ? `Step ${data.at(-1)[0]}` : new Date(data.at(-1)[0] * 1000).toISOString()}><span>Latest</span> <strong>{valueText(data.at(-1)[1], unit)}</strong></span>{/if}</figcaption>
  {#if !data.length}<p class="chart-empty">{empty}</p>
  {:else}
    <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (Arrow keys inspect the chart samples.) -->
    <div class="chart-plot" bind:this={plot} tabindex="0" role="group" aria-label={`${title}. Use left and right arrow keys to inspect samples.`} onpointermove={pointer} onpointerleave={() => selectedAt = crosshair.at = null} onfocusout={() => selectedAt = null} onkeydown={keyboard}>
      <div class="chart-y">{#each scale.ticks as tick}<span>{chartNumber(tick)}</span>{/each}</div>
      <svg bind:this={svg} viewBox="0 0 1000 1000" preserveAspectRatio="none" role="img" aria-label={`${title}. Latest ${valueText(data.at(-1)[1], unit)}.`}>
        {#each scale.ticks as tick}<line class="chart-grid" x1="0" x2="1000" y1={y(tick)} y2={y(tick)} vector-effect="non-scaling-stroke" />{/each}
        {#each paths as path}{#if path.area}<path class="chart-area" d={path.area} />{/if}{/each}
        {#each bands as band, i (band.key)}<clipPath id={`${uid}-${i}`}>{#each band.spans as [a, b]}<rect x={xAt(a)} width={Math.max(0, xAt(b) - xAt(a))} y="0" height="1000" />{/each}</clipPath>
          {#each paths as path}{#if path.area}<path class={`chart-band ${band.hue}`} class:sampler={band.role === "sampler"} d={path.area} clip-path={`url(#${uid}-${i})`} />{/if}{/each}{/each}
        {#each paths as path}<path class={path.area ? "chart-line" : "chart-dot"} d={path.line} vector-effect="non-scaling-stroke" />{/each}
        {#if !hover && crosshairX !== null}<line class="chart-crosshair" x1={crosshairX * 10} x2={crosshairX * 10} y1="0" y2="1000" vector-effect="non-scaling-stroke" />{/if}
      </svg>
      <div class="chart-x">{#each Array.from({ length: tickCount + 1 }, (_, i) => start + (end - start) * i / (tickCount || 1)) as at}<span>{label(at)}</span>{/each}</div>
      <div class="chart-hover" hidden={!hover} style:left={`${hover?.left || 0}px`}><div class="chart-cursor"></div><div class="chart-point" style:top={`${hover?.top || 0}%`}></div><div class="chart-tip" bind:this={tip} style:left={`${hover?.tipLeft || 0}px`}>{hover?.text || ""}</div></div>
    </div>
    <div class="chart-summary">{#if mode === "running"}<span>Something ran <strong>{valueText(100 * running.reduce((sum, [a, b]) => sum + b - a, 0) / (end - start || 1), "%")}</strong> of the range</span>{:else}<span title="Arithmetic mean of the reported samples">Average <strong>{valueText(values.reduce((sum, value) => sum + value, 0) / values.length, unit)}</strong></span><span>Peak <strong>{valueText(Math.max(...values), unit)}</strong></span>{#if readingSeconds}<span>{mode === "rolling" ? `Mean of the last minute's ${readingSeconds} s readings` : `One reading per ${readingSeconds} s`}</span>{/if}{/if}{#each bands as band (band.key)}<span class={`chart-band-key ${band.hue}`} class:sampler={band.role === "sampler"}><span class="chart-band-swatch"></span>{band.label}</span>{/each}{#if note}<span class="chart-note" role="status">{note}</span>{/if}</div>
  {/if}
</figure>
