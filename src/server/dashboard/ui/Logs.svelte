<script>
  import { untrack, tick } from "svelte";
  import { get, inspectUrl } from "./state.svelte.js";
  import { escapeRegExp } from "./format.js";
  let { run, view, range, allLogs, navigating = false } = $props();
  let data = $state.raw(null), pending = $state(true), error = $state(null);
  let controller, filters, failedCursor, lines, fromBottom = 0;
  const records = $derived(data?.records || []), cursor = $derived(records.length < 2000 ? data?.next_cursor : null);
  // A shared runtime's pods also log for other runs, so by default the run sees only the lines that name it.
  let everything = $state(false);
  const scope = $derived(view.q || (run.shared_runtime && !everything ? run.run_id : ""));
  const logRange = $derived({
    since: new Date(Math.max(range.start, view.event === null ? range.start : view.event - 120) * 1000).toISOString(),
    until: new Date(Math.min(range.now, view.event === null ? range.now : view.event + 120) * 1000).toISOString(),
  });
  const params = $derived({ run_id: view.id, q: scope, limit: "200", ...logRange, ...(view.pod ? { pod: view.pod } : {}) });
  // The user's selection, not the live window, which moves with every snapshot and is read at each load.
  const filterKey = $derived(JSON.stringify([view.id, scope, view.pod, view.event, view.duration]));
  const queryKey = $derived(`${filterKey}:${view.end}`);
  const sources = $derived.by(() => {
    const pods = new Map((run.pods || []).map((p) => [p.name, p]));
    for (const record of [...(data?.sources || []), ...records]) if (record.pod && !pods.has(record.pod)) pods.set(record.pod, { ...record, name: record.pod });
    if (view.pod && !pods.has(view.pod)) pods.set(view.pod, { name: view.pod });
    return [...pods.values()].sort((a, b) => a.name.localeCompare(b.name));
  });
  // Dozzle-style line tags: the part of the pod name that differs between the run's pods, e.g. "0-trainer".
  // The client is not a pod, so its name takes no part in the prefix and is never cut.
  const cut = $derived.by(() => {
    const [first = "", ...rest] = sources.map((p) => p.name).filter((name) => name !== "client");
    let n = 0;
    while (rest.length && n < first.length && rest.every((name) => name[n] === first[n])) n++;
    return rest.length ? first.lastIndexOf("-", n - 1) + 1 : first.length;
  });
  const hues = $derived(new Map(sources.map((p, i) => [p.name, `hue-${i % 8}`])));
  const tag = (name) => (name === "client" ? name : name?.slice(cut));
  const tagWidth = $derived(Math.min(20, Math.max(0, ...sources.map((p) => tag(p.name).length))));
  const match = $derived(view.q ? new RegExp(`(${escapeRegExp(view.q)})`, "i") : null);
  // The API pages newest first; a tail reads down to the newest line.
  const shown = $derived(records.toReversed());
  const paging = $derived(pending && view.end !== null);
  const status = $derived([
    { gke: "Cloud Logging" }[data?.source] || data?.source || "Logs",
    records.length || !error ? `${records.length.toLocaleString()} ${records.length === 1 ? "record" : "records"}` : "",
    pending && !records.length ? "Loading…" : error ? "" : view.end === null ? "Live" : "Paused",
  ].filter(Boolean).join(" · "));
  const stop = () => { controller?.abort(); controller = null; pending = false; };
  async function load(query, page = null) {
    stop();
    const active = controller = new AbortController(), windowEnd = range.now;
    pending = true;
    failedCursor = page;
    try {
      // Cloud Logging's page token is valid only with the same filters and window, which a paused view keeps.
      const { data: response } = await get(inspectUrl("logs", { ...query, ...(page ? { cursor: page } : {}) }), active.signal);
      if (active.signal.aborted) return;
      if (response.error) throw new Error(response.error);
      fromBottom = page ? lines.scrollHeight - lines.clientHeight - lines.scrollTop : 0;
      data = { ...response, queryKey, windowEnd, records: [...new Map([...(page ? records : []), ...(response.records || [])].map((r) => [r.id, r])).values()].slice(0, 2000) };
      error = null;
    } catch (failure) {
      if (!active.signal.aborted) { error = failure.message; if (page && failure.status === 400) failedCursor = null; }
    } finally { if (!active.signal.aborted) { pending = false; controller = null; } }
  }
  $effect(() => {
    if (navigating) return;
    const key = queryKey, follow = view.end === null, filter = filterKey;
    return untrack(() => {
      const changed = filter !== filters?.key, searched = filters && params.q !== filters.q;
      if (changed) { data = error = null; filters = { key: filter, q: params.q }; }
      let timer, active = true;
      const refresh = async (first = false) => {
        if (first || (!document.hidden && !pending)) await load(params);
        if (active && follow) timer = setTimeout(refresh, 5000);
      };
      if (key !== data?.queryKey) timer = setTimeout(() => refresh(true), searched ? 250 : 0);
      else if (follow) timer = setTimeout(refresh, 5000);
      return () => { active = false; clearTimeout(timer); stop(); };
    });
  });
  // Pin the tail to the newest line; an older page keeps the lines in view where they were.
  $effect(() => { shown; lines.scrollTop = lines.scrollHeight - lines.clientHeight - fromBottom; fromBottom = 0; });
  function pause() {
    stop();
    // Event-clipped `until` can be earlier than the inspected window end.
    view.end = data?.windowEnd ?? range.now;
    if (data) data = { ...data, queryKey };
  }
  function follow() {
    view.event = view.end = null;
    if (data) data = { ...data, next_cursor: null, queryKey: null };
  }
  async function older() {
    if (paging || !cursor) return;
    pause();
    // Wait for the paused query's effect cleanup before starting its next page.
    await tick();
    load(params, cursor);
  }
  const timestamp = (value) => Number.isFinite(Date.parse(value)) ? new Date(value).toISOString() : "";
  const severity = (record) => ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"].includes(record.severity) ? "error" : record.severity === "WARNING" ? "warning" : undefined;
</script>

{#if view.event !== null}
  <div class="log-time-scope"><span>{logRange.since.slice(0, 10)} · {logRange.since.slice(11, 19)}–{logRange.until.slice(11, 19)}</span><a href={allLogs}>All logs</a></div>
{/if}
<div class="log-toolbar">
  <input id="log-search" type="search" bind:value={view.q} placeholder="Search logs" aria-label="Search logs" />
  <select id="log-source" aria-label="Pod" bind:value={view.pod}><option value="">All pods</option>
    {#each sources as pod (pod.name)}<option value={pod.name}>{tag(pod.name) || pod.name}</option>{/each}
  </select>
  {#if run.shared_runtime}<label class="log-scope" title="These pods also serve other runs on this base model. Lines that do not name this run are hidden unless this is on."><input type="checkbox" bind:checked={everything} />Include other runs' lines</label>{/if}
  <button type="button" class="chip" onclick={() => view.end === null ? pause() : follow()}>{view.end === null ? "Pause" : "Jump to latest"}</button>
</div>
<div id="log-status" class="log-status" role="status"><span>{status}</span>
  {#if error}<span class="log-error">{error}{records.length ? " · Showing previously fetched records" : ""}</span><span><button type="button" class="chip" disabled={pending} onclick={() => load(params, failedCursor)}>Retry logs</button></span>{/if}
</div>
<!-- svelte-ignore a11y_no_noninteractive_tabindex (The log region supports native keyboard scrolling.) -->
<div id="log-lines" bind:this={lines} tabindex="0" role="region" aria-label="Run logs" style:--tag-width={`${tagWidth}ch`} onscroll={(event) => { const box = event.currentTarget; if (view.end === null && box.scrollHeight - box.clientHeight - box.scrollTop > 24) pause(); }}>
  {#if cursor}<button type="button" class="chip log-older" disabled={paging} onclick={older}>{paging ? "Loading…" : "Older logs"}</button>
  {:else if records.length === 2000}<p class="muted log-older">2,000 records shown. Narrow the time range or search to inspect more.</p>{/if}
  {#each shown as record (record.id)}
    {@const at = timestamp(record.timestamp)}
    <div class="workspace-logrow" data-key={record.id} data-severity={severity(record)}>
      <time class="log-time" datetime={at} title={at || "No timestamp"}>{at ? at.slice(11, 23) : "—"}</time><span class={`log-pod ${hues.get(record.pod) || ""}`} title={record.pod}>{tag(record.pod)}</span><pre class="log-message">{#each match ? record.message.split(match) : [record.message] as part, i}{#if i % 2}<mark>{part}</mark>{:else}{part}{/if}{/each}{#if record.message_truncated}<span class="muted">{"\n[Message truncated by source]"}</span>{/if}</pre>
    </div>
  {:else}{#if !pending && !error}<p class="empty-state">{data?.note || "No logs match this time range and filter"}</p>{/if}{/each}
</div>
