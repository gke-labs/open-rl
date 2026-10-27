<script>
  import Status from "./Status.svelte";
  import { inspectUrl, viewLink } from "./state.svelte.js";
  import { elapsedTime, KINDS, runDescription, runLabel } from "./format.js";

  let { inventory, view = $bindable() } = $props();
  const FILTERS = [
    ["all", "All", "Every recorded run"],
    ["active", "Active", "Runs whose workers are placed, or whose placement cannot be read"],
    ["queued", "Queued", "Waiting for a worker to be placed, or for the server to create one"],
    ["finished", "Finished", "Completed, or nothing serves the run and its session is gone"],
  ];
  const group = (run) => ({ ended: "finished", waiting: "queued" })[run.lifecycle] || "active";
  const runLink = (run, tab = "activity") => viewLink(`run/${encodeURIComponent(run.run_id)}/${tab}`);
  const shown = $derived.by(() => {
    const terms = view.q.toLowerCase().split(/\s+/).filter(Boolean);
    return inventory.runs.filter((run) => {
      const text = [run.run_id, run.name, run.model, run.recipe_name, runLabel(run), KINDS[run.fine_tuning_type], ...run.nodes, ...run.pods.map((pod) => pod.name)].filter(Boolean).join(" ").toLowerCase();
      return (view.status === "all" || group(run) === view.status) && terms.every((term) => text.includes(term));
    }).sort((a, b) => Number(group(b) === "active") - Number(group(a) === "active"));
  });
  // A run's session is the newest one that attached it, the last its metadata names.
  const sessionOf = (run) => run.session_ids?.at(-1) ?? null;

  const errors = $derived([...new Set(inventory.errors)]);
  const issues = $derived.by(() => {
    const pods = inventory.cluster.pods.map((pod) => {
      const run = inventory.runs.find((r) => r.pods.some((p) => p.name === pod.name));
      // A problem reads "Reason: detail". The reason is the status; the detail and restarts are the evidence.
      const [reason, detail = ""] = pod.problem.split(/:\s*(.*)/s);
      const evidence = [detail, pod.restarts ? `${pod.restarts} restarts` : ""].filter(Boolean).join(" · ") || "—";
      // Only a pod waiting for a node is not failing.
      const tone = ["Pending", "Unschedulable"].includes(reason) ? "warning" : "error";
      return { issue: reason, resource: pod.name, evidence, tone, link: run && runLink(run, "logs"), label: "Logs" };
    });
    return [...pods, ...(inventory.cluster.nodes || []).filter((node) => node.ready !== true).map((node) => ({ issue: "Node not ready", resource: node.name, evidence: "Ready condition is false or unknown", tone: "error", link: viewLink("nodes", { node: node.name }), label: "Nodes" }))];
  });
</script>

{#snippet healthStatus(label, tone)}
  <span class={`health-status health-${tone}`}><span class="health-dot" aria-hidden="true"></span>{label}</span>
{/snippet}

{#if view.page === "overview"}
  <h1 class="heading">Overview</h1>
  <div class="overview-toolbar">
    <input id="run-search" type="search" placeholder="Search by run ID, model, name, recipe, node or pod" aria-label="Search runs" bind:value={view.q} autocomplete="off">
    <div class="run-filters" role="group" aria-label="Filter by status">
      {#each FILTERS as [key, label, title]}
        <button type="button" class="chip" data-run-filter={key} {title} aria-pressed={view.status === key} onclick={() => view.status = key}>{label} <span class="run-filter-count">{key === "all" ? inventory.runs.length : inventory.runs.filter((run) => group(run) === key).length}</span></button>
      {/each}
    </div>
  </div>
  <div class="job-list run-list">
    {#if shown.length}
      <div class="job-list-head run-row">
        <span>Job</span><span>Status</span><span>Steps</span><span>Elapsed</span><span>Session</span>
      </div>
      {#each shown as run (run.run_id)}
        {@const label = runDescription(run, null)}
        {@const model = (run.model || "Run").split("/").at(-1)}
        {@const session = sessionOf(run)}
        <div class="job-list-row run-row" data-key={run.run_id}>
          <span class="job-identity"><a href={runLink(run)} title={run.run_id}>{model} · <span class="mono">{run.run_id.slice(0, 8)}</span>{#if KINDS[run.fine_tuning_type] || run.fine_tuning_type}{` · ${KINDS[run.fine_tuning_type] || run.fine_tuning_type}`}{/if}</a>{#if label}<span class="muted micro">{label}</span>{/if}</span>
          <span><Status value={runLabel(run)} /></span><span>{run.steps ?? "—"}</span><span>{elapsedTime(run, inventory.observed_at)}</span>
          <span class="mono micro run-session" title={(run.session_ids || []).join(", ") || undefined}>{session ?? "—"}</span>
        </div>
      {/each}
    {/if}
  </div>
  {#if !shown.length}<div class="empty-state">{inventory.runs.length ? "No runs match the search and status filter" : "No runs recorded"}</div>{/if}
{:else if view.page === "health"}
  <h1 class="heading">Health</h1>
  {#each errors as error}<p class="health-message" role="status">{@render healthStatus("Source unavailable", "warning")}<span>{error}</span></p>{/each}
  {#if issues.length}
    <div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Issue</th><th>Resource</th><th>Evidence</th><th></th></tr></thead><tbody>
      {#each issues as issue}<tr><td>{@render healthStatus(issue.issue, issue.tone)}</td><td>{issue.resource}</td><td>{issue.evidence}</td><td>{#if issue.link}<a href={issue.link}>{issue.label} ↗</a>{/if}</td></tr>{/each}
    </tbody></table></div>
  {:else if inventory.cluster.available === true && !errors.length}<p class="health-message">{@render healthStatus("Healthy", "success")}<span>No pod problems, node problems or source errors.</span></p>{/if}
  <p class="run-json-link"><a href={inspectUrl("inventory")}>Diagnostic JSON ↗</a> · <a href="/docs">API reference ↗</a></p>
{/if}
