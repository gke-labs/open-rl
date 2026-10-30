<script>
  let { value } = $props();
  const words = $derived(value === "waiting_for_placement" ? "queued" : String(value || "Unknown"));
  const label = $derived(words[0].toUpperCase() + words.slice(1));
  const tone = $derived({ running: "running", ready: "running", live: "running", stalled: "failed", failed: "failed", completed: "completed", ended: "completed", stopped: "neutral" }[label.toLowerCase()] || "pending");
</script>
<span class={`state state-${tone}`}><span class="state-dot" aria-hidden="true"></span>{label}</span>
