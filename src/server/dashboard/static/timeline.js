import { escape, encode } from "./ui.js";

// Only explicit operational failures and restarts deserve a shortcut above logs.
// Normal lifecycle/checkpoint events remain available in the raw run response.
const incidentReasons = new Map([
  ["OOMKilled", "error"],
  ["CrashLoopBackOff", "error"],
  ["Evicted", "error"],
  ["WorkerRestarted", "warning"],
  ["FailedScheduling", "warning"],
  ["Unschedulable", "warning"],
  ["FailedMount", "warning"],
  ["FailedAttachVolume", "warning"],
]);

// Kubernetes repeats may be coalesced. Use their last observed timestamp;
// do not invent a series of occurrences or timestamp a current pod condition.
export function runEvents(run, start, end) {
  return (run.pods || [])
    .flatMap((pod) => {
      const events = (pod.events || []).map((event) => ({
        ...event,
        pod: pod.name,
        at: Date.parse(event.last_seen_at || event.first_seen_at) / 1000,
        tone:
          incidentReasons.get(event.reason) ||
          (event.reason === "BackOff" &&
          event.type === "Warning" &&
          (pod.restarts > 0 ||
            String(pod.problem || "").split(":", 1)[0] === "CrashLoopBackOff" ||
            /back-off restarting failed container/i.test(event.message || ""))
            ? "warning"
            : null),
      }));
      for (const container of pod.containers || []) {
        for (const termination of [
          container.state === "terminated" ? container : null,
          container.last_termination,
        ]) {
          if (termination?.reason !== "OOMKilled") continue;
          events.push({
            pod: pod.name,
            container: container.name,
            reason: "OOMKilled",
            tone: "error",
            at: Date.parse(termination.finished_at) / 1000,
            message: `${container.name} exited with code ${termination.exit_code ?? "unknown"}`,
          });
        }
        if (container.state === "running" && container.restart_count > 0)
          events.push({
            pod: pod.name,
            container: container.name,
            reason: "WorkerRestarted",
            tone: "warning",
            at: Date.parse(container.started_at) / 1000,
            restart_count: container.restart_count,
            message: `${container.name} is running after a restart`,
          });
      }
      return events;
    })
    .filter(
      (event) =>
        Number.isFinite(event.at) && event.at >= start && event.at <= end,
    )
    .sort((a, b) => b.at - a.at);
}

export function runIncidents(run, minutes, end) {
  const unique = new Map();
  const events = runEvents(run, end - minutes * 60, end);
  for (const event of events) {
    const key = `${event.pod}\0${event.container || ""}\0${event.reason}`;
    if (
      !event.container &&
      events.some(
        (previous) =>
          previous.container &&
          previous.pod === event.pod &&
          previous.reason === event.reason &&
          Math.abs(previous.at - event.at) < 2,
      )
    )
      continue;
    if (event.tone && !unique.has(key)) unique.set(key, event);
  }
  const incidents = [...unique.values()].slice(0, 3);
  if (!incidents.length) return "";
  return `<section class="run-incidents" aria-label="Significant events">${incidents
    .map((event) => {
      const timestamp = new Date(event.at * 1000).toISOString();
      const label =
        event.reason === "WorkerRestarted" ? "Worker restarted" : event.reason;
      return `<div class="run-incident incident-${event.tone}"><time class="incident-time" datetime="${timestamp}" title="${timestamp}">${timestamp.slice(11, 19)}</time><span class="incident-label">${escape(label)}</span><span class="incident-context" title="${escape(event.message)}">${escape(event.pod)}${event.container ? ` / ${escape(event.container)}` : ""}${event.restart_count ? ` · ${escape(event.restart_count)} total restarts` : event.count > 1 ? ` · ${escape(event.count)} reported occurrences` : ""}</span><a href="#run/${encode(run.run_id)}/logs" data-event-at="${event.at}" aria-label="View logs near ${escape(label)} on ${escape(event.pod)}">View logs</a></div>`;
    })
    .join("")}</section>`;
}
