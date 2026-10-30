# PR #383 (run telemetry and dashboard): trial on a fresh GKE cluster

Date: 2026-10-27 (evening PDT, 2026-10-28 01:00-01:30 UTC)
PR: https://github.com/gke-labs/open-rl/pull/383 (ShubyM, `dashboard/main`, head `c7dbd3f`)
Cluster: `open-rl-dra-use5`, us-east5, 1 x a3-highgpu-8g (8 x H100 80GB, Spot), nothing of OpenRL installed before
Companion files: `notes.md` (step-by-step log), `shots/` (dashboard screenshots)

## Summary

The PR does what its description says. On a cluster brought up from nothing,
the rebased branch built, deployed, ran a five-step GSM8K RL job, and the
telemetry, the inspection API and the dashboard all showed that run
correctly: placements with GPU UUIDs, per-op records, step grouping,
activity lanes, client metrics and client logs.

Thirteen things need attention before this is useful on a GKE cluster as
shipped. Two are in the facts the API computes (goodput attribution, a
negative age), four are deployment gaps (nothing scrapes the scheduler
metric, no DCGM exporter in the bundle, no IAM or Workload Identity story,
client-directory depth does not match the repo's own e2e job), and two are
small. None blocks merging the records and the read API; the deployment
gaps decide whether the GPU charts show anything on a real cluster. Two
more came from a second run that diverged: duty on a shared GPU is shown
per lane but is per device, and a run whose client died on a 400 still
read as healthy on the overview.

## What was tested

| Item | Value |
|---|---|
| Branch | PR head `c7dbd3f` rebased onto upstream main `0644063` (#384); clean rebase; `py_compile` and `make -C scheduler/controller vet test` pass |
| Images | api-server, server, client, scheduler built with Cloud Build from the rebased branch, tag `pr383-844424a`, pushed to a new regional repo `us-east5-docker.pkg.dev/cdrollouts-sunilarora/open-rl` |
| Deployment | `k8s/deploy/fft` + the PR's `k8s/deploy/dashboard`, applied server-side; shared volume on Filestore Zonal (1 TiB) |
| GPUs | NVIDIA driver installer DaemonSet + NVIDIA DRA driver 25.8.0 (Helm), ResourceSlices for 8 GPUs |
| Run | `make cluster-e2e E2E_SCENARIO=fft-gsm8k-rl E2E_ARGS="base_model=Qwen/Qwen2.5-0.5B-Instruct steps=5"`, in-cluster client Job |
| Result | 5 steps, env/all/correct 0.41, 0.36, 0.44, 0.66, 0.81; step wall time 99 s then 47-51 s; trainer and sampler on one GPU (binpack), the other seven idle |

## What worked

**Scheduler.** Workload status carried `placedAt` and `gpuUUIDs` within
seconds of binding. The scheduler logged one `placement placed` line per
Workload and exported `openrl_gpu_held{UUID,node,pod,role,run_id}` on its
metrics port for both workers.

**Records.** Trainer and sampler printed `telemetry op|popped|answered`
JSON lines with the Cloud Logging labels. Redis held every key design 018
lists: `operations:<run>` and `:samples`, `sampled`, `sampled_at`, `steps`,
`step_times`, `ledger`, a `worker:<id>` record per process,
`session_record`, `placements:<runtime>`. After cleanup the placement
records had `released_at` and the activity lanes reported `ended: true`.

**Read API.** `/runs` showed lifecycle `placed` during the run and `ended`
after, 5 steps, 0 failures, 0 relaunches. `/runs/{id}/summary`,
`/runs/{id}/steps`, `/runs/{id}/commands?role=sampler` (one record per
served group: counts 64, 1, 63 at the time of the call), `/activity`,
`/logs?pod=client` and `/inventory` all answered. Errors from sources that
could not be read came back as text in the response, not as empty data.

**Dashboard.** Overview, Run (Activity, Metrics, Logs), Nodes and Health
render. The Node build stage inside `Dockerfile.api_server` worked the first
time on Cloud Build. Screenshots: `shots/overview.png`,
`shots/run-activity.png`, `shots/run-metrics-client.png`,
`shots/run-logs-client.png`, `shots/nodes.png`, `shots/health.png`.

## Findings

### 1. Goodput charges server dispatch time to the client

Summary for the finished run: wall 335 s, train 11 s, checkpoint 28 s,
rollout 7 s, client 290 s, goodput 0.032.

The client's own numbers for the same five steps: sampling 128 s,
train_step 144 s. The sampler's served groups took 1-1.5 s each and the
trainer's ops 10-12 s per step. So about 40 s of every 48 s step is spent
between a request leaving the client and a worker popping it, and the facts
call that "client".

The PR already lists "when each request was enqueued isn't recorded" as
open. On this run that gap is 85% of wall time, so until enqueue times exist
the goodput block points at the wrong party. Suggest either recording the
enqueue time (the API server has it) and adding a "queued" bucket, or
labelling the bucket "not in a worker" until then.

Separately, and not the PR's problem: 16 s of client-side sampling for
1.5 s of vLLM work is worth a look in the dispatch path.

### 2. `since_last_step_s` can be negative

Observed `-32` on `/runs/{id}/summary` at 01:24:42 UTC when the last
`optim_step` was at 01:24:37. `facts.py` subtracts the step time from
`state["observed_at"]`, which is when the cached fleet snapshot was read,
while the step comes from the run's ops cache, which refreshes on its own
schedule. The line below clamps `since_last_op_s` with `max(0, ...)`, which
hides the same thing. Clocks are not the cause: dashboard pod, trainer pod
and laptop agreed to the second. Suggest using the request time for ages.

### 3. The client-directory search does not reach the repo's own e2e layout

`k8s/eval/e2e-client-job.yaml` sets `log_dir=/mnt/shared/open-rl/runs/<scenario>`
and the recipe writes `open-rl-tmp/<name>` under it. That is five levels
below `/mnt/shared`; `client_runs.DEPTH = 3`. Result: `client_directory`
null, Metrics tab "0 metrics", no client logs.

Setting `OPEN_RL_CLIENT_RUNS_ROOT=/mnt/shared/open-rl/runs` on the dashboard
Deployment fixed it: 53 client metrics and `logs.log` paging appeared. The
dashboard manifest and the e2e job should agree on one layout, or the depth
should be larger.

### 4. Nothing scrapes `openrl_gpu_held`

The scheduler exports it, but neither `scheduler/deploy/base` nor
`k8s/deploy/dashboard` adds a `PodMonitoring` for the scheduler's `metrics`
port (8080), and the docs do not mention one. On GKE with managed
collection the PromQL join in design 018 has no right-hand side. Suggest
shipping the `PodMonitoring` with the dashboard manifests.

### 5. The bundle has no DCGM exporter, and GKE's managed one skips DRA pools

The PR edits `k8s/deploy/distributed-fft-timeslice/10-dcgm-monitoring.yaml`
(adds `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, 10 s interval), but `k8s/deploy/fft`
does not include that file. GKE's managed `dcgm-exporter` DaemonSet requires
`cloud.google.com/gke-gpu-driver-version in (default, latest)`, which a pool
created with `gpu-driver-version=disabled` (what DRA needs) never has; it
had 0 desired pods on this cluster and on `open-rl-dra`. So GPU duty, the
Nodes page and `/accelerators` stay empty on a DRA cluster unless the
exporter is applied by hand. Suggest moving the exporter manifest into the
bundle the dashboard is deployed with.

### 6. GKE reads need IAM the bundle does not set up or document

Every Prometheus and Cloud Logging read returned 403 here. The cluster has
no Workload Identity, the node service account has no
`roles/monitoring.viewer`, and its `logging.write` scope cannot read logs
at all. `docs/dashboard.md` says "monitoring.viewer on the workload
identity" in a manifest comment only. Suggest a short setup section:
Workload Identity or node roles, `roles/monitoring.viewer`,
`roles/logging.viewer`, and what the audit-log lookup for relaunches needs.

### 7. Node GPU inventory comes only from DCGM

With the 403, `/inventory` reported the H100 node as `gpu_capacity: 0,
devices: []` although its ResourceSlice lists eight GPUs and the run's
placement names a UUID. The Nodes page was an empty table under the error.
ResourceSlices, or the placements already in Redis, could seed the node
list when metrics are unavailable.

### 8. Trailing empty step

After the final `save_state`, `/runs/{id}/steps` lists a step 5 with no
client metrics while the client ran steps 0-4. The post-step ops form a
group of their own.

### 9. Duty on a time-sliced GPU is the GPU's, shown on every lane

With trainer and sampler sharing GPU 6 (binpack placement, the
time-slicer alternating them), both Activity lanes showed Duty 3 %. The
value is the device's `DCGM_FI_PROF_GR_ENGINE_ACTIVE` share; the join
reduces `openrl_gpu_held` per UUID, as design 018 says, so it cannot be
split by holder. Two lanes at 3 % is one 3 %. Suggest splitting the GPU's
duty by each holder's recorded op intervals (the lanes already have them),
or labelling the column as the GPU's duty when more than one lane holds
the same UUID.

### 10. A run whose client died on a 400 still reads as healthy in `/runs`

Our second run diverged: step 6 recorded loss 8.5 and grad_norm 0.0, at
step 7 the sparse delta was 100 % changed and full of NaN, vLLM rejected
the weight update, and the client exited on HTTP 400. Three minutes later
`/runs` still said `lifecycle placed, failures 0, relaunches 0,
unanswered 0`. `failures` counts failed trainer ops only. The signals were
all recorded elsewhere: 64 failed sampler records (`RuntimeError`, no
message text kept), `samplers.failed: 64` in the summary, and the SDK's
error events in the session summary. Suggest surfacing sampler failures
and SDK errors on the overview line, keeping the error text on failed
sample records, and flagging a live run whose only session has reported
SDK errors and gone quiet. To be fair to the UI: once the workers were
released, the Overview labelled the diverged run "Ended" while the clean
runs read "Completed" (`shots/overview-four-runs.png`), which is the right
distinction; the gap is the two to three minutes while its workers were
still up, and the failure count.

### 11. Workers sanitise NaN to 0.0 before telemetry sees it

Root cause of the diverged run: a non-finite gradient at step 6; with no
`grad_clip_norm` set, `clip_grad_norm_` spread the NaN to every gradient,
Adam made every parameter NaN, the step-7 delta was 100 % NaN, vLLM
rejected it, the client got a 400. The summary's `optimizer` block exists
to catch this, but `trainer_worker.sanitize_float` maps NaN to 0.0 and inf
to +-9999 before the op returns, so the record said `grad_norm:mean 0.0`
and `nonfinite []`. Suggest the telemetry hook read the raw metrics, or
the workers mark sanitised values, so a NaN grad norm is reported as
non-finite. The missing non-finite guard before `optimizer.step()` is an
OpenRL issue to file alongside.

### 12. Placement records are per run on a shared runtime, and 30 s behind

Building a live fleet view on `/inventory` showed two limits. A shared
LoRA runtime with four adapters appears as four placement records, one
per run, all naming the same pod and GPU, so a consumer cannot tell one
shared pod from four pods without grouping by pod itself. And placement
records are written by the API server's 30 s reaper sweep, so a new
placement can trail by half a minute, while the scheduler stamps
`placedAt` and `gpuUUIDs` on Workload status within seconds. For a live
view we read Workloads directly. Suggest one record per Workload with a
list of runs, and reading placement from Workload status (or a watch)
rather than the sweep.

### 13. The delta's density is computed and returned, but never recorded

Under the delta weight-sync strategy the FFT trainer computes, for every
`save_weights_for_sampler`, how many elements changed and what share of
the model that is (`density_pct`, also written to the delta's
`metadata.json` and printed as `Saved sparse delta (3.831% changed
elements, 195570520/5104297504) to …`). The save returns
`{"path", "density_pct"}`, but `command_record` keeps only
`result["metrics"]`, so the op record has `metrics: {}` and
`/runs/{id}/steps` cannot show it. Putting the figure under `metrics`
(for example `delta/density_pct`, `delta/changed_elements`,
`delta/total_elements`, and the save's collect and write seconds, which
the trainer also measures) would make "how much of the model moved this
step" a first-class step metric, with no new plumbing. Observed values:
Qwen3-0.6B 7–100 % (100 % at a NaN step), gemma-4-e2b around 4 % of 5.1 B
elements per step.

Done on the trial branch as commit `e38f54d` ("Record a sparse delta's
density and size on the save op"), offered for the PR or as its own
change: `BaseTrainerWorker.last_save_metrics`, set by the FFT worker after
a delta save (`delta/density_pct`, `delta/changed_elements`,
`delta/total_elements`, `delta/layers`, `time/delta_collect`,
`time/delta_write`) and cleared after a full save; the processor spreads
it into save results; `translate_future_result` drops `metrics` from the
three save types so the SDK's responses are unchanged. Two unit tests.
Verified on the cluster: a three-step Qwen3-0.6B run reported 11.6 %,
7.8 % and 6.0 % of 596 M elements per step through `/runs/{id}/steps`,
and the recipe client ran unchanged.

### 14. Off-GPU ops are painted as GPU occupancy

**Seen:** on the accelerator timeline for gpu-4 during a 14-job sweep, a
0.7 s turn of the gemma trainer (its `save_weights_for_sampler`) sits
inside another trainer's 2.6 s hold of the same GPU. Two holders at once
on a time-sliced claim.

**Checked:** the time-slicer daemon's log (with `--timestamps`) has the
claim held by one worker from 01:44:29.04 to 01:44:33.66 with no hand-off
in between, so the inner turn is not a real hold. The three trainers
share one claim (`claim-fft-…-93124282`, the scheduler moved two of them
onto it). An FFT trainer that is not exclusive runs with CPU offload
(`api_server.py`: `cpu_offload = … or not settings.exclusive`), and
`handle_batch` then runs saves without a time-slicer turn. The saves are
real ops, just not on the GPU.

**Cause:** `fleet.activity` builds a lane's `intervals` as
`merge(turns + every op span)`, so an op outside every turn counts as
occupancy, and `exact` is true as soon as the lane has any turn.

**Fix on the review branch (886bd93):** a lane with turns is occupied
only during them; its ops still show. A lane without turns keeps the ops
as the stand-in. Test `tests/test_dashboard_lanes.py`. The same reasoning
applies to the per-holder shares and the union in any view built on
`intervals`.

### Small things

- For up to a minute after the dashboard pod starts, `/inventory` can
  report "No GPU metrics source: not on GKE" because the metadata-server
  read uses a 1 s timeout and retries a failed read after 60 s. The text
  should say the read is pending and will retry.

- The Metrics tab's step charts label the x axis "0 1 2 2 3" for a
  four-step run: tick values are rounded without de-duplication.

- The Node stage adds `node:22-bookworm-slim` and an `npm ci` to every
  api-server build. About a minute on Cloud Build; fine.
- The readiness probe hits `/openapi.json` every few seconds and each hit
  is an INFO log line.

## Bringing up a fresh GKE cluster for this (what we hit)

Not PR findings, but anyone repeating this will meet them.

- The NVIDIA DRA driver chart 25.8.0 refuses its own default unless
  `--set resources.gpus.enabled=true --set gpuResourcesEnabledOverride=true`
  are given.
- Its kubelet-plugin DaemonSet ships no tolerations and uses
  `system-node-critical`. On a pool tainted `nvidia.com/gpu=present:NoSchedule`
  it schedules zero pods, and GKE's critical-pod quota rejects the priority
  class outside `kube-system`. Fix: `--set 'kubeletPlugin.tolerations[0].operator=Exists'
  --set kubeletPlugin.priorityClassName=""`. `open-rl-dra` was set up this
  way; `scripts/cell_up.sh` on `feat/cold-start-measure` lacks all four
  values and would fail the same way.
- Filestore Zonal provisioning took about seven minutes and one transient
  CSI polling error before the PVC bound; the retry succeeded on its own.
- A Cloud Build into an empty regional registry can seed its layer cache
  from another registry's `:buildcache` with a second `--cache-from`; the
  CUDA worker image took about 17 minutes that way.

## How to reproduce

```sh
# branch
git fetch https://github.com/ShubyM/open-rl.git dashboard/main:pr-383
git worktree add -b try/pr-383-rebased ../open-rl-pr383 pr-383
git -C ../open-rl-pr383 rebase upstream/main

# images (one per Dockerfile; scheduler uses scheduler/controller as context)
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_IMAGE=<repo>/open-rl-api-server,_DOCKERFILE=src/server/Dockerfile.api_server,_TAG=<tag> ../open-rl-pr383

# deploy (overlay: ../fft + ../dashboard, images rewritten, PVC on zonal-rwx)
kubectl apply --server-side --force-conflicts -k ../open-rl-pr383/k8s/deploy/gke-use5
kubectl -n openrl-system set env deployment/open-rl-dashboard OPEN_RL_CLIENT_RUNS_ROOT=/mnt/shared/open-rl/runs

# run and look
make cluster-e2e E2E_NAMESPACE=openrl-system E2E_SCENARIO=fft-gsm8k-rl \
  E2E_IMAGE=<repo>/open-rl-client:<tag> E2E_ARGS="base_model=Qwen/Qwen2.5-0.5B-Instruct steps=5"
kubectl -n openrl-system port-forward svc/open-rl-dashboard 8080:8000
open http://127.0.0.1:8080/dashboard
```

## Observation about OpenRL placement, not the PR

In both runs the scheduler put the FFT trainer and the FFT sampler on the
same GPU (GPU 6, `GPU-876ee4e3`) while the other seven H100s on the node
sat idle. That is the default `binpack` placement booking the sampler's
seat on the ledger the trainer already held, and the time-slicer then
alternating the two processes on one device: vLLM sleeps and frees its
70 GiB, the trainer reloads its weights from pinned CPU memory, runs its
op, offloads again, and the sampler wakes.

This is surprising for someone who expects trainer and sampler to be
separate processes on separate accelerators, and it means a disaggregated
trainer/sampler setup is not possible under `binpack` on a node with free
GPUs: the second worker is co-located by design, and nothing in the
request asks for a GPU of its own. The dashboard made the co-location
visible immediately (both lanes on GPU 6, the Nodes page with one of eight
claimed), which is a point in its favour. The duty figure it shows is then
the shared device's, see finding 9.

A second placement observation from the twenty-job mix under `spread`:
a shared LoRA runtime is two Workloads placed independently. Both LoRA
trainers got a GPU each, loaded, created five adapters, and then sat idle
for the whole run because the two LoRA samplers had no seat while ten FFT
jobs held the other six GPUs. Two H100s held and idle, ten jobs waiting.
Gang-placing the pair, or seating the LoRA sampler on its own trainer's
GPU, would avoid this.

Why the half-placed runtime happens: each worker is its own Workload,
placed on its own; LoRA claims are exclusive by design and never book a
seat on another ledger, so a LoRA sampler can only take a whole free GPU
while its trainer already holds one. Fixes, cheapest first: (1) the
design's own future-work item, owner-affinity, extended with a LoRA-pair
exception so a runtime's trainer and sampler may share one claim when the
second cannot get its own (needs the LoRA trainer to honour the
time-slicer's suspend/resume as the FFT trainer does); (2) pair-aware
placement in the controller, committing both halves' assignments in one
pass or leaving both pending, which keeps the standing pending claim as
the scale-up signal and never lets one half hoard; (3) Kubernetes-level
gang scheduling (coscheduling plugin, Kueue), held until the upstream
feature settles because of its interaction with DRA claims. Without code
changes: submit LoRA jobs before FFT so their samplers take GPUs while
the fleet is empty, or keep the shared LoRA runtimes warm before a sweep.

Worth raising as its own OpenRL issue: a placement hint or policy that
keeps trainer and sampler seats on different claims when capacity allows,
so time-slicing is a fallback for scarcity rather than the default on an
idle node. The seat cap on `feat/cold-start-measure` limits sharing but is
a different control.

## Identity follow-up (same evening)

We enabled Workload Identity on the cluster and the GKE metadata server on
the default pool, gave the dashboard its own Kubernetes service account
(`open-rl-dashboard`), and bound that account's federated principal
directly to `roles/monitoring.viewer` and `roles/logging.viewer`. No
Google service account, no key, nothing granted to the node identity.

Result: `/inventory` reports no errors, `/logs` returns worker lines from
Cloud Logging (200 in the first page), and the Prometheus query succeeds.
It returns no GPUs, which confirms findings 4 and 5: the 403s were only
IAM, and the GPU charts stay empty on a DRA cluster until a DCGM exporter
runs on the GPU nodes and a `PodMonitoring` scrapes the scheduler.

One thing the new Logs tab showed straight away: the sampler segfaults on
shutdown (`cuCtxSynchronize` during `torch.accelerator` teardown, two
seconds after SIGTERM, screenshot `shots/run-logs-gke.png`). The run was
already complete, so no failure was counted and the client saw a clean
exit. Not the PR's problem; worth its own issue.

Two things to note for the PR:

- The dashboard should have its own service account in the manifests.
  Reusing the API server's `open-rl-sa` means any cloud role the dashboard
  needs is also held by the API server.
- A node pool update to turn on the metadata server replaces nodes; one of
  ours could not be replaced for a while because the zone was out of
  e2-standard-4 capacity. A fresh cluster should be created with Workload
  Identity on from the start, which the setup docs could say.

## Metrics backend follow-up (same evening)

We added to our overlay a copy of the PR's `10-dcgm-monitoring.yaml`
(kustomize cannot reference a file outside the overlay), a `PodMonitoring`
for the scheduler's `metrics` port, and a `PodMonitoring` for sampler pods
on the `http` port the PR adds (9095, selector `openrl.io/worker` exists;
trainers have no such port). Five new objects, nothing else changed. DCGM
series reached Managed Prometheus about four minutes after the exporter
started; the `PodMonitoring` status fields stayed empty the whole time.

With that in place, during a second eight-step run:

- `/inventory` listed the H100 node with eight devices and no errors.
- `/query?name=gpu_busy&run_id=...` returned one series for the run's GPU,
  so the `openrl_gpu_held` join works. `gpu_util` and
  `vllm_generation_tokens_per_second` (118 to 384 tokens/s) answered too.
- The Activity lanes showed a DCGM duty of 3 % for trainer and sampler;
  the Nodes page showed all eight GPUs, GPU 6 claimed, held spans drawn.
- The live summary showed an unanswered `forward_backward` with its age
  and that the holder was alive, which is the exact question the PR set
  out to answer.

So findings 4 and 5 are confirmed and have a small, mechanical fix: ship
these three scrape objects and the exporter with `k8s/deploy/dashboard`.
The 3 % duty on an H100 for a 0.5B GRPO run is the same story as finding
1 seen from the GPU: most of each step is spent with the request in the
dispatch path, not on the device.

## Stability check after the NaN

Two concurrent 10-step FFT runs on Qwen3-0.6B with seeds 1 and 2 (separate
client Jobs, since the `-x2` scenario shares one `extra` string) both
completed cleanly: no NaN, no failed ops, entropy and KL flat, accuracy
trending up over the last five steps, completions shortening. Gradient
norms stayed at 10k-22k unclipped. The dashboard listed both as
`ended, steps 10, failures 0` next to the two earlier runs. So the stack
is stable here; the NaN was one run's divergence, which the trainer
should have caught (finding 11 and the missing guard).

Placement note: this time binpack crossed the pairs, trainer of one run
with sampler of the other on each of two GPUs, six idle.

## A setting we had been missing

None of the runs in this report set `openrl.recipe=rl`, the session tag
that makes the API server start the sampler together with the trainer.
Without it the sampler starts on the first sample request, 22 s after the
trainer on an idle H100 here. The e2e harness never sent it; it now does
for RL scenarios (commit `8cda9c0` on the trial branch), and the demo
app's Jobs set it. So the first-step times and the client-side sampling
waits quoted above include a sampler start that the tag removes. Finding
1's point stands either way: the dispatch wait is not recorded.

## Rebase onto sampling pools (2026-10-29)

Upstream main moved to #386, which rewrote the sampler around pools. The
trial stack was rebased on it and re-verified on the cluster with a mixed
LoRA (routed `vllm serve` pool behind a dispatcher pod) and FFT
(in-process sampler) sweep: samples recorded on both paths, lanes with
turns and ops for both, the delta metric per step, the demo app
unchanged. #383's hooks had to move: per-request and per-group sample
records now come from `sampling_dispatcher.Dispatcher`, GPU turns from
`holding_gpu`, the worker record from `run_sampling_worker`; the
`TelemetryRecordsTest` fixture pins the torch backend as upstream's
in-memory tests do. These are on the review branch for #383's author.

### Second rebase, onto 163533d

Upstream then landed its own first dashboard pieces (#388 inventory,
#389 Kubernetes reads) under the same package, `src/server/dashboard`,
plus the DCGM bundle move (#391) and checkpoint prefetch (#380). The
stack was rebased again. What this means for #383:

- **Same package, two designs.** Upstream's `fleet.py` and `router.py`
  are a subset of #383's (one route, runs with lifecycle and
  client_status). The rebase takes #383's files; upstream's
  `kubernetes.py` stays next to #383's `sources.py`, which reads the same
  pods, nodes, Workloads and ResourceSlices its own way. One of the two
  has to go when #383 merges, and upstream's two inventory tests need
  porting to #383's `fleet.join`.
- **The API server now serves `/api/v1/inspect` unconditionally** (#388).
  #383 gated that on `OPEN_RL_INSPECTION_API=true` and tested that the
  default server returns 404 there. The rebase keeps upstream's mount,
  turns the gate into a documented no-op, and drops that test. Whether the
  model-capability API should list every run is a question for the two
  authors; #383's RBAC note (node reads need `k8s/deploy/dashboard`) now
  applies to every API server on GKE.
- **DCGM.** Finding 5 is resolved upstream: the lora bundle includes
  `k8s/deploy/dcgm`, so every GPU bundle ships the exporter and its
  scraper. The overlay's copy is gone. Finding 4 stands: the scheduler
  and sampler PodMonitorings still come from nowhere in the tree.
- **`store.py`.** Both sides added `scan_values`; the merge needs one copy.

Unit tests (352), lint and the scheduler suite pass on the result.

## Open on our side

- Rerun the default-pool metadata update when us-east5-c has capacity; one
  node (api-server only) is still on the old config.
- File the sampler shutdown segfault as its own issue.
- Decide whether to offer the overlay's exporter and `PodMonitoring`
  objects to the PR as a commit on `k8s/deploy/dashboard`.
- The pooled-sampling dispatcher (`SAMPLING_BACKEND=vllm-pools`) was not
  exercised; the PR says it records no samples yet.
- The trial branch `try/pr-383-rebased` and the overlay `k8s/deploy/gke-use5`
  are local and unpushed.
