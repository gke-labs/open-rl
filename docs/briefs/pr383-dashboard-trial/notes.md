# PR #383 (run telemetry and dashboard) on a fresh cluster

Trial of https://github.com/gke-labs/open-rl/pull/383 on `open-rl-dra-use5`
(us-east5), a cluster with no OpenRL stack on it. Started 2026-10-27 evening
PDT (2026-10-28 01:00 UTC). This file is the running log; findings about the
PR itself are collected at the end.

## Branch

- PR head `c7dbd3f` (ShubyM/open-rl `dashboard/main`), five commits, base
  `3eb0c9e` (#368). Upstream main was two commits ahead (#384, shared volume
  storage class and size at render time).
- Rebased onto upstream main `0644063` as local branch `try/pr-383-rebased`
  in worktree `../open-rl-pr383`, head `844424a`. Rebase was clean.
- Checks on the rebased branch: `py_compile` of every changed Python file
  passes; `make -C scheduler/controller vet test` passes.

## Cluster as found

| Item | State |
|---|---|
| Context | `gke_cdrollouts-sunilarora_us-east5_open-rl-dra-use5` |
| Control plane | 1.35.8-gke.1796000, Managed Prometheus on, DCGM metrics component on, logging SYSTEM+WORKLOADS |
| default-pool | 3 x e2-standard-4 |
| kubecon-h100-8g | 1 x a3-highgpu-8g Spot (8 x H100 80GB), 16 local SSDs, GPU driver install `INSTALLATION_DISABLED`, labels `openrl.io/{enabled,trainer,sampler}=true`, `nvidia.com/gpu.present=true`, `gke-no-default-nvidia-gpu-device-plugin=true`, `group.timeslice.io/trainers=true` |
| sandbox-pool | gVisor pool, 0 nodes at the time |
| DRA | no NVIDIA DRA driver, no ResourceSlices; only `mrdma.google.com` and `netdev.google.com` device classes |
| OpenRL | nothing: no `openrl-system` namespace, no CRDs, no PVC |
| Workload Identity | not enabled (same as `open-rl-dra`). Node SA is the default compute SA with scopes `monitoring`, `logging.write`, `devstorage.read_only`; project roles artifactregistry.reader, cloudtrace.agent, storage.objectViewer |
| Filestore CSI | enabled; classes `standard-rwx`, `zonal-rwx`, `regional-rwx`, `enterprise-rwx` present |

## Decisions

- Images go to a new Artifact Registry repo in the cluster's region:
  `us-east5-docker.pkg.dev/cdrollouts-sunilarora/open-rl` (created
  2026-10-28 01:03 UTC). Tag `pr383-844424a`.
- Shared volume: Filestore Zonal (`zonal-rwx`) instead of the bundle's
  `standard-rwx` Basic HDD, created in-region by the PVC.
- Built with Cloud Build from the worktree. First build into the new repo
  seeds its layer cache from the us-central1 repo's `:buildcache` refs
  (config `/tmp/cloudbuild-seed.yaml`, the repo's `cloudbuild.yaml` plus a
  second `--cache-from`).
- Scheduler image built from `scheduler/controller` as the build context
  (`_DOCKERFILE=Dockerfile`): the PR changes the Workload CRD and the
  controller, so the upstream GHCR scheduler tag would not do.

## Builds (Cloud Build, project cdrollouts-sunilarora)

| Image | Build ID | Result |
|---|---|---|
| open-rl-api-server | 2912dfe5 | SUCCESS, pushed |
| open-rl-server | 4fbd3392 | SUCCESS, pushed (about 17 min with the seeded cache) |
| open-rl-client | 760f3505 | SUCCESS, pushed |
| open-rl-scheduler | be62534c | SUCCESS, pushed |

The first four submissions (807c67e5, 883ea021, 9a88b8a7, 7bf0719c) failed
with `invalid tag`: a zsh `for spec in ...; set -- $spec` loop does not
word-split, so each image name carried the whole spec. Resubmitted 01:12 UTC.

## Steps

1. 01:03 UTC: registry created, builds submitted.
2. GPU drivers: the NVIDIA driver installer DaemonSet and the NVIDIA DRA
   driver Helm chart (25.8.0, `nvidiaDriverRoot=/home/kubernetes/bin/nvidia/`),
   same as `scripts/cell_up.sh` on `feat/cold-start-measure`. Pending: the
   session's permission classifier blocked these two cluster-wide installs;
   handed to the user. The driver installer DaemonSet applied at 01:16 UTC.
   The chart refuses its own default (`resources.gpus.enabled=true`) unless
   `gpuResourcesEnabledOverride=true` is also set; `open-rl-dra` was
   installed with both values, and `scripts/cell_up.sh` on
   `feat/cold-start-measure` lacks them and would fail the same way.
   Installed 01:10 UTC (user). The kubelet-plugin DaemonSet then had 0
   desired pods: the chart ships no tolerations, and the GPU pool carries
   `nvidia.com/gpu=present:NoSchedule`; its `system-node-critical` priority
   class also trips GKE's critical-pod quota outside `kube-system`
   (`insufficient quota to match these scopes`, visible on the controller
   Deployment, which stays 0/1 on `open-rl-dra` too). `open-rl-dra`'s
   DaemonSet has `tolerations: [{operator: Exists}]` and no priority class,
   so the fix is `--set 'kubeletPlugin.tolerations[0].operator=Exists'
   --set kubeletPlugin.priorityClassName=""`.
4. Bundle apply (`kubectl apply --server-side --force-conflicts -k
   ../open-rl-pr383/k8s/deploy/gke-use5`) also handed to the user; the
   classifier blocked it. Applied 01:13 UTC: 35 objects. Scheduler and Redis
   up within seconds; api-server and dashboard Pending on the zonal PVC;
   time-slicer in ImagePullBackOff until the worker image push finished at
   01:21 UTC.
3. Overlay `k8s/deploy/gke-use5/kustomization.yaml` (untracked, copied into
   the worktree so it can reach `../dashboard`): `../fft` + `../dashboard`,
   images rewritten to the us-east5 repo, PVC patched to `zonal-rwx`.
   Renders 35 objects; ConfigMap `open-rl-images` picks up the rewritten
   worker and dispatcher images.

5. 01:21 UTC: all pods Running (api-server, dashboard, scheduler, redis,
   time-slicer, snapshot-agent). Dashboard port-forwarded to
   `http://127.0.0.1:8080/dashboard` (loop in the background).
6. 01:21 UTC: `make cluster-e2e E2E_NAMESPACE=openrl-system
   E2E_SCENARIO=fft-gsm8k-rl E2E_IMAGE=<us-east5 client:pr383-844424a>
   E2E_ARGS="base_model=Qwen/Qwen2.5-0.5B-Instruct steps=5"`. Completed
   01:27 UTC, 5 steps, accuracy 0.41 -> 0.36 -> 0.44 -> 0.66 -> 0.81, step
   wall time 99 s then 47-51 s (client: sampling 16-18 s, train_step 28 s,
   save 3-8 s). Trainer and sampler were both placed on the same GPU
   (`GPU-876ee4e3`) with seven idle; binpack default, not the PR's doing.
7. 01:27 UTC: dashboard Deployment env `OPEN_RL_CLIENT_RUNS_ROOT=
   /mnt/shared/open-rl/runs` (see finding 3). Screenshots in `shots/`.

## Run under test

| Item | Value |
|---|---|
| run_id | `31b33c82-39f5-49ac-a2e6-da35865a47fd` |
| session | `sess-b313188013eb` |
| workers | `orw-fft-31b33c82-...-trainer`, `orw-fft-31b33c82-...-sampler`, both on GPU `GPU-876ee4e3-fa1b-b40f-21a9-808cc6f615cc` |
| client dir | `/mnt/shared/open-rl/runs/fft-gsm8k-rl/open-rl-tmp/fft_gsm8k_rl` |

## Findings about the PR

### What worked as described

- **Scheduler stamps placements.** Workload status carried `placedAt` and
  `gpuUUIDs` within seconds of binding; the scheduler logged one
  `placement placed` line per Workload; `/metrics` on port 8080 exported
  `openrl_gpu_held{UUID,node,pod,role,run_id}` for both workers.
- **Records.** Trainer and sampler print `telemetry op|popped|answered`
  JSON lines with the Cloud Logging labels. Redis held every key the
  design lists: `operations:<run>` and `:samples`, `sampled`, `sampled_at`,
  `steps`, `step_times`, `ledger`, two `worker:<id>` records,
  `session_record`, `placements:<runtime>`. Placement records had
  `released_at` after cleanup, and the activity lanes showed `ended: true`.
- **Read API.** `/runs` (lifecycle `placed` during the run, `ended` after,
  5 steps, 0 failures, 0 relaunches), `/runs/{id}/summary` (steps, step
  seconds, last op, last metrics, goodput, ledger counters, saves, grad
  norms, sampler sends and median), `/runs/{id}/steps` (ops per step with
  metric spread), `/runs/{id}/commands?role=sampler` (one record per served
  group: counts 64, 1, 63), `/activity` (lanes, ops, waits), `/logs`
  (`pod=client` pages `logs.log`), `/inventory`.
- **Dashboard.** Overview, Run (Activity lanes, Metrics with client and
  trainer charts, Logs), Nodes and Health render; a source that cannot be
  read shows its error text, as the docs promise. Build inside the api-server
  image's Node stage worked first time.
- **Rebase.** Clean on upstream main `0644063`; scheduler tests pass.

### Needs attention

1. **"client" time is mostly server dispatch time.** Summary goodput for
   the finished run: wall 335 s, train 11 s, checkpoint 28 s, rollout 7 s,
   client 290 s (goodput 0.032). The client measured sampling 128 s and
   train_step 144 s over the five steps, while the sampler's served groups
   took 1-1.5 s each and the trainer's ops 10-12 s per step. The 40 s per
   step between a request leaving the client and a worker popping it is
   attributed to the client. The PR lists "when each request was enqueued
   isn't recorded" as open; on this run that gap is 85% of wall time, so the
   goodput block answers the wrong question until enqueue times exist.
   (Separately worth a look in OpenRL itself: 16 s client-side sampling for
   1.5 s of vLLM work.)
2. **`since_last_step_s` can be negative.** Observed `-32` on a live run
   (`/runs/{id}/summary` at 01:24:42 UTC, last `optim_step` at 01:24:37).
   `facts.py` subtracts the step time from `state["observed_at"]`, the time
   the cached fleet snapshot was read, while the step comes from the run's
   ops cache, which refreshes separately. `since_last_op_s` is clamped with
   `max(0, ...)` one line below, which hides the same thing. Clocks were
   checked: dashboard, trainer and laptop agree to the second.
3. **The repo's own e2e layout is too deep for the client-directory search.**
   `k8s/eval/e2e-client-job.yaml` sets `log_dir=/mnt/shared/open-rl/runs/
   <scenario>` and the recipe writes `open-rl-tmp/<name>` under it, five
   levels below `/mnt/shared`; `client_runs.DEPTH = 3`, so
   `client_directory` was null and the Metrics tab showed "0 metrics".
   Setting `OPEN_RL_CLIENT_RUNS_ROOT=/mnt/shared/open-rl/runs` on the
   dashboard Deployment fixed it (53 client metrics, `logs.log` paged). The
   dashboard manifest or the e2e job should agree on one layout.
4. **Nothing scrapes `openrl_gpu_held`.** The scheduler exports it, but
   neither `scheduler/deploy/base` nor `k8s/deploy/dashboard` adds a
   `PodMonitoring` for the scheduler's `metrics` port, and the docs do not
   mention one. On GKE with managed collection the PromQL join in design
   018 therefore has no right-hand side.
5. **No DCGM exporter in the bundle the dashboard ships with.** The PR
   edits `k8s/deploy/distributed-fft-timeslice/10-dcgm-monitoring.yaml`
   (adds `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, 10 s interval), but `k8s/deploy/fft`
   does not include that file, and GKE's managed `dcgm-exporter` DaemonSet
   requires `cloud.google.com/gke-gpu-driver-version in (default, latest)`,
   which a pool with `gpu-driver-version=disabled` (needed for DRA) never
   has: it had 0 desired pods here and on `open-rl-dra`. So GPU duty, the
   Nodes page and `/accelerators` stay empty on a DRA cluster unless the
   exporter is applied by hand.
6. **GKE reads need IAM the bundle does not set up.** Every Prometheus and
   Cloud Logging read returned 403: this cluster has no Workload Identity,
   the node SA lacks `monitoring.viewer`, and its `logging.write` scope
   cannot read logs at all. The docs say "monitoring.viewer on the workload
   identity" in a manifest comment only. Worth a short setup section:
   Workload Identity (or node roles), `roles/monitoring.viewer`,
   `roles/logging.viewer`, and what the audit-log lookup needs.
7. **Node GPU inventory comes only from DCGM.** With the 403, the H100 node
   showed `gpu_capacity: 0, devices: []` although its ResourceSlice listed
   eight GPUs and the run's placement named a UUID. The Nodes page was an
   empty table under the error. ResourceSlices (or the placements) could
   seed the node list when metrics are unavailable.
8. **Trailing empty step.** After the final `save_state`, `/runs/{id}/steps`
   lists a step 5 with no client metrics while the client ran steps 0-4; the
   post-step ops form a group of their own.

### Small things

- The Dockerfile's Node stage adds `node:22-bookworm-slim` and an `npm ci`
  to every api-server build (fine on Cloud Build, about a minute).
- The dashboard pod's readiness probe hits `/openapi.json` every few
  seconds and logs each hit at INFO.

## Identity (2026-10-28, after the first report)

User chose Workload Identity over node-SA roles: workload pool on the
cluster, GKE metadata server on `default-pool` (rolls its three nodes;
the H100 pool untouched), a dedicated KSA `open-rl-dashboard` (overlay
`dashboard-sa.yaml` + patches: Deployment serviceAccountName, second
subject on the PR's RoleBinding and ClusterRoleBinding), and direct
principal bindings to `roles/monitoring.viewer` and `roles/logging.viewer`
(no Google service account). The dashboard reads credentials with
`google.auth.default` and the cluster name from the metadata server, so
the GKE metadata server serves both. All four commands run by the user;
the classifier blocked them here.

Timeline (UTC, 2026-10-28):
- 01:46 cluster `--workload-pool` update started (the user's first
  invocation was interrupted but the operation kept running; a rerun
  returned CLUSTER_ALREADY_HAS_OPERATION). Done by 01:55.
- 01:49-01:51 IAM: `roles/monitoring.viewer` and `roles/logging.viewer`
  bound to the principal
  `principal://iam.googleapis.com/projects/288171024361/locations/global/workloadIdentityPools/cdrollouts-sunilarora.svc.id.goog/subject/ns/openrl-system/sa/open-rl-dashboard`.
  The project already had a federated principal for `ns/default/sa/open-rl-sa`
  with `roles/cloudtrace.agent`, so this pattern was used here before.
- 01:58 `node-pools update default-pool --workload-metadata=GKE_METADATA`.
  Nodes hlr9 (us-east5-a) and o9ku (us-east5-b) replaced and labelled
  `iam.gke.io/gke-metadata-server-enabled=true`; Redis, scheduler and
  dashboard rescheduled onto them. The third node, 45i1 in us-east5-c,
  could not be replaced: `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` for
  e2-standard-4 (the MIG shows the same error on 2026-10-25). The command
  exited 1 at 02:07, pool status RUNNING_WITH_ERROR, old node still
  serving (api-server only). Rerun the same command later to finish.
  Mitigation: the overlay pins the dashboard with
  `nodeSelector: iam.gke.io/gke-metadata-server-enabled: "true"`.
- 02:12 overlay applied. The new ServiceAccount landed in `default`:
  the overlay kustomization has no `namespace:` of its own (the bundles
  below it set theirs), so a resource added at the overlay level keeps the
  file's namespace. The ReplicaSet reported `serviceaccount
  "open-rl-dashboard" not found` for three minutes. Fixed by writing
  `namespace: openrl-system` into `dashboard-sa.yaml`, deleting the stray
  account, re-applying (02:16). The ReplicaSet controller was then in
  backoff after 16 failures; `kubectl rollout restart` got the new pod up
  at 02:20 (`open-rl-dashboard-68b8ddb5dd-7h7dj`, sa `open-rl-dashboard`,
  node o9ku).
- 02:22 verified with the new identity: `/inventory` errors `[]`;
  `/logs?run_id=...&since=2h` available with 200 records from Cloud
  Logging (sampler shutdown lines, worker pods by owner label);
  `/accelerators` no error but `gpus: []`, and node `devices: []`. So IAM
  was the only thing behind the 403s, and the empty GPU data that remains
  is findings 4 and 5 (no DCGM exporter on the DRA pool, nothing scraping
  `openrl_gpu_held`).

## Metrics backend (2026-10-28 02:20 UTC)

Added to the overlay (user applied 02:23): a copy of
`distributed-fft-timeslice/10-dcgm-monitoring.yaml` as `dcgm-monitoring.yaml`
(kustomize cannot reference files outside the overlay), moved into
`openrl-system` by an overlay-level `namespace:`; `monitoring.yaml` with a
`PodMonitoring` for the scheduler (`app=open-rl-scheduler`, port `metrics`)
and one for sampler pods (`openrl.io/worker` exists, port `http` = 9095,
which the PR adds to sampler pod templates; trainers have no such port).
The kustomize diff showed exactly five new objects. The exporter pod
started on the H100 node within 30 s (image from the us-central1 GKE
release registry). Second run (8 steps) launched at 02:25 to exercise the
GPU-to-run join.

02:28 UTC, about four minutes after the exporter started: `/accelerators`
lists eight `NVIDIA H100 80GB HBM3` with `memory_total_mib 81559`, no
error; `/inventory` shows the H100 node with `gpu_capacity 8, devices 8`
and `errors []`; `placements` lists the second run's trainer on
`GPU-876ee4e3`. So finding 7 (node inventory only from DCGM) holds, and
with DCGM present the inventory is complete. PodMonitoring `.status`
stayed empty throughout; Managed Prometheus scraped anyway.
Second run `d034c113-c3aa-4e18-82af-32fcae2b455b`: trainer placed
02:24:22, sampler 02:24:42, both on `GPU-876ee4e3` again (binpack).

02:29-02:30 UTC, run d034c113 at 5 of 8 steps, every GPU route answers:
- `/query?name=gpu_busy&run_id=...`: one series for `GPU-876ee4e3`
  (gpu 6), the join through the placement UUID works; values 0 -> 1.3 %.
- `/query?name=gpu_util&run_id=...`: `DCGM_FI_DEV_GPU_UTIL` 0-8.7 %.
- `/query?name=vllm_generation_tokens_per_second&run_id=...`: 118 ->
  384 tokens/s, from the sampler PodMonitoring on port 9095.
- `/runs/{id}/summary`: live `requests.unanswered` showed one
  forward_backward, age 1 s, holder alive; ledger enqueued 19 / popped 18 /
  answered 17 / lost 0; samplers 384 sends, median 1.0 s. Goodput again
  0.036 with client 270 of 307 s.
- UI: Activity lanes show Duty 3 % for trainer and sampler
  (`shots/run-activity-live-dcgm.png`); Nodes page shows the H100 node,
  "1/8 claimed", GPU 6 with held spans, 0 % duty on the node
  (`shots/nodes-dcgm.png`).
The 3 % duty on an H100 for a 0.5B GRPO run is the same story as the
goodput split: the GPU is idle most of each step while requests sit in
the dispatch path.

## Second run: diverged at step 7 (02:32 UTC)

The 8-step run d034c113 died with the client getting HTTP 400
`RequestFailedResponse: vLLM weight update failed: ... Sparse checkpoint
patches contain NaN values`. Trainer log at step 7: `Saved sparse delta
(100.0% changed elements, 494032768/494032768)`. Client accuracy per step:
0.31, 0.39, 0.33, 0.59, 0.73, 0.48, 0.38. `/runs/{id}/steps`: step 5 loss
3.56, grad_norm 8160; step 6 loss 8.51, grad_norm 0.0; step 7 no trainer
metrics. `nonfinite` stayed `[]` because the recorded op metrics were
finite; the NaNs were in the weights. Not the PR's problem (the first
run, 5 steps, was fine; this is RL instability with the recipe's defaults
for the 0.5B model) but a good test of the diagnostics:
- `/runs` line: `lifecycle placed, steps 7, failures 0, relaunches 0,
  trainer_requests_unanswered 0` three minutes after the client died.
  `failures` counts failed trainer ops only; the sampler had 64 failed
  records (`RuntimeError`, no message text in the record) and the API
  server had ingested the SDK's error events (`telemetry sdk_event`
  severity ERROR, visible in `/sessions/sess-610ce40746a5/summary` as
  `sdk_errors.count/last`). So the overview said healthy-ish for a dead
  run. -> finding 10.
- `/runs/{id}/summary` did show `samplers.failed: 64` and the ledger
  balanced (23/23/23, lost 0). Workers stayed Running until the session
  reaper released them (~2 min after the client exit).

Root cause (02:40 UTC, from Cloud Logging via `/logs?q=NaN`, the client's
`metrics.jsonl` and the trainer code), chain in order:
1. Step 6 `forward_backward` returned a finite loss (loss:sum 544 vs 228
   at step 5) but a non-finite gradient. Completions had been getting
   longer every step (ac_tokens/turn 217 -> 281 -> 298 toward max_tokens
   512) and reward had fallen from 0.73 to 0.37 over two steps; the exact
   NaN source (bf16 importance-sampling loss on long sequences, most
   likely) is not recoverable without the batch.
2. `fft_trainer_worker.optim_step`: the recipe sets no `grad_clip_norm`, so
   `max_grad_norm = inf` and `torch.nn.utils.clip_grad_norm_` runs with a
   NaN total norm; its coefficient multiplies every gradient, so every
   gradient became NaN, then `optimizer.step()` made every parameter NaN.
   Hence "100.0 % changed elements" in the step-7 delta.
3. `save_weights_for_sampler` wrote that delta; vLLM's patch loader
   rejected it (`Sparse checkpoint patches contain NaN values or
   out-of-range indices`, sampler log 02:31:40); the API server returned
   400 `RequestFailedResponse` to the client's sample call; the recipe
   raised and the Job failed. Unclipped grad norms of 8k-14k through the
   whole run (and run 1) show how little margin this configuration has.
4. Why telemetry did not flag it: `trainer_worker.sanitize_float` turns
   NaN into 0.0 and inf into +-9999 before the metric leaves the worker,
   so the op record held `grad_norm:mean 0.0`, `nonfinite []`, and the
   summary's `optimizer` block (built for exactly this) stayed empty.
   -> finding 11. The trainer also has no non-finite guard before
   `optimizer.step()` (OpenRL issue, not the PR).

## Duty on a shared GPU (user question, 02:35 UTC)

Both lanes of run d034c113 show Duty 3 % because both processes hold
GPU 6 (binpack placed the sampler on the trainer's ledger; the
time-slicer alternates them: vLLM sleep mode frees 70 GiB, trainer
reloads/offloads to pinned CPU memory, ~0.3 s each way). DCGM measures
the device, and design 018 reduces `openrl_gpu_held` per UUID before the
join, so the figure is the GPU's engine-active share, not the lane's; two
lanes showing 3 % is one 3 %, not 6 %. -> finding 9.

## Co-location of FFT trainer and sampler (user, 02:45 UTC)

Both runs: trainer and sampler on GPU 6 with seven GPUs idle, from the
default `binpack` placement plus the time-slicer. The user flags this as
surprising and as a blocker for a disaggregated trainer/sampler setup
under `binpack`: there is no way to ask for separate accelerators when a
seat is free on the trainer's ledger. Recorded in findings.md as an
OpenRL placement observation (not a PR finding) with a proposed issue: a
hint or policy that places trainer and sampler seats on different claims
when capacity allows. The dashboard showed the co-location at a glance.

## Stability check: two Qwen3-0.6B FFT runs, seeds 1 and 2, 10 steps (02:50 UTC)

User, after the NaN: run the FFT GSM8K scenario on Qwen3-0.6B with two
seeds for 10 steps to see whether OpenRL itself is stable, and to give the
dashboard two concurrent runs. The `fft-gsm8k-rl-x2` scenario passes one
`extra` string to both jobs, so seeds cannot differ there; instead two
single-job client Jobs were rendered with `run_cluster_e2e.render_manifest`
and launched side by side: `open-rl-e2e-client-seed1` / `-seed2`,
`log_dir=/mnt/shared/open-rl/runs/fft-gsm8k-rl-seed{1,2}`,
args `base_model=Qwen/Qwen3-0.6B steps=10 extra=seed={1,2}` (the recipe
`examples/autoresearch/recipes/math_rl/train.py` has `seed: int = 0`).
Manifests in `/tmp/e2e-seeds/`. The failed `open-rl-e2e-client` Job from
the diverged run was deleted first.

Result (03:04 UTC): both Jobs Complete, 11 min each, 10 steps, no NaN,
no failed ops, clean exits. Runs `97df6712` (seed 1) and `4a3b708f`
(seed 2).

| step | seed 1 correct | seed 2 correct | seed 1 grad_norm | seed 2 grad_norm | seed 1 ac_tok | seed 2 ac_tok |
|---|---|---|---|---|---|---|
| 0 | 0.56 | 0.17 | 19712 | 13504 | 462 | 502 |
| 1 | 0.42 | 0.52 | 19456 | 17536 | 466 | 468 |
| 2 | 0.59 | 0.31 | 21760 | 14592 | 447 | 484 |
| 3 | 0.45 | 0.27 | 19328 | 15744 | 426 | 460 |
| 4 | 0.39 | 0.63 | 21248 | 15360 | 453 | 423 |
| 5 | 0.47 | 0.39 | 17024 | 15680 | 390 | 447 |
| 6 | 0.58 | 0.63 | 10944 | 9984 | 357 | 345 |
| 7 | 0.66 | 0.59 | 17664 | 16640 | 368 | 362 |
| 8 | 0.64 | 0.67 | 16512 | 11584 | 349 | 389 |
| 9 | 0.52 | 0.66 | 13888 | 15232 | 306 | 357 |

Entropy 0.34-0.58 throughout, KL(sample, train) 3e-4 to 9e-4, no
monotone drift. Accuracy on 64 samples is noisy but both seeds trend up
over the last five steps and completions shorten from ~480 to ~330
tokens (the 0.5B run went the other way, lengthening toward the 512 cap
before it diverged). Gradient norms 10k-22k, again unclipped. Step wall
time 45-74 s with the two runs concurrent, against 48 s for one run
alone, so the second run cost little.

Placement this time: binpack crossed them. GPU `876ee4e3` held seed 1's
trainer and seed 2's sampler; GPU `03954844` held seed 2's trainer and
seed 1's sampler. Two GPUs busy, six idle. Dashboard `/runs` lists all
four runs of the evening, the two Qwen3 runs `ended, steps 10, failures 0`.

Reading: OpenRL's FFT path is stable on Qwen3-0.6B for 10 steps across
two seeds; the earlier NaN is specific to the Qwen2.5-0.5B-Instruct run
(lengthening completions, falling reward, then a non-finite gradient at
step 6) and to the missing non-finite guard and clip default rather than
to the stack. One diverged run out of four is still enough to want the
guard.

## Something the dashboard surfaced

The GKE logs tab for the run (`shots/run-logs-gke.png`) shows the sampler
segfaulting during shutdown at 01:27:23 UTC: `!!!!!!! Segfault
encountered !!!!!!!` in `cuCtxSynchronize` from `torch.accelerator`
teardown, right after the EngineCore received SIGTERM at 01:27:21. The run
had already finished and the API server had requested cleanup, so nothing
counted it as a failure or relaunch, and the client saw a clean exit. It
is an OpenRL sampler shutdown issue, not the PR's, but it is the kind of
thing this view exists for. Worth an issue with the log excerpt.

## Cluster state at the end

`open-rl-dra-use5` runs the FFT bundle plus dashboard from
`try/pr-383-rebased` (`844424a`), images `pr383-844424a` in the us-east5
repo, NVIDIA driver installer + DRA driver 25.8.0, Filestore Zonal 1 TiB
PVC `open-rl-shared-pvc`, dashboard env `OPEN_RL_CLIENT_RUNS_ROOT=
/mnt/shared/open-rl/runs`, dedicated KSA `open-rl-dashboard` pinned to
metadata-server nodes, overlay-level `namespace: openrl-system`, DCGM
exporter DaemonSet (copy of the legacy manifest) and PodMonitorings for
the exporter, the scheduler and sampler pods. Workload Identity on; default-pool 2 of 3 nodes
on GKE_METADATA (45i1 in us-east5-c pending a capacity retry, pool
RUNNING_WITH_ERROR); IAM: monitoring.viewer + logging.viewer on the
dashboard KSA principal only. The e2e Job `open-rl-e2e-client` is left
(TTL 24 h). Port-forward loop to 8080 may still be running locally.

## Turning the KubeCon demo mock into a real demo (assessment, 03:20 UTC)

Mock: `decks/kubecon-keynote/demo-mock.html` (canvas, three scenes on a
100 s clock, pure function of time; `render.sh` -> `mp4/demo-mock.mp4`,
87.5 s at 12 fps; slide 13 embeds it in an iframe). Scene 1 terminal
submits 20 GRPO jobs (gemma-4-e2b + qwen3-0.6b, lora/full, reward v1-v5);
scene 2 fleet view (counters, job table with state and steps bar, 6 GPU
tiles with LoRA shared runtimes and FFT seats, waiting lane, pool-busy
gauge, end card) driven by an invented PLAN; scene 3 Harvey reward curve
0.57 -> 0.72 (numbers from examples/harvey_labs/README.md).

What the PR gives for scene 2: `/inventory` (runs with lifecycle
ended/waiting/placed, nodes with GPUs, placements held now with role, pod,
UUIDs, pods with problems), `/runs` (steps, since-last, failures),
`/activity` (lanes with phase, GPU turns, op intervals), `/query`
(gpu_busy etc.). Enough for every element except: the job's intended
step count (the demo spec knows it), a display name (the SDK sends only
`recipe_name` and `git_rev`; `tag_metadata` keeps only `openrl.*` tags,
so no `name` reaches the server; map run -> job from the Job's client log
instead), and the state split starting/running (worker phase from
`/activity`, or steps > 0).

Pieces to reuse: `scripts/demo_jobs.py` + `demo_rehearsal.sh` +
`demo_report.py` on `feat/cold-start-measure` (N Jobs from kind:model:count
specs, labels `openrl.io/demo-run`); PR #385's `scripts/perf_compare.py`
`job_manifest`/`load_workload` and `k8s/perf/workload.conf` (23-job mix with
`openrl.exclusive=true` tags, run labels, per-job log dirs, client-log
parsing). Scheduler on the PR build has `OPEN_RL_PLACEMENT_STRATEGY=spread`
(dedicated claim first, seat only when none can be cut), which is how to
make 20 jobs use all 8 GPUs without the feature branch's seat cap.

Blockers on use5: gemma-4-e2b not in the HF cache (only Qwen2.5-0.5B-
Instruct, Qwen3-0.6B, gsm8k); the seeding bucket
`gs://cdrollouts-sunilarora-openrl-cache` no longer exists; `open-rl-dra`
still has `models--google--gemma-4-e2b` (9.6 GB) to re-seed from with
`scripts/cell_seed_hf.sh` (recreates the bucket). Reward variants v1-v5 do
not exist in the recipe; use seeds or learning rates and say so.

## Demo app direction (user, 03:25 UTC)

Decisions: the demo app runs in the cluster (no laptop traffic or
connection loss during a screen share), has a basic spec editor for the
job mix with start / stop / stop-all, reads the dashboard's inspect API,
and leaves the dashboard UI untouched until the shape is understood (the
dashboard is a separate PR). Stop must also POST
`/api/v1/delete_model {model_id}` to the API server so GPUs free at once
instead of after the session reaper. Gemma cache: warm it with a one-step
Gemma FFT job through the normal path rather than copying files
(`open-rl-warm-gemma`, launched 03:27 UTC). Scheduler gets
`OPEN_RL_PLACEMENT_STRATEGY=spread` via the overlay so a sweep spreads over
the eight GPUs; apply pending.

## Demo app v1 written (03:40 UTC)

`k8s/deploy/gke-use5/demo-app/` (untracked; copy in the worktree):
`app.py` (FastAPI: `/` page, `/api/inspect/{path}` proxy to the dashboard
Service, `/api/sweep/presets|jobs|start|stop|stop_all`; Jobs rendered
from the e2e client template with labels `app=openrl-demo`,
`openrl.io/demo-run`, per-job log dir `/mnt/shared/open-rl/runs/demo/<run>/<job>`,
`extra=seed=N`; run id read from the client pod log with the regex the e2e
cleanup uses; stop = POST `/api/v1/delete_model` then delete the Job),
`index.html` (Sweep tab: spec editor rows kind/model/count/steps/seed,
presets keynote-20 / qwen-8 / smoke-2, start, stop, stop-all, a streaming
submit log; Fleet tab: the mock's scene 2 canvas fed by `/inventory`,
`/runs?ended=true` and `/query?name=gpu_busy`; Harvey tab: the mock's
scene 3 unchanged), `demo-app.yaml` (ServiceAccount `open-rl-demo`, Role
for jobs create/delete/list and pods + pods/log get/list, Deployment on
the api-server image with the code from a generated ConfigMap, Service
`open-rl-demo:8000`), `kustomization.yaml` (configMapGenerator). Wired
into the overlay; render 47 objects; diff = the demo objects plus the
scheduler's `OPEN_RL_PLACEMENT_STRATEGY=spread`.

## Gemma warm-up result (04:12 UTC)

`open-rl-warm-gemma` (fft-gsm8k-rl, google/gemma-4-e2b, 1 step) completed:
download 9.6 GB without a token (HF only warned about rate limits), one
step trained, `Training completed successfully`. Cache now holds
gemma-4-e2b, Qwen3-0.6B, Qwen2.5-0.5B-Instruct, gsm8k. Flag: step 0
`env/all/correct 0.03` and the printed trajectory answered a different
problem than the GSM8K prompt (the e2e picks `recipes.math_rl.train_gemma`
with renderer `gemma4`); check the Gemma renderer/model pairing before
quoting Gemma numbers in the demo. Worker pods for the warm-up were
recreated around 04:03 (ages 7 min and 3 min at 04:10), cause not
investigated; the job still succeeded.

Demo app deployed 04:09 UTC (user applied; scheduler now
`OPEN_RL_PLACEMENT_STRATEGY=spread`). Port-forward `svc/open-rl-demo
8090:8000`; presets, jobs and the inspect proxy answer. Smoke sweep
`smoke-1` (2 x FFT Qwen3-0.6B, 2 steps) started from the API at 04:11.
Added URL-hash tab selection to index.html (pending re-apply).

## Smoke sweep through the demo app (04:11-04:16 UTC)

`smoke-1`: 2 x FFT Qwen3-0.6B, 2 steps, seeds 1 and 2, started with POST
`/api/sweep/start`. Jobs appeared in the submit log, run ids were read
from the client logs (`35090c1a`, `181a6291`), states went
submitting -> starting -> running -> done, steps 2/2, Jobs Succeeded.
Spread placement: the four workers took four different GPUs
(`03954844`, `93184bf2`, `56732dec`, `7ceb0f4d`); under binpack earlier
runs shared one. Two fixes after the first look: the page sent the
PromQL window as `300s`, which the inspect API rejects (ISO or Unix
seconds only), so the pool gauge stayed at "waiting for DCGM"; and the
gauge now accumulates GPU time held by a job from the polled placements
(the mock's meaning) with DCGM duty as a second line, since duty on
0.6B models is about 1 %. Tabs now follow the URL hash. Re-apply pending.

## Faster refresh and live placements (04:20-04:30 UTC)

User: refresh more aggressively on all tabs. Measured: each proxied route
answers in ~0.3 s, so the page interval was not the limit. The lags are
upstream: the dashboard caches the fleet snapshot 5 s and run ops 10 s,
and the API server records placements only on its 30 s reaper sweep, so
`/inventory.placements` trails a placement by up to 30 s. Changes: the
demo app now reads placements straight from Workload status
(`/api/sweep/placements`: role, owner, pod, node, gpuUUIDs, placedAt,
shared = name starts with `lora-`; Role gained `workloads` get/list);
the page polls jobs and placements every 1 s and runs/inventory every
1-1.5 s, each source on its own timer so a slow one never blocks the
rest; pool tiles and held-time use the live placements, run labels come
from the dashboard's records. Two page bugs from the user's screenshot of
the qwen8 sweep: done jobs showed 0/2 once the dashboard dropped the run
from `/runs` (now fall back to the job's own step count), and a shared
LoRA runtime showed as one chip per run because the dashboard records one
placement per run on a shared runtime (now one chip per Workload, "N
adapters"). qwen8-1 (4 LoRA + 4 FFT Qwen3-0.6B, 5 steps) under spread:
8/8 GPUs busy at +11:30, LoRA sampler on gpu-3 and trainer on gpu-6, FFT
pairs on the other six, nothing waiting.
Finding for the PR (12): on a shared runtime the placement records are
per run, so a consumer cannot tell one shared pod from several; and the
30 s placement sweep is too slow for a live view, Workload status is
seconds behind.

04:33 UTC: qwen8-1 finished, 8 of 8 Jobs Succeeded (LoRA 5/5 each, FFT
5/5 each). After the demo pod restarted it had lost its job-to-run map
and the re-read only scanned the last 400 log lines, so job 10 showed
"submitting" with its workers placed under a bare run id. Fix: scan the
whole client log once and store the run id as the Job annotation
`openrl.io/demo-run-id` (Role gained `patch` on jobs); all ten Jobs now
carry it. Note for port-forward loops: `pkill -f 'port-forward svc/...'`
also kills the `sh -c while` wrapper because its command line contains
the same string; anchor the pattern with `^kubectl`.

## Keynote-20 mix live through the demo app (04:37 UTC, user-started, sweep demo-043751)

Page fixes on the way: jobs newest first, numbered within their sweep;
Fleet tab scoped to the latest sweep (clock, gauge, counters reset per
sweep); tile chips pack into two columns above three holders, six shown
then "+n"; Workload spec fields are `modelID` / `ownerID` (not `owner`),
which is why samplers had shown their phase instead of a job number.
At +3:59: 10 FFT jobs starting/running with their 20 workers spread over
gpu-0..5 (three or four per GPU, time-sliced), the two LoRA trainers
(gemma, qwen; 5 adapters each) on gpu-6 and gpu-7, and the 10 LoRA jobs
waiting because their shared samplers have no seat: 8/8 GPUs busy, 100 %
of GPU time held, DCGM duty 2.3 %. This is the queue-and-backfill scene
the mock invented, now real. Screenshot `shots/demo-fleet-keynote.png`.

04:48 UTC, user question on the keynote mix: "if the LoRA samplers are
pending, why do gpu-6 and gpu-7 show them occupying GPUs?" Verified: the
GPUs are held by the two shared LoRA *trainers* (gemma on `876ee4e3`,
qwen on `03954844`), which are Running, loaded, and have created and
saved five adapters each (jobs 01-05 and 06-10). The two LoRA *samplers*
are Pending with no GPU, so the ten LoRA jobs sit in lifecycle `waiting`.
The chip said "T 5 adapters", read as "job 5"; relabelled "T lora 01–05".
OpenRL observation (not the PR): a runtime's trainer and sampler are
placed independently, so one half can hold a GPU idle for the whole wait
while the other half has no seat. Gang placement of the pair, or letting
the LoRA sampler seat on its trainer's GPU, would have put ten jobs to
work on two otherwise idle H100s.

05:00 UTC, two sweeps in parallel (user): chip text spilled out of the
tiles (long unsorted job lists) and job numbers were ambiguous because
each sweep numbers from 01. Fixes: chip text is clipped to the tile with
an ellipsis; numbers are sorted and collapse to a range; workers that
belong to another sweep's jobs draw as dim "other" chips (the pool stays
honest about what is held while the Fleet tab shows one sweep); shared
chips drop the word "lora" since the teal stroke already says it. A run
id that maps to no known job falls back to its first four hex chars
(seen as "T 06e9"): that happens when the job's annotation is not yet
written; it self-heals on the next poll.

## Keynote-20 sweep timeline (demo-043751, started 04:37 UTC)

- +0:00 20 Jobs submitted from the Sweep tab (5 gemma LoRA, 5 qwen LoRA,
  5 gemma FFT, 5 qwen FFT; LoRA 8 steps, FFT 4 steps, seeds 1-5).
- +4:00 10 FFT jobs starting/running, 20 FFT workers over gpu-0..5, LoRA
  trainers on gpu-6/7, LoRA samplers Pending, 10 LoRA jobs waiting,
  8/8 GPUs held (`shots/demo-fleet-keynote.png`).
- +12:47 first three FFT jobs done (qwen 17, 19, 20), gpu-2 and gpu-5
  down to one worker each, LoRA still waiting
  (`shots/demo-fleet-keynote-backfill.png`).
- ~+25:00 (05:02) all 10 FFT done; LoRA samplers placed on freed GPUs;
  LoRA jobs backfilled: 3 done by 05:03, 7 running. Per-job timings to
  follow from the client metrics (watcher running).
- The user started a second sweep (demo-044850, 4 LoRA + 4 FFT qwen) at
  04:48 while the first was still going; the Fleet tab switched to it
  (latest sweep) and showed the first sweep's workers as "other". A sweep
  selector on the Fleet tab would help when several run at once; not
  built yet. DCGM duty across the pool during the overlap: 12 %.
Takeaway for the demo narration: with ten FFT pairs the pool fills at
once and the LoRA half of the mix waits ~25 min for seats; for a
one-minute time-lapse the mix should be lighter on FFT (or FFT steps
fewer) so the backfill happens within the recording, and a LoRA sampler
should be able to land next to its trainer (see the placement
observation).

05:05 UTC keynote sweep final: 14 of 20 Jobs Succeeded; the user stopped
jobs 01-05 (gemma LoRA) and 10 (qwen LoRA) by hand while they were
waiting or just started. From metrics.jsonl per job (first step = time
from client start to the end of step 0; later = last step's time/total):
qwen LoRA 06-09: 8/8 steps, first step 1094-1107 s (sampler seat wait),
later 47-64 s, accuracy 0.19-0.72. gemma LoRA 01-04: 2-3 steps before
stop, first step 1309-1317 s, later 86-111 s, accuracy 0.02-0.13.
gemma FFT 11-15: 4/4, first step 250-344 s, later 102-274 s, accuracy
0.02-0.09. qwen FFT 16-20: 4/4, first step 112-230 s, later 143-306 s,
accuracy 0.17-0.78.
Flags: (1) gemma-4-e2b accuracy is near zero for both LoRA and FFT while
Qwen reaches 0.7; consistent with the warm-up's off-topic sample. Check
whether `google/gemma-4-e2b` is the base checkpoint (vs an instruction-
tuned one) or whether the `gemma4` renderer in `recipes.math_rl.train_gemma`
matches it, before Gemma appears in the demo mix. (2) Every LoRA job
waited 18-22 min for a sampler seat, then trained normally: the
half-placed runtime issue is the main cost in this mix.

05:07-05:10 UTC, "GPUs still occupied" after the sweep: three shared LoRA
Workloads (gemma sampler, qwen sampler + trainer) stayed Running until the
session reaper released them at 05:07:36 and 05:08:06, about 2.5 min
after the last client exited at 05:05 (session TTL 120 s + 30 s sweep).
Normal for shared runtimes: `delete_model` on one adapter cannot free a
runtime other sessions still use, so a stopped LoRA job frees its GPU
only when the last session on that runtime is gone; a stopped FFT job's
dedicated pair goes at once. Did the app's stop call delete_model? The
API server logged 14 `POST /api/v1/delete_model` between 05:04:31 and
05:05:43, matching the 14 jobs that finished on their own (the recipe's
cleanup); none at the time of the six manual stops. Either they were
`kubectl delete job` or the app's stop lost the run id (its cache is
in-memory and the demo pod had restarted). Access logs were off, so it
cannot be told apart. Fixes deployed 05:10: stop reads the run id from
the Job annotation before falling back to the pod log, every stop prints
a JSON `event: stop` line with `released`, and uvicorn access logging is
on. Direct test of the path: `delete_model` with a full run id returns
200 `{"status":"ok"}`; with an 8-char prefix it is 404 `Unknown model`.

## Timeline drill-down (05:15-05:25 UTC)

User asked for drill-down views: an accelerator's timeline (which jobs
take turns on it, in colours) and a job's timeline (its trainer and
sampler on their accelerators, interleaved when they share one), and for
live views to scroll like a monitor trace. Built on the inspection API's
`/activity` route (`node_id=` for an accelerator, lanes filtered by GPU
UUID; `run_id=` for a job): one row per worker (accelerator view) or one
row per device with the trainer and sampler lanes stacked (job view);
faint bar = span held, solid = GPU turns (`intervals`), dim = `waits`,
ticks on the top edge = ops by type; colour per job number; an
"any worker holding the GPU" union row with the busy share of the window
for an accelerator; the window slides with the clock, lanes fetched every
1.5 s (the API caches 5 s / 10 s), a pulsing live edge. Reached from a
selector (accelerator, job, window 15 m / 30 m / 1 h / this sweep), from
clicking a GPU tile or a job row on the Fleet canvas (hit rects), or by
URL `#timeline?gpu=<uuid>` / `#timeline?job=<job>`.
First look at gpu-6 (`shots/demo-timeline-gpu.png`): the gemma LoRA
trainer runtime with five adapter lanes from the stopped sweep (shown by
run prefix since their Jobs are gone) and five from the user's next sweep;
turns are short bursts around each op; union 16 % of the window.

05:27 UTC: the per-worker lane rows were confusing (user); replaced by
one occupancy trace per accelerator: a step at 1 while a worker holds a
turn, coloured by the holder (job number, or role in the job view), 0
when idle; a thin dim band at the base for a worker that has the claim
but no turn; per-trace "occupied N % of the window · now: <holder>"; a
"held by" legend with each holder's share. gpu-0 over 30 min: 51 %
occupied by six workers in turn (T 11 gemma 22 %, S 20 qwen 15 %, ...);
gpu-6: 16 %, the gemma LoRA trainer's bursts. Screenshots
`shots/demo-timeline-gpu0-trace.png`, `shots/demo-timeline-gpu-trace.png`.

05:35 UTC: timeline tooltip (user request). Hover draws a crosshair with
the time at the cursor; over a block: holder (job number, model, colour),
worker role and whether it is a shared runtime, job (sweep, number, kind,
seed), run id, pod, the turn's start, end and duration, and the ops that
ran inside it; over the claim band: who holds the claim, since when, and
how many waits are recorded; over nothing: "idle". `#timeline?...&hover=
fx,fy` places the cursor for stills (like the mock's `?still=`). Screenshot
`shots/demo-timeline-tooltip.png`. Demo app files are in
`k8s/deploy/gke-use5/demo-app/` (main repo, untracked) and mirrored in the
worktree; every change is applied through the overlay.

05:43 UTC: job-view overlay (user request): a toggle "overlay other jobs
on these GPUs" draws, under the job's own turns, every other worker's
turns on the same GPUs (muted, in their job colours; the node's lanes
fetched alongside the run's); the occupancy line reads "this job N % of
the window · M % with K other workers"; the legend lists the others
dimmed; the tooltip labels them "other job". `&others=1` on the URL turns
it on. Job 15 (gemma FFT): trainer on gpu-4 27 % alone, 41 % with two
others (T 14, T 18); sampler on gpu-3 14 % alone, 41 % with S 17 and
S 12. Screenshot `shots/demo-timeline-job-others.png`. Also: the last
tick label is skipped when it would collide with "now".

05:50 UTC: drag-to-select on the timeline (user request, for exact
occupancy numbers). Dragging across the trace selects a window; the
view freezes on it (no live sliding), the fetch is bounded with
`since`/`until`, the axis re-ticks down to 5 s, and every percentage
(this job, with others, legend shares) is computed over exactly that
span; a yellow banner shows the selected range and duration; Escape or
double-click returns to the live window. `&sel=<unix a>,<unix b>` on the
URL opens a selection for stills.

05:55 UTC: user asked how to solve trainers placed / samplers pending and
what gang scheduling would mean. Analysis added to findings.md (placement
section): root cause is per-Workload placement plus LoRA exclusivity; fix
order owner-affinity with a LoRA-pair exception -> pair-aware placement
in the controller -> Kubernetes gang scheduling later; operational
workarounds are LoRA-first submission order and warm LoRA runtimes.

## Weights tab: delta sync in action (2026-10-28 ~17:00 UTC)

User: show the share of weights that change per step for full fine-
tuning, as a tab or link from a job; ideally recorded natively in the
backend. Built a Weights tab (job selector; "weights →" link on the job
timeline; double-click a job row on the Fleet canvas) with headline tiles
(average % changed per step, parameters, deltas published, elements per
sync), one bar per sampler weight version, and a line comparing a full
checkpoint per step with the average delta. Data: `/api/sweep/deltas`
parses the trainer's `Saved sparse delta (...)` lines from the inspection
API's logs route (pages through `next_cursor`), a stopgap. The native fix
is finding 13: the save already returns `density_pct` but the op record
keeps only `result["metrics"]`, so one change in the trainer (or the
telemetry hook) makes it a step metric. Also noted: the session clock
jumped (user away); it is 16:55 UTC, so "since" windows must be long
enough to reach the 05:xx sweeps.

## Delta density as a native op metric (2026-10-28 ~17:10 UTC)

User: if it is easier, add the telemetry in the save op. Done on
`try/pr-383-rebased` as commit e38f54d "Record a sparse delta's density
and size on the save op": `BaseTrainerWorker.last_save_metrics` (empty
by default); the FFT worker sets `delta/density_pct`,
`delta/changed_elements`, `delta/total_elements`, `delta/layers`,
`time/delta_collect`, `time/delta_write` after a sparse-delta save and
clears them after a full save; the processor spreads them into the
`save_state` / `save_weights_for_sampler` results, which is what the op
record keeps; `translate_future_result` pops `metrics` from the three
save types so the SDK's save responses are unchanged (forward_backward
keeps its metrics). Two unit tests added (processor result carries the
metrics; API server strips them for saves only). Affected modules pass
(test_worker_manager, test_trainer_optimizer_correctness,
test_telemetry_faults, test_fft_batch_failure); ruff clean. Images
building: `pr383-delta-e38f54d` server (9e37824c) and api-server
(9b32f73a). The Weights tab now reads `/runs/{id}/steps` first
(`save_weights_for_sampler.metrics['delta/density_pct']`) and falls back
to the trainer log parse; the footer names the source. Note for the PR
author: the dashboard's Metrics tab will show these as trainer charts
automatically once the worker image carries the change.

17:22 UTC: verified on the new images (`pr383-delta-e38f54d`, applied
17:14; api-server, dashboard, demo app, time-slicer and the worker
ConfigMap rolled). Run `delta-1` (FFT Qwen3-0.6B, 3 steps, seed 7, run
2f304942) finished at 17:19. `/runs/{id}/steps` now shows per step under
`save_weights_for_sampler.metrics`: step 0 density 0 (empty delta before
the first step), step 1 11.6 % (69.2 M of 596 M elements, 263 layers,
write 5.7 s), step 2 7.8 % (46.4 M, 2.9 s), step 3 6.0 % (35.7 M, 2.7 s).
`/runs/{id}/summary.last_metrics` carries them too. Finding 13 is
therefore fixed on the trial branch; the client SDK was unaffected (the
run completed through the normal recipe).

17:25 UTC: screenshots `shots/demo-weights-native.png` (Weights tab,
"source: op metrics via /runs/{id}/steps": 8.5 % average, 596 M
parameters, 50.4 M elements per sync, 12x fewer than a full checkpoint)
and `shots/dashboard-metrics-delta.png` (the PR's own Metrics tab charts
delta/density_pct, delta/changed_elements, delta/layers,
delta/total_elements, time/delta_collect, time/delta_write for both
save_weights_for_sampler and save_state, with no dashboard change). Two
side observations: (1) the dashboard showed "Client metrics: 0" for demo
runs because the demo app wrote `runs/demo/<run>/<job>/open-rl-tmp/...`,
five levels under the dashboard's root; the demo app now writes
`runs/<job>/...` (three levels), deployed 17:26, so the next sweep's
client metrics appear on the Run page; (2) the dashboard's step charts
label the x axis "0 1 2 2 3" for a four-step run (tick rounding), a small
thing for the PR.

## Sampler started with the trainer (2026-10-28 17:30 UTC)

User asked whether runs set `openrl.recipe=rl`, which makes the API
server start the sampler alongside the trainer (`launch_worker_and_enqueue
(start_sampler=meta.recipe == "rl")`). They did not: session metadata had
only fine_tuning_type, recipe_name and git_rev, and on run 2f304942 the
sampler was placed 22 s after the trainer (lazy start on the first sample
request). Fixes: (1) the demo app's Jobs now set `TINKER_TAGS=
openrl.recipe=rl` (the e2e script merges its own settings into it);
verified on run 56a28037: trainer and sampler placedAt both 17:27:43, and
the session carries `openrl.recipe: rl`. (2) Commit 8cda9c0 on
`try/pr-383-rebased`: `scripts/run_training_e2e.py` sends
`openrl.recipe=rl` for scenarios whose name carries `rl` or `textsql`, so
`make cluster-e2e` runs get it too; client image `pr383-delta-8cda9c0`
building (26111397). Every earlier timing in this file (first step
100-125 s for a 0.6B FFT job, the 17 s client-side sampling waits) was
measured without this tag.

## Sweep page links, sweep-agnostic Fleet view (2026-10-28 17:35 UTC)

User: make jobs on the Sweep page clickable for historic jobs, and make
the Fleet view agnostic of sweeps (anchoring to one sweep breaks when
several are active). Done: every Sweep-page row has timeline / weights /
fleet buttons; opening a finished job frames the timeline to the job's
lifetime (Jobs now carry started_at / completed_at) instead of the empty
live window; the Fleet tab shows all sweeps by default with a selector
to filter to one; sweeps get a letter by recency (A = newest) so tags
read A01, D15 on rows, tiles and the waiting lane; active jobs sort
first in submission order, finished ones after, newest first; the clock
reads "first submit to last finish" once everything shown has ended.
First-step comparison from the recipe tag (same model, solo on an idle
H100): without `openrl.recipe=rl` first step 67 s with 39 s of sampling
wait; with it 48 s and 19 s.
17:40 UTC: demo app Jobs now use client image `pr383-delta-8cda9c0`
(e2e harness with the recipe tag default); for `make cluster-e2e` pass
`E2E_IMAGE=us-east5-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-client:pr383-delta-8cda9c0`.
Polish: the all-sweeps Fleet clock reads "idle · last finish HH:MM:SSZ"
when nothing shown is active; the sweep legend is clipped; the submit
log no longer replays the backlog on page load.

## Branch for review (2026-10-28 17:50 UTC)

Pushed `try/pr-383-rebased` to the fork as `droot/open-rl:feat/dashboard-
trial-use5`: the five #383 commits rebased on main 0644063, then e38f54d
(delta metrics on save ops), 8cda9c0 (e2e recipe tag), ed434f5 (demo app
+ use5 overlay under k8s/deploy/gke-use5, report under
docs/briefs/pr383-dashboard-trial with findings.md, notes.md and shots/,
4.8 MB; `runs/` is gitignored so the report lives in docs/briefs for the
PR). The repo-scoped token cannot create pull requests
("Resource not accessible by personal access token"), so the PR is opened
by hand from the compare URL; the description is in
runs/2026-10-27_pr383_dashboard_use5/pr-body.md. The demo app's app.py
was reformatted for ruff before the commit (two long lines split).

## Rebase onto upstream with sampling pools (2026-10-29)

Upstream main moved to 8d17acb: #385 (perf_compare) and #386 "sample
through sampling pools" (each model's requests go to a Redis stream per
pool; one-replica pools run vllm_sampler in process, multi-replica and
shared-LoRA pools run `vllm serve` Workloads behind a dispatcher pod with
Envoy and the llm-d EPP; the queue sampler and its patch are deleted;
vllm_sampler.py rewritten, 504 lines changed). PR #383 itself has not
moved (head still c7dbd3f). Rebased our stack on a new local branch
`try/pr-383-rebased-2`: #383's five commits, then ours. Conflicts and
how they were resolved:
- "Record run telemetry where it happens": Makefile test list (union,
  minus the deleted tests.test_sampler_patch), api_server imports (both),
  delete_model (upstream's pool-aware body plus `await sweep_placements()`
  before the release), the sample route (upstream's `accept_pool_sample`),
  scheduler_worker_manager (both imports; ops settings added to the
  passed env; pod identity env via `downward()`; both method sets),
  training_requests_processor imports (both), vllm_sampler (took
  upstream's file and re-inserted the hooks: `gpu_turn` around
  `time_slicer.acquire` in `holding_gpu`, `sampled_group` after a
  time-sliced batch, `serve_metrics()` and `serving("sampler", pool)`
  around the worker body, `set_phase("ready")` after READY_FILE; the
  per-request `observe_operation` moved into `sampling_dispatcher.
  Dispatcher.send` as `op_finished` on success and on error, plus a
  group record per dispatch for the exclusive in-process sampler;
  `telemetry.run_of` now also reads the pool record's `job`). The routed
  dispatcher pod (api-server image) records ops too, without pod identity.
- "Serve run facts...": Makefile test list again (dashboard tests).
- "Read client metrics...": docs/configuration.md, both sections kept.
- our delta-metric commit: processor's sampler save returns upstream's
  dict plus `**self.save_metrics()`.
Then ruff --fix and format folded into the telemetry commit. Scheduler
vet/test pass. Python unit tests running (38 modules). Images building as
`pr383-pools-884ddca` (server d1c6139f, api-server c85fd203, client
85aec954); the scheduler image is unchanged (pr383-844424a). #386 needs
`OPEN_RL_TMP_DIR` on the API server (the bundle sets it), Redis with AOF
(bundle), and list on deployments/networkpolicies in the worker RBAC
(bundle). Branch head 884ddca.

Unit tests on the rebased tree: 341 ran, all pass (25 skipped, GPU-only).
One fix was needed in #383's own `TelemetryRecordsTest`: it deletes a
model through the in-memory store, and under #386 the default sampler
backend is vLLM, whose delete path needs Redis (`store.redis`); upstream's
own tests pin `SAMPLING_BACKEND=torch` for in-memory stores, so the
fixture now does the same. Folded into the telemetry commit by autosquash;
branch head is bed05e5 (commits 703dec6 delta metrics, 1e18e1e e2e tag,
bed05e5 demo app). The images tagged `pr383-pools-884ddca` were built
from the pre-fixup tree, which differs only in that test file, so they
are what runs; the tag name is one hash behind the branch.

19:25 UTC: pools stack deployed (api-server, dashboard, demo, time-slicer,
worker ConfigMap on `pr383-pools-884ddca`; the first apply had used the
worktree's reset overlay and left the old tags, re-applied from the
synced copy). Smoke sweep `pools-1` (2 LoRA + 2 FFT Qwen3-0.6B, 2 steps):
all four Succeeded in ~3 min. Under #386 the shared LoRA pool ran as a
`vllm serve` Workload (`lora-qwen-qwen3-0-6b-fb68f5a6-0-sampler`) behind a
dispatcher Deployment `sampler-qwen-qwen3-0-6b-fb68f5a6` (containers
dispatcher = api-server image, envoy, llm-d EPP); FFT pools ran the
in-process sampler. Telemetry on the new paths: every run reports
`samplers.sends 128`, median group 1.8 s (in-process) and 6.1 s (routed);
the dispatcher pod prints `telemetry op` group lines; FFT sampler worker
records exist with turns and GPU UUIDs. Limitation: a routed `vllm serve`
replica runs no worker record, so its lane has no turns (DCGM duty still
covers it). Regression from my merge: `workloads()` and
`pod_terminations()` had landed nested under module-level `sampler_state`
instead of inside `SchedulerWorkerManager`, so the API server logged
"Placement sweep skipped: ... no attribute 'workloads'" and wrote no
placement records (activity lanes empty for the pools-1 runs). Fixed and
folded into the telemetry commit; overlay tags committed too; branch head
d4d7d58; api-server image `pr383-pools-d4d7d58` building (78694880);
server and client stay on `pr383-pools-884ddca` (same source). Branch
force-pushed to `droot/open-rl:feat/dashboard-trial-use5`.

19:30 UTC: api-server `pr383-pools-d4d7d58` applied. Sweep `pools-2`
(1 LoRA + 1 FFT, 2 steps) finished in ~2.5 min; placement records are back:
the FFT run has sampler and trainer lanes with turns (3 and 8) and ops;
the LoRA run's lanes are the shared runtime's (`...fb68f5a6-0-sampler`,
`-0-trainer`), both with turns and ops, so the routed pool's `vllm serve`
replica does appear with turns after all (the dispatcher's group records
land on the pool's sampler Workload pod). "Placement sweep skipped" count
since the fix: 0. Fleet tab on the pools stack: `shots/demo-fleet-pools.png`
(the shared LoRA trainer and sampler Workloads stay up between sweeps,
holding gpu-6 and gpu-7, as before). New: the dashboard's inventory
reported "GPU inventory unavailable: No GPU metrics source: not on GKE and
OPEN_RL_PROMETHEUS_URL is unset" right after its pod was replaced, though
the pod sits on a metadata-server node and can read the metadata server;
investigating whether discovery is one-shot at startup.

19:33 UTC: the "not on GKE" inventory error cleared by itself: `discover()`
reads the metadata server with a 1 s timeout and retries a failed read
after a minute, so the first requests after a dashboard pod starts can
report "not on GKE" for up to a minute. Small thing for the PR: the
message should say the metadata read is pending and will retry.

19:41 UTC: user saw negative occupancy percentages ("this job -205 % of
the window") in a selected window. Cause: the sums kept every turn with
`b > t0`, so a turn starting after a selected window's end was clamped to
the end and contributed a negative length. Fix: only turns overlapping
the window count (`b > t0 && a < t1`) for the job's own, the overlaid and
the union sums. Deployed; the same selection now reads 0 % / 5 % / 8 %.

## Second rebase: upstream 163533d (2026-10-29, evening)

Upstream main gained five commits after 8d17acb: #388 (an inspection
inventory from Redis, `src/server/dashboard/{fleet,router}.py`, mounted on
the API server unconditionally), #389 (dashboard Kubernetes reads,
`dashboard/kubernetes.py`), #391 (DCGM exporter moved to `k8s/deploy/dcgm`
and included by the lora bundle, so fft and distributed-fft-timeslice get
it too), #380 (trainer checkpoint prefetch, `training/prefetch.py`), and
a Makefile test-list change. Upstream has therefore started its own
dashboard under the same package name as #383's.

Rebased `try/pr-383-rebased-2` (d4d7d58) onto 163533d as
`try/pr-383-rebased-3`. Stops and how each was resolved:

- "Record run telemetry...": `store.py`, both sides kept (upstream's
  `scan_values` plus #383's `set_value_sync`, samples, counters). #383 had
  added an identical `scan_values` further down the file, so each class
  ended with two copies (ruff F811); the upstream copies were dropped.
  Makefile test list unioned.
- "Serve run facts...": add/add on `dashboard/fleet.py`, `dashboard/router.py`
  and `tests/test_dashboard.py`. Took #383's versions (upstream's inventory
  is a subset of #383's: run_id, lifecycle, client_status; #383 adds
  placements, workers, nodes and the other routes). Upstream's
  `dashboard/kubernetes.py` and `tests/test_dashboard_kubernetes.py` stay
  untouched next to #383's `sources.py`, which reads the same objects its
  own way; one of the two should go when #383 lands. Upstream's two
  inventory tests were lost with its `test_dashboard.py` (they are written
  against upstream's `fleet.join` signature). `api_server.py` auto-merged
  into a double mount: upstream's unconditional `include_router(router)`
  plus #383's `OPEN_RL_INSPECTION_API=true` block. Kept upstream's; the
  gate is now a no-op and its doc row says so. #383's test
  `test_default_gateway_does_not_expose_fleet_inspection` (expects 404 on
  `/api/v1/inspect/inventory`) contradicts #388 and was removed.
- "Read client metrics...": `docs/configuration.md`, upstream's
  "Inspection API" section kept above #383's "Run telemetry" section.
- Our overlay: `dcgm-monitoring.yaml` copy removed, the lora bundle brings
  `k8s/deploy/dcgm` now; the overlay keeps the scheduler and sampler
  PodMonitorings (`kubectl kustomize` renders 47 objects, one DCGM
  DaemonSet, three PodMonitorings).

Lint clean. Scheduler fmt/vet/manifests/test pass (upstream did not touch
`scheduler/`, so the scheduler image stays `pr383-844424a`). Unit tests:
353 ran, the one failure was the removed test; 352 pass after it. Images
`pr383-pools-c0cd426` building for server, api-server and client
(c0cd426 is the head before the test removal; the trees differ only in
that test file).

01:28 UTC (2026-10-30): overlay applied from the rebased-3 worktree; all
rollouts clean (api-server, dashboard, demo and time-slicer on
`pr383-pools-c0cd426`, worker ConfigMap too; DCGM DaemonSet now the lora
bundle's `k8s/deploy/dcgm` copy, same image). `GET /api/v1/inspect/inventory`
on the API server itself answers 200 now (#388's mount).

01:30 UTC: smoke sweep `pools-3` (1 LoRA + 1 FFT Qwen3-0.6B, 2 steps):
LoRA done in 91 s, FFT in 121 s, both Succeeded. Checks on the new stack:

| Check | LoRA 18e20d46 | FFT 6b8f36ed |
|---|---|---|
| inventory lifecycle / client_status | ended / completed | ended / completed |
| summary steps, sampler sends | 2, 128 (median 3.7 s, routed pool) | 2, 128 (median 1.6 s, in-process) |
| goodput | 0.18 (wall 68 s, client 38 s) | 0.09 (wall 101 s, client 70 s) |
| placement records | 2 (shared runtime's lanes) | 2 |
| activity lanes | sampler 2 intervals, trainer 11, `exact: false` | sampler 3, trainer 8, `exact: true`, trainer ops incl. 4 saves |
| gpu_busy (DCGM) | sampler 3.1 %, trainer 2.5 % | trainer 1.5 %, sampler 1.4 % |
| delta density per step | n/a | step 0: 0 %, step 1: 11.8 % (263 layers), step 2: 7.8 % (261 layers) |

Same shape as pools-2 before the rebase; the trailing third step row and
the routed replica's inexact lanes are as noted earlier. Demo app's job
list carries both run ids (annotation path intact). `/runs` lists only
live runs unless `ended=true`, which is why the two old "placed" runs
(2577a30d, 1a4aed73: the shared LoRA runtime's owners from 10-28 and
10-29, since_last_op 44 h) are the only entries there; nothing new.

01:34 UTC: one difference from the pools-2 picture: the shared LoRA
runtime (`lora-qwen-qwen3-0-6b-fb68f5a6-0-{trainer,sampler}`) no longer
stays up between sweeps. Its Workloads were cut fresh at 01:30:18 for
pools-3 (the api-server restart on the new image had dropped the old
ones) and released at 01:33:49, two minutes after the LoRA run deleted
its model: `[API_SERVER] No live session uses qwen-qwen3-0-6b-fb68f5a6;
tearing its workers down`. That is the existing session-lifecycle
teardown, not a change in #383 or the new upstream commits; before, the
two stale owners of that runtime (2577a30d from 10-28, 1a4aed73 from
10-29, clients that died without delete_model) still had sessions in
Redis and kept it alive. Those sessions are gone now (`open_rl:session:*`
empty), so the Fleet tab shows all eight GPUs idle after a sweep, and
both stale runs read `ended` in the inventory while their
`client_status` stays `active` (finding 10). While pools-3's LoRA run
held the shared runtime they read `placed`, which is finding 12 from the
other side: a shared runtime's placement counts for every run that ever
owned it.

Screenshots on the new stack: `shots/demo-weights-pools3.png` (deltas 12 %
and 7.8 %, 58.4 M of 596 M elements per sync), `shots/demo-timeline-pools3.png`
(sampler on gpu-1, trainer on gpu-7, spread placement), `shots/demo-fleet-pools3.png`
(30 of 30 done, five sweeps lettered A–E).

01:50 UTC: user ran a 14-job sweep (`demo-014306`) and saw two trainers'
turns overlapping on gpu-4 in the accelerator timeline (A14 hold
01:44:29.04 → 01:44:31.65 with a 0.7 s A11 gemma turn inside it at
01:44:30.68). The daemon's timestamped log settles it: Snapshot A13 at
28.04, checkpointed 29.04, A14 granted; Snapshot A14 at 31.65,
checkpointed 33.66. One holder throughout. The inner "turn" is the gemma
trainer's `save_weights_for_sampler` op, run off the GPU because
non-exclusive FFT trainers are CPU-offloaded and `handle_batch` saves
without a turn then; `fleet.activity` unions op spans into `intervals`.
Same for the 0.014 s "turns" of the other two trainers' saves. Finding 14.
Fixed in `fleet.activity` (turns only when the lane has turns), test
`tests/test_dashboard_lanes.py`, commit 886bd93; api-server image
`pr383-pools-886bd93` building.

01:55 UTC: api-server `pr383-pools-886bd93` applied (api-server, dashboard,
demo). The same gpu-4 window re-read through the fixed activity route:
three holds, one per trainer (01:43:45.6–01:44:22.2, 01:44:24.2–28.0,
01:44:29.0–31.7), no overlaps; the saves remain as ops on each lane.
`shots/demo-timeline-gpu4-fixed.png`.
