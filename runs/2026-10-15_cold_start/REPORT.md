# Cold start on open-rl-dra, measured

Date: 2026-10-15. Cluster `open-rl-dra` (GKE, us-central1), v0.0.6 bundle with dev
api-server and worker images. Demo definition, fixed: one LoRA job on a base model
already in the Hugging Face cache, client on a laptop behind a port-forward, trainer
and sampler on separate nodes. No node pool was recreated and no storage was added.

This replaces the derived timing table in `docs/briefs/cold-start-options.md`
section 5 with measurements. Every number below comes from a primary record;
anything not measured says so.

## 1. How it was measured

- `scripts/cold_start_client.py` runs the demo and stamps each client phase with the
  laptop clock.
- `scripts/cold_start_timeline.py watch|collect|timeline` records, per run: the
  Workload and pod objects (apiserver timestamps), Kubernetes events (kubelet's
  `Pulled ... in Xs` carries the pull it measured), the worker containers' logs with
  containerd timestamps (vLLM and the trainer print their own phase timings), the
  Redis ready keys polled once a second from inside the Redis pod, and the API
  server and scheduler logs. It joins them into `<case>/timeline.md`.
- `scripts/cold_start_case.sh` steers placement with the `openrl.io/<role>` node
  labels, clears the previous workers, and runs one case end to end.
- NFS client counters (`/proc/self/mountstats`) were sampled inside the worker pods
  during the 8B loads: bytes, read ops, bytes per op, round-trip time, queue time.
- `scripts/cold_start_fio.sh` ran fio from an L4 node and an H100 node against real
  safetensors files on the shared volume.
- `scripts/cold_start_monitoring.py` pulled Filestore's `read_bytes_count` and
  `read_ops_count` from Cloud Monitoring for each window.
- `scripts/cold_start_resident.py` (mincore) checked page-cache residency of the
  weights inside a pod.

Clocks: apiserver, nodes, the Redis pod and the laptop all agree to within a second
(container `Started` event versus first log line, and `ready_key` versus the
worker's own "loaded" line, differ by under 1 s in every run).

## 2. Before the fix: four cases, one run each

Worker image `open-rl-server:agent3-9f8dc13`. "Image cached" means the tag was on the
node; "cold" means it was not. Seconds.

| Case | Nodes | Total to first sample | create_model blocked | Trainer image pull | Trainer weights load | First sample blocked | Sampler image pull | vLLM weights load | torch.compile | profile + graph capture |
|---|---|---|---|---|---|---|---|---|---|---|
| B: Qwen2.5-0.5B, L4, image cached | sr2c / bs96 | **132.0** | 17.6 | 0 (present) | 4.2 | 106.7 | 0 (present) | 3.0 | 31.4 | 26.7 |
| A: Qwen2.5-0.5B, L4, image cold | lvw8 / 2208 | **471.0** | 186.7 | 168.6 | 4.3 | 277.0 | 168.6 | 3.3 | 31.5 | 26.6 |
| C: Qwen3-8B, H100, image cold | 1g3a / 0853 | **1350.2** | 1008.6 | 92.5 | **903.5** | 331.4 | 95.3 | 103.6 | 36.9 | 39.3 |
| D: Qwen3-8B, H100, image cached, rerun of the same nodes | 1g3a / 0853 | **182.6** | 52.2 | 0 (present) | 39.9 | 120.7 | 0 (present) | 13.5 | 37.4 | 39.5 |

Fixed costs seen in every run: placement (Workload to scheduled pod) 0.1 to 1.4 s;
container create 0 to 1 s; trainer process start to load start 5 to 7 s; sampler
process start to first log 7 to 12 s, then 18 to 28 s of imports until vLLM's engine
init line; ready key within 1.5 s of the worker's own "loaded" line.

Sources: `<case>/timeline.md`. Image pull durations are kubelet's own, from the
`Pulled` event message. `D-8b-h100-cached-interrupted/` is an earlier attempt at D
whose trainer load (266 s) is valid but whose client was stranded when the laptop's
port-forward dropped for 40 minutes and the reaper released the workers; the
numbers above are from the clean rerun.

### 2.1 The largest phase per case

- **0.5B on L4 (A and B):** the sampler's vLLM start: torch.compile (31 s) plus
  profiling and CUDA graph capture (27 s), 58 s of the 107 s from sampler pod to
  first sample, with another 39 s of process start and imports before it. The
  weights are 3 s. On a cold node the two image pulls (169 s each) dwarf everything.
  The brief's fix for the compile and capture is A3 and A4; not implemented in this
  round (see section 7).
- **8B on H100 (C):** the trainer's weights load, 903 s of 1350. The brief's fix for
  this phase is Group B (new storage), excluded this round. Section 3 shows why a
  code change removes most of it instead.

## 3. Why the trainer takes 15 minutes to read what vLLM reads in 100 seconds

Both workers in case C read the same 16 GB from the same Filestore Basic HDD
instance, on nodes with equally cold page caches (both nodes were created the day
before and had never run this model). From `/proc/self/mountstats` inside each pod:

| Reader | Wall | Bytes from server | NFS READ ops | Bytes per op | Mean RTT per op | Ops in flight (RTT x rate) | Throughput |
|---|---|---|---|---|---|---|---|
| Trainer, `transformers.from_pretrained` (mmap) | 903 s | 16.4 GB | 63,031 | 260 KiB | 27.9 ms | about 2 | 10.8 MiB/s (60 s sample: 43.5 ops/s, 46 ms RTT, 0 ms queue, CPU 4 millicores) |
| Sampler, vLLM 0.30 loader | 102.5 s | 16.4 GB | 19,591 | 846 KiB | 11.0 ms | about 2 | 153 MiB/s |

The trainer's reads are page faults on a memory-mapped file: one request of
roughly the NFS readahead window at a time, each waiting a full disk round trip,
with the CPU idle. vLLM 0.30 detects the network filesystem and first streams the
files into the page cache with 8 threads and 16 MiB reads (its log:
`Filesystem type for checkpoints: NFS. Checkpoint size: 15.26 GiB`, then
`Prefetching checkpoint files into page cache started (in background,
num_threads=8, block_size=16777216 bytes)`), so its mmap afterwards hits memory.
The difference is concurrency and request size, not the storage.

Two more facts the before-runs established:

- **The node's page cache does not survive the pod.** Inside the case C pods, mincore
  showed all 15.3 GiB resident. The next trainer pod on the same node (case D, 50 s
  later) read all 16.4 GB from the server again (mountstats: 15.06 GB after 167 s).
  Filestore is mounted per pod by the CSI driver and unmounted when the last pod on
  the node releases it, which drops the cache.
- **Filestore's server-side cache does matter.** The per-read round trip for the
  same files fell from 27.9 ms (case C, first read in days) to 5.2 ms (first attempt
  at D) to 1.15 ms (clean D rerun), and the trainer's serialized load fell from
  903 s to 266 s to 40 s with no change in its read pattern. A demo repeated on a
  warm instance is a different measurement from the first run of the day.

## 4. The Filestore hypothesis, tested directly

fio from the nodes, O_DIRECT so the node's page cache does not answer
(`fio-*/summary.md`).

Against a file read minutes earlier (hot in the server cache), from L4 node lvw8:

| Job | MiB/s | IOPS | Mean latency |
|---|---|---|---|
| sequential 1 MiB, one stream | 478.7 | 479 | 2.09 ms |
| random 4 KiB, one stream | 4.2 | 1,066 | 0.94 ms |
| sequential 256 KiB, queue depth 1 (the trainer's pattern) | 210.2 | 841 | 1.19 ms |
| sequential 1 MiB, four streams | 1,775.4 | 1,775 | 2.25 ms |

Against a file not read that day (cold on disk), from H100 node 1g3a:

| Job | MiB/s | IOPS | Mean latency |
|---|---|---|---|
| sequential 1 MiB, one stream | 16.2 | 16 | 61.7 ms |
| random 4 KiB, one stream | 0.1 | 33 | 30.2 ms |
| sequential 256 KiB, queue depth 1 | 44.8 (partly cached by the previous job) | 179 | 5.6 ms |
| sequential 1 MiB, four streams | 134.7 | 135 | 29.7 ms |

Answer to the brief's open question: the weights load is **latency-bound, not
bandwidth-bound and not IOPS-bound**. One stream against cold disk gets 16 MiB/s
because each request waits 30 to 60 ms; four streams get 135 MiB/s; a hot server
cache gives 480 MiB/s on one stream and 1.7 GiB/s on four. The "100 MiB/s"
Basic HDD figure in the brief was never the binding constraint in any run.

**Cloud Monitoring cannot confirm client throughput.** `file.googleapis.com/nfs/server/read_bytes_count`
showed about 10 MiB/s during the trainer's cold load (matches), 0 during the
sampler's 153 MiB/s load and during the 1.7 GiB/s fio run (both served from the
server cache), and 7 to 22 MiB/s during the cold fio run that measured 16 to
135 MiB/s. The metric appears to count disk-backed reads with coarse sampling. It is
evidence that cold disk reads happened, not a measure of what clients received.

## 5. The fix

`src/training/prefetch.py`: before `from_pretrained`, read every checkpoint file into
the page cache with 8 threads and 16 MiB reads, in 256 MiB tasks so large shards
spread across threads. Automatic on network filesystems when the files fit in 90%
of available memory, `OPEN_RL_PREFETCH=always|off` to force or disable,
`OPEN_RL_PREFETCH_THREADS` and `OPEN_RL_PREFETCH_BLOCK_MIB` to tune. Wired into the
LoRA, FFT and Automodel trainers. This is the mechanism vLLM already uses on the
sampler side, applied to the trainer.

Also implemented, A6 from the brief: the trainer and sampler ready keys now carry a
JSON record (`ready_at`, `phases`: process start, prefetch result, load start and
end; engine build start and end for the sampler), and `GET /api/v1/openrl/models/{id}`
returns it per worker, so the next slow start can be attributed without logs.

Images: `open-rl-server:coldstart1-dca985a`, `open-rl-api-server:coldstart1-dca985a`
(commit `053fb8d` on branch `feat/cold-start-measure`). Deployed by patching
ConfigMap `open-rl-images` and the api-server Deployment; the v0.0.6 bundle is
otherwise unchanged. The automodel image was not rebuilt.

## 6. After the fix

Worker and api-server images `coldstart1-dca985a`. Same node pairs, same procedure.
The first run on each pair pulls the new tag (cold image); the next runs find it
cached. By the time these ran, the Qwen3-8B and Qwen2.5-0.5B files were hot in
Filestore's server cache (section 3), so the before/after pairs below compare
against the warm-cache "before" where one exists, and the within-run control is
the sampler: vLLM's own 8-thread prefetch loads the same files in the same minute.

| Run | Total to first sample | create_model blocked | Trainer weights load (prefetch + mmap) | First sample blocked | vLLM weights load | compile | capture |
|---|---|---|---|---|---|---|---|
| after-A1: 0.5B, L4, image cold | 493.7 | 197.1 | 15.0 (prefetch 12.4 s at 76 MiB/s, 4 streams: a 0.92 GiB file splits into four 256 MiB tasks; cold disk) | 289.2 | 2.4 | 31.5 | 26.7 |
| after-B1: 0.5B, L4, cached | 157.4 | 15.5 | 2.5 (prefetch 0.7 s, 1,341 MiB/s) | 134.6 (32 s of it between pod created and container started, not seen in the other runs) | 2.5 | 31.4 | 26.9 |
| after-B2: 0.5B, L4, cached | 130.5 | 15.8 | 2.5 | 107.4 | 2.6 | 31.3 | 26.8 |
| after-C1: 8B, H100, image cold | 329.3 | 111.5 (90 s of it the image pull) | **13.0** (prefetch 9.4 s, 1,666 MiB/s) | 207.9 (90 s pull) | 14.0 | 37.5 | 39.9 |
| after-D1: 8B, H100, cached | 154.5 | 25.3 | **13.5** | 119.4 | 12.9 | 37.6 | 39.3 |
| after-D2: 8B, H100, cached | 154.3 | 24.1 | **12.1** | 119.9 | 13.1 | 37.2 | 39.5 |
| after-E1: gemma-4-e2b 9.54 GiB, H100, trainer only, file cold on disk | n/a | 103.7 | **92.5** (prefetch 85.1 s at 115 MiB/s; 9,809 NFS reads of 1 MiB, mean RTT 119 ms, about 14 in flight) | n/a | n/a | n/a | n/a |

Before and after, the phase the fix targets:

| Comparison | Before | After | Server cache state | Factor |
|---|---|---|---|---|
| 8B trainer load, warm server cache | 39.9 s (D) | 12.1, 13.0, 13.5 s (median 13.0) | hot, both | 3.1x |
| 8B trainer load versus vLLM in the same run | 903.5 s vs 103.6 s (C, cold); 39.9 vs 13.5 (D, hot) | 13.0 vs 14.0 (C1); 13.5 vs 12.9 (D1); 12.1 vs 13.1 (D2) | | trainer now equals the 8-thread loader |
| Cold-disk read rate in the trainer | 10.8 MiB/s (60 s sample, case C) to 17 MiB/s (whole load) at 260 KiB per read, about 2 in flight | 115 MiB/s (E1) at 1 MiB per read, about 14 in flight | cold, both | 7 to 11x |
| 8B total to first sample, everything cached | 182.6 s | 154.3, 154.5 s | hot | 1.2x; the rest is vLLM compile + capture (77 s) and startup |
| 0.5B trainer load, hot | 4.2 s | 2.5 s | hot | the prefetch is 0.7 s; no regression |
| 0.5B trainer load, cold disk | 4.3 s (file was hot then) | 15.0 s (file cold, 4 streams) | not comparable | see note below |

The 8B cold-disk before (903 s) could not be rerun after the fix because the files
stayed hot in the server cache; E1 on gemma-4-e2b, read from disk at 119 ms per
request, is the cold-disk measurement of the fixed path. Projected from E1's rate,
the case C trainer load would be about 135 s instead of 903 s; measured directly
only when the cache goes cold again, which this report does not claim.

Not measured three times: the cold-image cases (A, C) ran once each after the fix,
because the only way to make a node forget the image again is to delete it from
containerd on the node, which this round did not do. The cached cases ran twice.

**Where the targeted phase did not shrink:** the 0.5B cold-disk load went from 4
to 15 s. The old path read a hot file in 4 s; the new path read a cold file with
four streams. The honest statement is that for a 0.92 GiB file the prefetch's
256 MiB task size leaves half the threads idle; a smaller task size for small
checkpoints is the obvious follow-up, not measured here.

Side effect worth noting: the sampler's vLLM load fell from 103.6 s (C) to 14 s
(C1) between the two cold-image 8B runs. That is the server cache warming between
runs, not the fix; it shows how much the first demo of the day differs from the
second.

## 7. What this round did not do, with the numbers to decide it

- **A3/A4, the sampler's compile and capture:** 31 to 37 s of torch.compile and 27
  to 40 s of profiling plus CUDA graph capture on every sampler start, in all eight
  runs. vLLM writes its compile cache to `/root/.cache/vllm` inside the container,
  which dies with the pod. For the 0.5B demo this is the largest phase once the
  image is on the node.
- **Image pulls:** 169 s per pull on L4 (pd-balanced boot disk), 92 to 95 s on H100
  (local SSD), measured by kubelet, every time a node lacks the tag; two pulls per
  cold demo. A2 (pre-pull) and C1 (image streaming) in the brief.
- **Keeping the page cache across pods** needs the volume to stay mounted on the
  node (a pod that holds it) or a node-local copy: B2 in the brief.
- **The reaper during a connectivity drop** ended one measurement run; the client was
  never told (issues #246, #247; design 016 phase 2).

## 8. What confirms this independently of this work

- kubelet's own pull durations: `kubectl -n openrl-system get events --field-selector reason=Pulled -o custom-columns='T:.firstTimestamp,P:.involvedObject.name,M:.message'`,
  or after the pods are gone, `gcloud logging read 'logName="projects/cdrollouts-sunilarora/logs/events" AND jsonPayload.reason="Pulled" AND jsonPayload.involvedObject.name:"orw-"' --freshness 1d --format='value(timestamp,jsonPayload.message)'`.
- vLLM's own lines in the sampler log: `kubectl -n openrl-system logs <sampler pod> | grep -E 'Loading weights took|torch.compile took|Graph capturing finished|init engine'`;
  the trainer's `[prefetch] done:` and `Successfully loaded.` lines with `--timestamps`.
- The NFS client counters, which neither OpenRL nor vLLM write: `kubectl -n openrl-system exec <pod> -- grep -A40 /mnt/shared /proc/self/mountstats`
  (`bytes:` field 5 is bytes read from the server; `READ:` fields are ops, ..., total RTT ms).
- Cloud Monitoring, for disk-backed reads only: `python3 scripts/cold_start_monitoring.py --start ... --end ... --out /tmp/m`.
- A stopwatch rehearsal of the demo by someone else, from a fresh terminal:
  `uv --project examples run python scripts/cold_start_client.py base_model=Qwen/Qwen3-8B out=/tmp/rehearsal.jsonl`,
  reading `t` in the printed lines; run it once on a node pair that has not run the
  model that day and once again two minutes later.
