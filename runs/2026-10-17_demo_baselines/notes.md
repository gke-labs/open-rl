# Fleet demo: measurements and drills (2026-10-17 to 2026-10-18)

## The goal

Run 20 to 25 GSM8K GRPO jobs at the same time. Each job uses 8 prompts, 8 samples per
prompt, and 512 max tokens. Half of the jobs use LoRA and half use full fine-tuning
(FFT). The models are gemma-4-e2b and Qwen3-0.6B. The fleet mixes H100 and L4 GPUs.
The target: 75% of the jobs finish 2 or 3 steps within 2 or 3 minutes of kickoff.

Clients run as in-cluster Jobs from `scripts/demo_jobs.py`. Timings come from each
recipe's own `metrics.jsonl`, read by `scripts/demo_report.py --per-step`.

Cluster: open-rl-dra on GKE. At the start: 4 H100 (2 trainer nodes, 2 sampler nodes)
and 8 L4 (2 trainer nodes and 2 sampler nodes, 2 GPUs each). Storage: Filestore Basic
HDD. Every run below started its workers cold unless the text says otherwise.

## Summary

Scorecard, jobs that finished 3 steps within 180 s:

| Drill | Fleet | Setting | Result |
|---|---|---|---|
| 1 | 4 H100 + 8 L4 | binpack, no fixes | 0 of 12 |
| 2 | 4 H100 + 8 L4 | binpack, all cold-start fixes | 0 of 12, 1 failed |
| 3 | 6 H100 | binpack | 3 of 12 |
| 4 | 6 H100 | binpack, seat limit 2 | **7 of 12** |
| 5 | 6 H100 + 8 L4 | binpack, seat limit 2 | 0 of 12 |
| 6 | 6 H100 | binpack, seat limit 2, 24 jobs | 10 of 24 |
| 7 | 6 H100 | binpack, seat limit 2, LoRA runtimes warm | **8 of 12** |

What we learned, in order of weight:

1. **A cold start cannot meet the window; a warm runtime can.** A sampler needs 60 to
   130 s to start on an H100 after every fix. A shared LoRA runtime with three jobs
   waiting needs 165 s for its first step. Drill 7 ran the same jobs onto runtimes that
   were already up: three gemma LoRA jobs finished three steps in about 90 s instead of
   245 s, and the drill went from 7 to 8 of 12 with the gemma LoRA group moving from the
   slowest to the fastest.
2. **The seat limit works.** With at most two workers per shared claim, no job runs more
   than two-way shared. The 12-job drill on six H100s went from 3 to 7 inside the window.
3. **The L4s hurt as long as binpack may choose them.** Binpack takes the smallest device
   that fits. The LoRA runtimes land on L4s and run three times slower. The same 12 jobs
   went from 7 to 0 when the L4s came back. A device-tier preference is the lever.
4. **gemma FFT cannot run on an L4.** Its weight update needs 11 GiB of transient memory.
   Fixed: the sampler wakes the KV cache after the update, and the claim now says 24 GiB,
   so the scheduler puts it on an H100 by size.
5. **Training was slow for a bad reason.** The cluster never set a token budget, so the
   trainer ran one sequence per pass. The budget is automatic now.
6. **Storage grows under load.** Finished FFT jobs leave 2 to 10 GB each behind. Clean
   between drills. Under a burst, adapter saves on Basic HDD took up to 88 s.

## Part 1: one job alone

Images `coldstart1-dca985a` (prefetch fix in), client `demo1-18409bd`. No token budget set.

| Job | GPU | Step 1, includes worker start | Warm step: sampling + train + save = total | 3 steps done |
|---|---|---|---|---|
| LoRA Qwen3-0.6B | L4 | 187 s (sampler start 150) | 17 + 30 + 1 = 49 s | 284 s |
| FFT Qwen3-0.6B | L4 | 147 s | 18-20 + 21 + 2 = 41-44 s | 232 s |
| LoRA Qwen3-0.6B | H100 | 130 s (sampler start 106) | 5 + 20 + 1 = 26 s | 183 s |
| FFT Qwen3-0.6B | H100 | 194 s (sampler start 170) | 9 + 18-23 + 2 = 29-35 s | 259 s |
| LoRA gemma-4-e2b | H100 | 288 s (sampler start 240) | 9 + 31-35 + 2-4 = 42-49 s | 378 s |
| FFT gemma-4-e2b | H100 | 247 s (sampler start 175) | 15-18 + 31-35 + 12-15 = 58-67 s | 371 s |

The client pod itself: 26 s to pull its image on a node without the tag, then 11 s from
container start to the first API request.

What the numbers say:

- **The worker start dominates the first step.** The sampler takes 106 to 240 s of it:
  vLLM load, compile, and graph capture.
- **Training is the largest warm-step cost, and the H100 barely helps.** A 0.6B model took
  20 s for 64 sequences. The cause: the deployment did not set
  `OPEN_RL_TRAIN_TOKEN_BUDGET`, so `make_training_batches` ran one sequence per
  forward-backward pass, 64 passes per step. The e2e harness sets 65536 locally. The
  cluster never did.
- **Sampling is fine.** 5 to 10 s on H100 and 17 to 20 s on L4 for 64 completions.
- **FFT weight sync is the storage cost per step.** `time/save_checkpoint` is 12 to 15 s
  for gemma on Basic HDD and 2 s for 0.6B. It grows with the fleet.

### Token budget 65536, set on the api-server and forwarded to workers

| Job | GPU | Step 1 | Warm step: sampling + train + save = total | 3 steps done | Train before -> after |
|---|---|---|---|---|---|
| LoRA Qwen3-0.6B | H100 | 110 s | 5 + 3-5 + 1 = 9-11 s | 130 s | 20 -> 4 s |
| FFT Qwen3-0.6B | H100 | 95 s | 10 + 6-8 + 2 = 19-20 s | 134 s | 18-23 -> 7 s |

One environment variable. A warm 0.6B job now does 3 steps in about 30 s on an H100.

gemma-4-e2b failed with 65536. Both gemma trainers got `CUDA out of memory. Tried to
allocate 21.94 GiB` on an 80 GB H100. The client got a 400 and the Job failed within 5
minutes, so the error path works. The cause: gemma has a 262k vocabulary, so the logits
of a 65k-token batch are about 22 GiB in bf16. The budget must depend on the model's
vocabulary and the device. It cannot be one number for the fleet.

### Token budget 8192, gemma-4-e2b on H100

| Job | Step 1 | Warm step: sampling + train + save = total | Train before -> after |
|---|---|---|---|
| LoRA gemma-4-e2b | 257 s | 9-11 + 8-10 + 2 = 19-23 s | 31-35 -> 8-10 s |
| FFT gemma-4-e2b | 229 s | 15-18 + 12-27 + 11-14 = 44-54 s | 31-35 -> 12-27 s |

gemma FFT is now bound by sampling (15 to 18 s) and by the weight sync write on Basic HDD
(11 to 14 s). The FFT sampler goes through the time-slicer even when it is alone.

Code change: the trainer now derives the budget from the model's vocabulary and the free
device memory (`train_token_budget`). It caps any explicit number the same way. No
deployment has to pick a number per fleet.

### Token budget 8192, Qwen3-0.6B on L4

| Job | Step 1 | Warm step: sampling + train + save = total | Train before -> after |
|---|---|---|---|
| LoRA Qwen3-0.6B | 176 s | 17 + 18 + 1 = 36 s | 30 -> 18 s |
| FFT Qwen3-0.6B | 145 s | 19 + 15 + 2 = 36 s | 21 -> 15 s |

On an L4 the step splits evenly between sampling and training. Both are compute-bound
there. A warm L4 job does 3 steps in about 110 s.

### Warm steps after the budget, one job alone

| Job | H100 | L4 |
|---|---|---|
| LoRA Qwen3-0.6B | 9-11 s | 36 s |
| FFT Qwen3-0.6B | 19-20 s | 36 s |
| LoRA gemma-4-e2b | 19-23 s | not run |
| FFT gemma-4-e2b | 44-54 s | not run |

The automatic budget (image `demo2-22fa006`, no budget env) gave the same numbers:
LoRA Qwen3-0.6B on L4 had warm steps of 34 to 36 s with 17 s of training. LoRA gemma on
L4 had warm steps of 77 to 81 s and a sampler start of 190 s, of which 122 s was
torch.compile. gemma does not belong on an L4 for this demo.

## Part 2: jobs that share a GPU pair

All on H100, budget 8192, workers started cold.

### Three Qwen3-0.6B LoRA jobs on one runtime

| Job | Step 1 | Warm steps | 3 steps done |
|---|---|---|---|
| lora 01 | 122 s | 20.6, 21.1 s | 164 s |
| lora 02 | 126 s | 20.5, 18.8 s | 165 s |
| lora 03 | 123 s | 20.5, 21.1 s | 165 s |

A shared LoRA runtime serves three jobs at about 20 s per step each (9 to 11 s alone).
Training serializes and sampling overlaps. All three finished inside 180 s even with the
cold start. Warm, 3 steps take about 60 s.

### Three Qwen3-0.6B FFT jobs time-slicing one pair

Binpack put all three pairs on the same two H100s.

| Job | Step 1 | Warm steps | 3 steps done |
|---|---|---|---|
| fft 01 | 246 s | 26.4, 37.2 s | 310 s |
| fft 02 | 239 s | 31.9, 32.4 s | 303 s |
| fft 03 | 254 s | 25.7, 37.4 s | 317 s |

Time-sliced FFT runs at 26 to 37 s per step each (19 to 20 s alone). Three jobs' warm
steps take about 95 s. The cold start tripled to 220 s because three vLLM samplers
initialize in turn on one GPU.

### gemma-4-e2b

Three LoRA jobs on one runtime: warm steps of 34 to 42 s per job (19 to 23 s alone),
about 110 s for 3 warm steps, and a cold start of 240 to 260 s.

Three FFT jobs: only two could be placed. The third trainer waited with
`WaitingForCapacity ... 1 Insufficient memory` until a job finished. A gemma FFT trainer
pod asks for 91 GiB of host memory for the parked copies that the time-slicer and CPU
offload need. Two of them fill the trainer node's RAM. The two that ran took 44 to 66 s
per step (44 to 54 s alone), about 150 to 180 s for 3 warm steps. That is at the 3-minute
line. gemma FFT needs its own capacity plan: 2 per pair.

### Fleet sizing from these numbers, runtimes warm

| Job type | Jobs per GPU pair | 3 warm steps per job | Limit |
|---|---|---|---|
| LoRA Qwen3-0.6B | 3 | about 60 s | trainer serializes; 4 would still fit |
| FFT Qwen3-0.6B | 3 | about 95 s | time-slice switches |
| LoRA gemma-4-e2b | 3 | about 110 s | trainer serializes |
| FFT gemma-4-e2b | 2 | 150-180 s | 91 GiB host memory per trainer |

For 24 jobs split evenly over the four types, that is 2 + 2 + 2 + 3 = 9 H100 pairs, or
18 H100 GPUs. The gemma FFT jobs decide whether 75% make it. L4s can carry the 0.6B LoRA
share: one job per L4 pair does 3 steps in about 110 s.

Everything in this table assumes warm runtimes. Cold starts measured on this day: 100 to
130 s for a 0.6B sampler, 240 to 260 s for a gemma sampler, and 220 to 330 s when three
samplers start on one time-sliced GPU.

## Part 3: making the cold start fast

### Pre-pull DaemonSet `open-rl-worker-prepull`, first rollout 20:36 UTC

Eight nodes. Three already had the tag. Four H100 nodes pulled it in 1.6 to 2.5 s: the
new tag shares every layer but the last with an image already on the node. One L4 node
had none of the dev layers and pulled the full image. A tag bump costs seconds when the
big layers are present and 90 to 170 s when they are not. For the demo: build the demo
image on the same base the nodes hold, or pre-pull once and keep it.

### vLLM compile cache on the node

hostPath `/var/lib/openrl/vllm-cache`, api-server `demo3-3f5d1bb`. The same LoRA
Qwen3-0.6B sampler started three times on L4 nodes.

| Run | Node | Cache state | torch.compile | engine init | Step 1 (sampling part) |
|---|---|---|---|---|---|
| k1 | 2208 | empty | 55.7 s | 88 s | 176 s (151) |
| k2 | bs96 | empty, other node | 56.0 s | 88 s | 175 s (151) |
| k3 | bs96 | warm from k2 | **0.9 s** | **18 s** | **94 s (70)** |

The cache is keyed by model, GPU type, and engine config. The first sampler of a
combination on a node pays the compile once. Every later one loads the AOT artifact
("Directly load AOT compilation"). Graph capture is not cached and still runs.

### vLLM compile cache on the shared volume

`OPEN_RL_VLLM_CACHE_DIR=/mnt/shared/open-rl/vllm-cache`, api-server `demo4-459e718`.

| Run | Node | Shared cache | torch.compile | engine init | Step 1 (sampling part) |
|---|---|---|---|---|---|
| sc-s1 | bs96 | empty, written here | 67.1 s | 106 s | 195 s (170) |
| sc-s2 | 2208, never ran the model | warm from s1 | **1.1 s** | **20 s** | **95 s (71)** |

The shared directory gives the same gain as the hostPath, from any node, and it survives
node recreation. One model on one GPU type is 61 MB in 913 files. The first write costs
about 10 s more than local disk. The bundle now defaults to the shared directory. The
hostPath stays as the option for installs without a shared volume.

### Where the sampler's start time goes

A probe inside the worker image on an L4 node, no GPU work:

- `uv run python -c pass`: 0.23 s. The uv wrapper is not the cost.
- `import server.vllm_sampler`: 11.0 s (vllm 4.5, vllm.config 3.1, torch 1.8, engine 1.0).
- `import server.training_requests_processor`: 7.3 s.

vLLM V1 runs the engine core as a second process that imports everything again. That is
the 18 to 28 s of "imports until engine init" in the timelines.

### Sampler start A/B, gemma-4-e2b LoRA on H100 node 0853

Images `demo5-8736ff0`, shared compile cache.

| Variant | torch.compile | graph capture, two passes | engine init | first step (sampling part) | warm step |
|---|---|---|---|---|---|
| vLLM default capture list | 102 s (cold entry) | 29 + 24 = 53 s | 199 s | 301 s (285) | 23 s |
| power-of-two capture sizes | 103 s (new entry: the sizes are part of the key) | 19 + 11 = 30 s | 175 s | 272 s (257) | 21 s |
| pow2 + in-process engine | 2.4 s (warm) | 12 + 11 = 23 s | 59 s | 151 s (136) | 24 s |
| pow2, warm, engine as a subprocess | 1.4 s | 12 + 10 = 22 s | **57 s** | **147 s (132)** | 24 s |

Reading it:

- Power-of-two capture saves about 23 s per start on an H100 for gemma.
- The in-process engine changes nothing (57 vs 59 s). Leave vLLM's default.
- gemma's first compile on an H100 is 100 s. That is the one-time cost a warm-up pays per
  model, per GPU type, per fine-tuning type. Change the capture list before warming.
- Warm steps were the same in all variants. None of the knobs cost throughput.

What this delivered for a gemma sampler on an H100: engine init 199 -> 57 s and first
step 301 -> 147 s. The shared cache gave 100 s and power-of-two capture gave 23 s. The
57 s that remain: weights 13 s, compile 1.4 s, capture 22 s, profiling and KV cache
about 20 s. About 40 s of pod start, imports, and config come before the engine line.
About 20 s sit between "ready" and the client's first samples.

## Part 4: the drills

### Drill 1, 21:00 UTC: 12 jobs, cold, binpack

Mix for every 12-job drill: gemma LoRA x3, gemma FFT x2, 0.6B LoRA x3, 0.6B FFT x4.
Pre-pull and the compile cache were in place. Result: **0 of 12**.

| Group | Where the scheduler put it | 3 steps done |
|---|---|---|
| 0.6B LoRA x3, one runtime | L4 | 200-223 s |
| 0.6B FFT x4 | 2 on H100 pairs, 2 on L4 | 312-434 s |
| gemma FFT x2 | H100, samplers packed 3 to a GPU | 465-469 s |
| gemma LoRA x3, one runtime | **L4** | about 630 s |

Two causes:

1. **Cold start.** Every runtime was built after kickoff. The fastest group's first step
   was 88 s. The gemma sampler on an L4 took 192 s to initialize.
2. **Placement by fit, not by speed.** The scheduler packs by memory. A 10 GB gemma
   sampler fits an L4, so the gemma LoRA runtime went there at 80 s per step while two
   0.6B FFT pairs took H100s. A Workload asks for a memory size and a role, nothing else.

Storage after the drill: 17%. `sampler_full` grew from 34 to 66 GB in one day. Finished
FFT jobs leave their sampler weights, checkpoints, and adapters behind: 1.8 GB for 0.6B
and about 10 GB for gemma. The demo needs a cleanup step between drills and a retention
rule.

### Drill 2, 22:43 UTC: the same 12 jobs, cold, every cold-start fix in

The worker image was on every node. The compile cache was warm for the LoRA samplers.
CUDA graphs used the power-of-two list. The token budget was automatic.
Result: **0 of 12**, one failed. The fastest group needed 219 s.

| Group | Where | First step | Warm steps | 3 steps done |
|---|---|---|---|---|
| 0.6B LoRA x3, one runtime | L4 | 105-120 s | 56-61 s | 219-240 s |
| 0.6B FFT x4 | 2 pairs on L4, 1 on H100 | 187-290 s | 40-53 s | 275-376 s, one failed |
| gemma FFT x2 | H100, samplers 3 to a GPU | 422-429 s | 43-58 s | 510-539 s |
| gemma LoRA x3, one runtime | **L4**, same nodes as the 0.6B runtime | 290 s | **164-175 s** | 630 s |

Four things happened only because the jobs started together:

1. **Placement.** Both LoRA runtimes went to the same two L4 nodes. The H100s took the
   FFT pairs. On a shared L4 one gemma LoRA step takes 165 s.
2. **Storage contention.** Sixteen workers started on one Basic HDD volume at once. An
   adapter save takes 1 to 2 s alone. In the burst it took 17 s for 0.6B and 47 to 88 s
   for gemma, on every step.
3. **The compile cache key includes the engine config.** An FFT sampler runs without LoRA
   and with sleep mode, so its key differs from a LoRA sampler's key. The warm-up had
   warmed only the LoRA keys. Each gemma FFT sampler compiled for 100 s, three in turn
   on one GPU. That is the 420 s first step. A warm-up must cover model x GPU type x
   fine-tuning type.
4. **One client failed on NFS.** `OSError: [Errno 116] Stale file handle` from `filelock`
   inside `datasets.load_dataset`. Twelve clients opened one Hugging Face cache on the
   shared volume at once. Fixed later: each client seeds its own cache on an emptyDir.

### Drill 3, 2026-10-18 00:11 UTC: 6 H100 only

Changes since drill 2: both H100 pools grew from 2 to 3 nodes. The four L4 nodes carry
`openrl.io/trainer=false` and `openrl.io/sampler=false`. The compile cache was warm for
all eight keys on H100. Each client seeds its own Hugging Face cache.
Result: **3 of 12**. All 12 succeeded.

| Group | First step | Warm steps | 3 steps done |
|---|---|---|---|
| 0.6B LoRA x3, one runtime | 59-63 s | 20 s | 100-106 s |
| gemma LoRA x3, one runtime | 166-172 s | 34-51 s | 258-261 s |
| gemma FFT x2 | 167, 252 s | 98-148 s | 294, 517 s |
| 0.6B FFT x4 | 235-260 s | 89-148 s | 413-498 s |

Where the 16 workers went, from the Scheduled events:

| GPU | Workers |
|---|---|
| sampler 0853 | gemma LoRA sampler, alone |
| sampler h9sb | 0.6B LoRA sampler, alone |
| sampler ed3t | **all six FFT samplers** |
| trainer 1g3a | gemma LoRA trainer, alone |
| trainer 7zrs | 0.6B LoRA trainer + 1 FFT trainer |
| trainer tuqu | **five FFT trainers** |

The finding: the scheduler packs without a limit. Six FFT samplers time-sliced one H100
while two sampler GPUs each held one LoRA runtime. FFT sampling took 110 to 130 s per
step. A second finding: three jobs on one LoRA runtime serialize the first step, 133 s
of sampling against about 60 s alone.

### Drill 4, 01:51 UTC: 6 H100, seat limit 2

Workers `demo7-87417c2`, scheduler `demo8-943eb38` with `OPEN_RL_MAX_SEATS_PER_CLAIM=2`.
Result: **7 of 12**. All 12 succeeded.

| Group | 3 steps done | Drill 3 |
|---|---|---|
| 0.6B LoRA x3 | 95-97 s | 100-106 s |
| 0.6B FFT x4 | 112-180 s | 413-498 s |
| gemma LoRA x3 | 245-246 s | 258-261 s |
| gemma FFT x2 | 280-287 s | 294, 517 s |

At +2 min: both LoRA runtimes held H100 pairs exclusively. Two FFT pairs shared one H100
pair with two seats each. The other four FFT trainers were Pending on standing claims of
their own, with the reason `WaitingForCapacity: ... shared claims hold at most 2
workers`. They took seats as the first jobs finished. No worker ran more than two-way
shared.

What is left is GPU count. Two exclusive LoRA runtimes hold 4 of the 6 GPUs for 6 jobs,
so 6 FFT jobs queue behind one pair.

### Drill 5, 02:02 UTC: 6 H100 + 8 L4, seat limit 2

Result: **0 of 12**. All 12 succeeded. Every worker ran at once, no Pending.

| Group | Where | 3 steps done | Drill 4, H100 only |
|---|---|---|---|
| 0.6B LoRA x3 | runtime on L4 | 232-270 s | 95-97 s |
| 0.6B FFT x4 | two L4 pairs, 2 seats each | 195-237 s | 112-180 s |
| gemma LoRA x3 | trainer on L4, sampler on H100 | 559-568 s | 245 s |
| gemma FFT x2 | H100, 2 seats | 277-303 s | 280-287 s |

Binpack takes the smallest device that fits. The LoRA runtimes and the Qwen FFT pairs
went to the L4s while H100 GPUs sat idle. Only gemma FFT stayed on H100, because its
claim no longer fits an L4. The seat limit held and changed nothing here. Adding eight
L4 GPUs made the drill slower than the six H100s alone. The lever is a device-tier
preference for the LoRA runtimes. It is on hold.

### Drill 6, 02:23 UTC: 24 jobs on 6 H100, seat limit 2

Six jobs of each type. Result: **10 of 24**. All 24 succeeded. The last job finished at
+480 s.

| Group | 3 steps done | Note |
|---|---|---|
| 0.6B LoRA, one runtime | 122-149 s | six jobs on one runtime, all inside the window |
| 0.6B FFT | 106-215 s | queued two at a time behind one pair, 4 of 6 inside |
| gemma FFT | 196-326 s | same queue, after the Qwen FFT jobs |
| gemma LoRA, one runtime | 398-480 s | six jobs on one runtime, 75-80 s of sampling per step |

At +3 min: 9 workers running, 8 FFT trainers Pending with the limit in the reason. The
queue drained in about 5 minutes. No claim ever held more than two seats. Storage went
from 13% to 16% during the run.

Reading: six H100s do 24 jobs in 8 minutes with every job bounded, and the Qwen half
mostly inside 3 minutes. The gemma LoRA runtime needs more: a warm runtime, a second
sampler, or fewer jobs per runtime.

## Part 5: gemma FFT on an L4, the weight-update bug

### The probe, 00:42 UTC

L4s back in, one gemma LoRA job and one gemma FFT job alone on an idle fleet. Binpack
put the gemma LoRA trainer and sampler on L4s, and the gemma FFT sampler on an L4 with
its trainer on an H100.

| Job | First step | Warm steps | Result |
|---|---|---|---|
| gemma LoRA alone on L4 | 202 s | 76 s | 3 steps at 354 s |
| gemma FFT, sampler on L4 | 134 s | none | **failed at the first weight update** |

The sampler died with `CUDA out of memory. Tried to allocate 4.38 GiB` and 1.86 GiB free
of 22.03 GiB. A gemma step on an L4 is about three times the H100 step even alone.

### Where the memory goes

The sampler applies a sparse delta through vLLM's checkpoint patch loader. For each
patched checkpoint tensor the loader allocates a dense NaN-filled copy in bf16. Then its
patched `Tensor.copy_` computes `torch.where(torch.isnan(src), dst, src)`: a bool mask
and a second dense result. gemma-4-e2b's per-layer input embedding has 2.35 billion
elements. Its update needs 4.38 + 2.19 + 4.38 = 10.95 GiB at once, on top of 9.0 GiB of
weights and 1.4 GiB of CUDA graphs.

| vLLM share on the L4 | Free at update time | Result |
|---|---|---|
| 0.90, the default | 1.9 GiB | OOM at the 4.38 GiB dense copy |
| 0.68, headroom = 1x largest tensor + 0.5 GiB | about 5.7 GiB | OOM at the 2.19 GiB mask |
| 0.45, forced | about 10.7 GiB | 3 steps in 284 s, steps of 70 s, KV cache 0.38 GiB |

### The fix, commits d5361ed and 87417c2, verified at 01:36 UTC with images demo7-87417c2

- The sampler wakes only the weights when it takes the GPU. It applies the update, then
  wakes the KV cache. This is vLLM's documented order for RL weight updates.
- The runtime headroom is 2.5 times the largest checkpoint tensor plus 1 GiB, read from
  the safetensors headers. vLLM's share never drops below the resident weights plus 1 GiB.
  `OPEN_RL_WEIGHT_UPDATE_HEADROOM_GIB` overrides it.
- The estimator adds the same room to the FFT sampler's placement claim. gemma FFT's claim
  went from 11.7 to 23.6 GiB. It no longer fits a 24 GB device.

Verification on the idle full fleet: the scheduler put the gemma FFT sampler on an H100
with no hint. The log shows wake weights, update, wake KV cache on every step. The job
finished 3 steps in 225 s with warm steps of 39 to 44 s, the single-job H100 number. The
KV cache on the H100 is 48 GiB.

## Part 6: other probes

### gemma LoRA x3 with `openrl.sampler_replicas=2`, 02:15 UTC, 6 H100

3 steps in 233 to 235 s, against 245 s with one replica in drill 4. Warm steps fell from
about 82 s to 34 to 35 s, so two samplers halve the shared runtime's step. The first step
did not move: 164 to 166 s, of which 128 to 134 s is the sampler's own start with three
jobs waiting on it.

So a shared LoRA runtime on a cold start has a floor of about 165 s plus 35 s per further
step. Three steps inside 180 s needs the runtime already warm. A warm runtime would do the
same three jobs in about 105 s. That is the number for the warm-standby discussion. A
second replica costs a sampler GPU per runtime. On six H100s that starves the FFT
samplers, so it is a setting for the H200 fleet.

### The H200 pool, 23:41 UTC

`gcloud container node-pools create h200-8x` with one a3-ultragpu-8g in us-central1-b
needed two config fixes: a Hyperdisk Balanced boot disk and the fixed 32 local SSDs. It
then failed on capacity: eight attempts in 35 minutes, all `ZONE_RESOURCE_POOL_EXHAUSTED`.
A later a3-highgpu-4g pool failed in seconds on quota: `GPUS_PER_GPU_FAMILY ... Limit: 0.0
in region us-central1`. Both pools were deleted. The request text is in
`fleet-request-h200.md` in this directory.

## Where things stand, 02:40 UTC

Cluster: api-server and workers on `demo7-87417c2`, scheduler on `demo8-943eb38` with
`OPEN_RL_MAX_SEATS_PER_CLAIM=2`. The L4 nodes are labeled out of the drill. No workers
are running. Storage is at 16%.

Branch `feat/cold-start-measure`, unpushed:

```
fcf8b8d scripts: let demo Jobs carry openrl.* session tags
943eb38 scheduler: cap the seats a shared claim may hold
87417c2 sampler: wake the KV cache after the weight update, and size the update's room at its peak
ed0f629 scripts: give each demo client its own Hugging Face cache
d5361ed sampler: keep room for the weight update's dense staging copy
```

Open decisions:

- A device-tier preference for the LoRA runtimes, so binpack does not choose an L4 for
  gemma. On hold.
- Warm standby runtimes, so the first step does not pay the sampler start. On hold.
- A per-tier seat limit, one number for 24 GB devices and another for 80 GB. Later.
- The fleet request: quota for the H100 and H200 families, and capacity for one H200 node.

### Drill 7, 2026-10-19 07:18 UTC: 6 H100, seat limit 2, LoRA runtimes warm

Two one-step jobs brought the gemma and Qwen LoRA runtimes up first. The 12 jobs launched
five seconds after those finished. Result: **8 of 12**. All 12 succeeded.

| Group | 3 steps done | Drill 4, cold runtimes |
|---|---|---|
| gemma LoRA x3 | 88-97 s | 245-246 s |
| 0.6B LoRA x3 | 109-121 s | 95-97 s |
| 0.6B FFT x4 | 134-211 s | 112-180 s |
| gemma FFT x2 | 235-274 s | 280-287 s |

The gemma LoRA group is the whole story: first step 21 to 31 s instead of 170 s. The Qwen
LoRA group was slower than in drill 4 because its sampler was reaped in the five seconds
between the warm-up and the kickoff (the API server removes a runtime about two minutes
after its last session ends, and the Qwen warm-up job had finished first), so its jobs
paid a sampler start again. A real standby needs a keep-alive, not timing luck. The FFT
groups are unchanged: they still queue behind one H100 pair.

Added this morning: a Reason column on `kubectl get workloads` (commit aba8144, CRD
applied), `scripts/demo_rehearsal.sh` (idle check, volume trim, optional warm-up,
kickoff, watch), and the cell scripts for other regions (commit fc1ccbb). Quota: all 17
self-service requests for H200, B200 and H100 families were auto-denied within seconds;
see fleet-request-h200.md.

