# 017: Sampling pools

**Status:** Implemented, not merged.

A sampling pool serves one model's sampling requests. The pool keeps its pending
requests in a Redis stream. A dispatcher reads the stream, gives each request to
a sampler, and saves exactly one result for each request. This replaces the old
per-model Redis list, which lost a request when the sampler that read it
stopped.

This document covers cluster mode (`OPEN_RL_WORKER_MANAGER=scheduler`). Local
mode runs every pool as one in-process sampler.

## Current architecture on main

![Main today. The API server creates a sampler Workload, the scheduler creates its pod, and the pod pops a Redis list named after the model id.](017-before.svg)

- `save_weights_for_sampler` or `create_sampling_session` calls
  `ensure(model, "sampler")`. The API server creates a sampler Workload, and
  the scheduler creates its pod.
- The pod runs `python -m server.vllm_sampler --model-id <runtime>`, with vLLM
  in its own process. When vLLM is up, it writes
  `open_rl:sampler_ready:<runtime>`. `create_sampling_session` waits for that
  key for up to 300 seconds.
- `asample` pushes each request onto `open_rl:sampler_queue:<model_id>`. An FFT
  request names a weights path. A LoRA request names the adapter's working
  directory, `<tmp>/peft/<id>/<id>`.
- The sampler waits for one request with `BLPOP`, then pops the rest of the
  list. It groups the batch by weights path, loads new weights between groups,
  and writes each result to `open_rl:future:<request_id>`.
- An FFT sampler on its trainer's GPU takes one time-slicer turn around each
  batch it pops.

Sampling pools close four gaps in that design:

| On main | With sampling pools |
| --- | --- |
| A request leaves Redis when it is popped. A sampler that stops mid-batch loses the batch, and the client waits until its own timeout. | The dispatcher reads without removing, and deletes an entry only when its result is saved. |
| Every sampler of a model pops the same list, so nothing picks a replica by prefix cache or load. | A pool with two or more replicas routes through the llm-d EPP. |
| `create_sampling_session` blocks for up to 300 seconds while the sampler starts. | `asample` adds to the stream and returns. |
| A LoRA request reads the adapter's working directory, which the next save overwrites. | Each save is a frozen snapshot with its own name. |

## Two pool shapes

The number of replicas decides the shape. A replica is one process that runs
vLLM.

| Replicas | What runs | Sampler |
| --- | --- | --- |
| One | One pod with vLLM in its own process. That process also runs the dispatcher loop. | `vllm_sampler.Sampler` |
| Two or more | N `vllm serve` pods, plus one CPU pod with the dispatcher, Envoy, and the llm-d EPP. | `HttpSampler` |

A router spreads requests across replicas, so a pool with one replica has no
router. `sampler_replicas` comes from the job's metadata
(`openrl.sampler_replicas`, default 1). An FFT sampler that shares its
trainer's GPU always has one replica.

### One replica

![A pool with one replica. The scheduler creates one sampler pod, which runs the dispatcher loop and vLLM in one process. When it shares its trainer's GPU, it takes a time-slicer turn around each batch.](017-one-replica.svg)

### Two or more replicas

![A pool with two or more replicas. The scheduler creates replica pods with pool labels, Kubernetes creates the dispatcher pod from a Deployment, and the EPP finds replicas by those labels.](017-n-replicas.svg)

The scheduler labels each replica pod with `app.kubernetes.io/component=sampler`
and `app.kubernetes.io/instance=<pool>`. The EPP runs with
`--endpoint-selector` set to the same labels and watches those pods itself, so
it needs no registry and no InferencePool.

## What creates what

`save_weights_for_sampler` and `create_sampling_session` call
`ensure(model, "sampler")`. The first call for a pool creates it. Later calls
find the objects there and create nothing.

| Job | Workloads | Other objects |
| --- | --- | --- |
| Two or more replicas | `sampler_replicas` sampler Workloads, each on a GPU of its own | A dispatcher Deployment and a NetworkPolicy, both named after the pool |
| One replica with a GPU of its own | One sampler Workload | None |
| FFT that is not exclusive and has one trainer GPU | One sampler Workload, seated on the trainer's claim | None |

All LoRA jobs on one base model share one pool, `sampler-<base model>`. An FFT
job has a pool of its own. A request creates nothing. It is an entry in the
stream until it has a result.

## One request

1. `asample` checks that the weights exist, pins an FFT request to one
   revision, and adds the request to `open_rl:sampling:<pool>`. It returns at
   once and does not wait for a sampler.
2. The dispatcher reads the entry without removing it, and gives it to its
   sampler.
3. The dispatcher saves the result with `set_result`. One transaction writes
   `open_rl:future:<request_id>` and deletes the stream entry.
4. The client calls `retrieve_future`. The API server waits up to 60 seconds
   for the result, then tells the client to try again. The result expires after
   300 seconds.

Each pool has one dispatcher. The dispatcher Deployment uses the `Recreate`
strategy, and an in-process pool has one replica. A restarted dispatcher reads
the whole stream again, so an entry that a stopped sampler never answered is
sent again. Each request runs one or more times and gets exactly one result. A
request with no result after one hour fails.

## The dispatcher loop

`Dispatcher` in `server/sampling_dispatcher.py` runs a pass when a new entry
arrives, or after one second. Each pass does these steps:

1. Read new entries.
2. Fail entries older than one hour.
3. Change revision when no pending entry names the current one. The new
   revision is the one the oldest pending entry names. If that revision's
   directory is gone, fail its entries.
4. Call `sampler.ensure_weights(revision)`. If it returns false, send nothing
   this pass.
5. Call `sampler.generate(record)` for pending entries of the current revision,
   up to 256 at a time. A result of None sends the entry again later.

`HttpSampler` writes the pool's revision file and starts a weight load on each
replica that does not hold the revision. A replica is ready only when its
`/weight_info` equals the revision file, and the EPP routes only to ready
replicas. After each answer, `HttpSampler` checks the version of the replica
that answered, which Envoy names in `x-openrl-upstream`, and returns None on a
mismatch.

`vllm_sampler.Sampler` loads the revision into its own engine between batches,
so every answer in a batch comes from one revision. It calls the engine
directly.

Both samplers build their vLLM settings with `engine_args`, their sampling
params with `sample_params`, and their results with `sample_result`.

## Weights

An FFT pool serves one revision at a time. A revision is an immutable directory
of weights on the shared volume. The trainer writes `metadata.json` last. A
revision can be a delta that applies only to the saves its metadata lists. A
sampler that holds none of those saves does not load the delta.

A LoRA request names a frozen snapshot of its adapter instead of a revision.
Each save is a new snapshot with its own name, so a LoRA pool never changes
revision and never loads weights.

## Time slicing

Only an in-process sampler on its trainer's GPU takes turns. For each batch it
acquires the GPU from the node's time slicer and wakes vLLM. Then it reads the
stream again, so requests that arrived during the wait join the batch. It
answers the batch, puts vLLM to sleep, and releases the GPU. The time slicer
then moves the process's GPU memory to the host and gives the GPU to the
trainer, which takes its own turns the same way.

The process that uses the GPU is the only one that talks to the time slicer.
The dispatcher pod and the router never know that a GPU is shared, and the
slicer protocol does not change.

The process exits with an error, and Kubernetes restarts the pod, in each of
these cases:

- The vLLM engine died.
- A weight update stopped partway.
- The time slicer could not move the process's memory off the GPU. The
  process still holds the GPU, so exiting is how it gives the GPU back.

## Teardown

The client sends a heartbeat every 10 seconds, and the session key expires
after 120 seconds. Every 30 seconds the reaper calls `release_owner` for each
owner that no live session uses. It deletes the pool's NetworkPolicy and
dispatcher Deployment, if the pool has them, and then every Workload of the
owner. The scheduler's finalizer deletes each pod, then frees its claim.

An FFT job is its own owner. Shared LoRA workers belong to their base model, so
the reaper releases them only when no session uses any job on that base model.
No worker or pool has an idle timeout.

## Seams

| Seam | Contract | To swap it |
| --- | --- | --- |
| Router | The dispatcher posts `/inference/v1/generate` to one URL. Replicas are found by pod labels and are ready by `/weight_info`. | Run another router in the dispatcher pod, after the version check stops using `x-openrl-upstream`. |
| Sampler | `ensure_weights(revision)` and `generate(record)` | Write one class, for example for SGLang. |
| Engine settings | `engine_args` | Change one function. |
| Placement | `ensure` and `release` on a worker manager | Write another worker manager. |
| GPU sharing | The in-process sampler runs its batches through `TimeSlicedDispatcher` | Use the plain `Dispatcher`. |
| Store | Accept, read pending, and `set_result` | Write one class. |

## Open items

- `HttpSampler` finds the replica that answered from `x-openrl-upstream`, which
  only Envoy adds. Another router needs the replica to report its own version.
- No unit test runs `vllm_sampler.Sampler`, because vLLM needs a GPU. It is
  tested on GKE.
