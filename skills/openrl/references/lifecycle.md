# Workers and their lifecycle

Numbers are from a GKE cluster with L4 and H100 nodes on 2026-10-12, with the worker
image already on the nodes and the model already in the shared cache. A cold node adds
an image pull (the worker image is several GiB) and a model download.

## Who runs what

| Your call | What starts | Shared with |
|---|---|---|
| `create_lora_training_client` (LoRA, default) | one trainer for the base model, Workload `lora-<base>-<hash>-0-trainer` | every other LoRA job on the same base model; each job is one adapter in it |
| first `sample` or `save_weights_and_get_sampling_client`, or already at `create_lora_training_client` with `openrl.recipe=rl` | one sampler, `lora-<base>-<hash>-0-sampler` (plus `-1`, `-2`... up to the largest `openrl.sampler_replicas` any job asked for), behind a CPU dispatcher pod | same |
| `create_lora_training_client` with `openrl.fine_tuning_type=full`, or `openrl.exclusive=true`, or `openrl.trainer_gpus>1` | a trainer of its own, `fft-<model_id>-trainer`, and later `fft-<model_id>-sampler` | nothing; full fine-tuning workers take turns on a GPU with other jobs' workers unless exclusive |

The workers that serve one job appear together in the job's model view,
`GET /api/v1/openrl/models/{model_id}`.

## Worker states

Each worker in the model view has a `state`, the same five values on every server, and a
`detail` in the scheduler's own words (Workload phase and reason, node). Branch on `state`;
show `detail` to a human. `kubectl -n openrl-system get workloads` shows the same facts
`detail` quotes. The pods behind them are the scheduler's; the view reads only whether a
running sampler's pod is Ready, and names the reason when its container is waiting, for
example `CrashLoopBackOff`. A sampler is also `starting` while its pool's dispatcher has no
pod available, since no sample is sent without it.

| State | Meaning | What `detail` says | Normal duration |
|---|---|---|---|
| `waiting_for_placement` | no accelerator assigned yet | `Pending (NoCapacity: ...)`: nothing in the cluster fits; `Pending (WaitingForCapacity)`: something fits but is busy | seconds; longer only when the cluster is full |
| `starting` | placed on a node; pulling the image, mounting volumes, or loading the model | `Placing (PodCreated) on <node>` is the image pull and volume mount; `Running ..., loading the model` is the trainer loading the base model or vLLM starting | under a minute on a warm node; 5 to 10 minutes on a cold node. Trainer: `ready` about 2 minutes after `create_model`; sampler: 3 to 4 minutes more while vLLM starts |
| `ready` | serving requests: the worker has its model loaded and is draining its queue | node name | |
| `failed` | the process exited with an error, or the request cannot be satisfied | the scheduler's reason, for example `Failed (PodFailed)`. A container that cannot pull its image or crash-loops is not reported as failed yet; it stays `starting` or `ready` past the normal duration (a crash-looping sampler is `starting` with the reason in `detail`), see the runbook | |
| `stopped` | exited normally, or being released | `being released; ...` while the reaper or `delete_model` tears it down | |

While a call blocks, ask the server about the model from another shell; the examples
print the model id and this command right after `create_model`:

```
curl -s $OPENRL/api/v1/openrl/models/<model_id> | jq .summary
"trainer ready: Running (PodCreated) on gke-...-lvw8, 201s; sampler starting: Running on gke-...-2208, loading the model, 120s"
```

So, from a laptop against a warm cluster: about 2 minutes to the first loss line and about
4 more minutes to the first sample. A second run on the same base model within two minutes
of the first finds both workers up and starts in seconds.

## Bigger models: when one GPU is not enough

The server sizes every worker from the model's parameter count before it asks the
scheduler for a device, and refuses with `Pending (NoCapacity: ...)` when nothing fits. Use
the same arithmetic before you choose a model or a setting:

| Worker | Device memory it asks for | Example: 8B | Example: 27B |
|---|---|---|---|
| LoRA trainer (default) | 2 bytes per parameter + 4 GiB | about 19 GiB | about 54 GiB |
| full fine-tuning trainer | 8 bytes per parameter + 4 GiB | about 65 GiB | about 205 GiB |
| sampler (vLLM, always one device) | 2 bytes per parameter + KV cache for eight max-length requests + about 2 GiB; LoRA adds about 1 GiB plus 0.25 bytes per parameter | about 30 GiB | about 62 GiB |

A model the server does not know is sized as 8B, so the request is never too small.
On a cluster of L4s (22 GiB per device) that puts the ceiling for the default trainer at
about 8B. H100s (80 GiB) take a 27B LoRA trainer and sampler on one device each, and an
8B full fine-tuning trainer; a 27B full fine-tuning trainer fits no single device, and
full fine-tuning cannot span devices.

**Decision order for a bigger model:**

1. **Prefer a bigger device.** `openrl.exclusive=true` gives the job its own trainer and
   sampler. If the cluster has a larger pool, the scheduler places them where they fit;
   nothing else changes. Ask the platform owner which pools exist; the endpoint does not
   list them yet.
2. **Then spread the trainer over several GPUs.** When no single device holds the
   trainer, set `openrl.trainer_backend=automodel` and `openrl.trainer_gpus=N` (2 to 8).
   The Automodel trainer shards the model with FSDP2 across `N` GPUs on one node, so each
   device needs only its share: a 27B LoRA trainer over four GPUs asks for about 14 GiB
   each, which fits an L4. Multi-GPU is LoRA only, needs the Kubernetes deployment, and
   makes the job exclusive. For long sequences, `openrl.trainer_cp=C` makes `C` of those
   GPUs share each sequence as context parallelism, with the rest data parallel; `C` must
   divide `N`.
3. **The sampler still has to fit one device.** Nothing spreads it. A model whose bf16
   weights plus KV cache exceed the largest device cannot be sampled, so it cannot do RL
   or sample after SFT on that cluster.

What changes when you choose Automodel:

- The trainer runs a different image, the one the platform owner set as
  `OPEN_RL_AUTOMODEL_IMAGE`. The endpoint does not report which, and a deployment that
  never set it runs `ghcr.io/gke-labs/open-rl/automodel:latest`. A first start pulls that
  image (several GiB) on top of the usual durations.
- The job gets a trainer of its own, keyed `automodel-<base-model>`, so it never shares
  with PyTorch LoRA jobs on the same base model and starts cold every time: expect the
  5 to 10 minute cold-node figure, not the warm one.
- The model view is the same; the trainer is `starting` until all `N` ranks have loaded
  their shard.
- `openrl.trainer_backend` also accepts an image reference, for a trainer you built
  yourself; the job then runs on that image with the same module on its `PATH`.

## Sessions and release

- The SDK heartbeats its session every 10 seconds. The server keeps a session alive for
  120 seconds after the last heartbeat (`limits.session_ttl_seconds` in capabilities).
- A reaper runs every 30 seconds and releases every worker whose sessions are all gone.
  Observed: workers gone 2 minutes 7 seconds after the client exited.
- Release deletes the Workloads and pods. Adapter weights that were only in the trainer's
  memory are gone with it; checkpoints written by `save_state` or `save_weights_for_sampler`
  stay on the shared volume and can be restored with `create_training_client_from_state`.
- A heartbeat gap longer than the TTL has the same effect as exiting, including a
  port-forward that is down for two minutes. On servers up to v0.0.6 the client is not told;
  it keeps polling `try_again`. The model view then says
  `no live session and no workers; the model's workers were released`.
- `POST /api/v1/delete_model` releases right away, or frees the adapter in a shared runtime.

## Sharing and speed

- LoRA jobs on one base model take turns in one trainer, so a step takes longer when several
  run at once; their sampling overlaps with training.
- Full fine-tuning jobs each get a trainer; on the FFT release they time-slice a GPU, parking
  state in host memory between turns, so concurrent FFT jobs slow each other down.
- A sampler that is still loading a model is `starting` with a `detail` that ends in
  `loading the model`; requests queue until it is `ready`.
