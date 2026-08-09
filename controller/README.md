# The GPU scheduler

A worker says how much accelerator memory it needs and who it can share weights
with. The scheduler decides which bundle of accelerators it lands on and who it
takes turns with. That is the whole contract.

```yaml
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorker
metadata:
  name: adapter-a
spec:
  role: trainer                # which node pools may host it
  modelId: adapter-a           # its identity everywhere: pod name, time-slice job
  memory: 6Gi                  # total accelerator memory, across however many devices
  cohort: Qwen/Qwen3-0.6B      # optional: who it may be resident alongside
```

Everything else is derived and reported back in `status`: the device count, the
per-device split, the claim, the node, and what a context switch is expected to
cost.

## The model, in one sentence

**A claim is a bundle of accelerators; cohorts within a claim take turns;
members of a cohort run together.**

- **Memory sums *within* a cohort.** Several lora adapters over one frozen base
  model name that base model, are resident at once, and their memory genuinely
  adds up against the device.
- **Cohorts *take turns*.** Anything else is a switch: the node-local timeslicer
  parks the outgoing cohort in host RAM and restores the incoming one.
- **The cohort is an opaque string the scheduler never interprets**, only
  compares. A worker that names none is a cohort of one. That is deliberate:
  only the caller knows what its workers share, so a new way to share is a
  caller-side decision, not a new release of this binary.
- **`role` selects nodes, never claims.** On a node labelled for both, a trainer
  and a sampler share one accelerator by turns. Refusing to mix would only
  strand hardware that time-slicing can already share.
- **There is no sharding.** The model is laid out layer by layer over whatever
  devices it is given, so the device count is plain ceiling division and any
  count will do. A second GPU adds capacity, not speed, which is why the count
  is always derived and never requested.

## Layout

| path | what it is |
| --- | --- |
| `api/v1alpha1` | the CRD: the request, and what was decided about it |
| `internal/placement` | the decision. Pure functions, no Kubernetes imports |
| `internal/controller` | the part that reads and writes Kubernetes objects |
| `internal/sim` | the scheduler against a cluster that does not exist |
| `scenarios/` | hand-editable clusters and workloads for the simulator |

## Kicking the tires

Four rungs, cheapest first. Each one is worth exhausting before climbing.

**1. The simulator — no cluster at all.** It drives `internal/placement`
directly, so what it shows is the real decision, not a model of one. What it
adds is time: workloads arrive, cohorts take turns, switches cost what the
measured cost model says they cost, and workloads finish and give their seats
back.

```
make sim                                     # scenarios/one-l4.yaml
make sim SCENARIO=scenarios/box.yaml TIMELINE=-timeline
```

```
TIMELINE
  0s     place  fft-qwen cut claim-fft-qwen (1x10Gi/dev) as cohort "fft-qwen"
  0s     place  lora-alpha joined claim-fft-qwen (1x6Gi/dev) as cohort "qwen3-0-6b"
  1m0s   turn   claim-fft-qwen -> cohort "qwen3-0-6b" (switch 3.5s)
  ...
CLAIMS
  NAME            NODE  GPUS  COHORTS  BUSY    SWITCHING  OVERHEAD
  claim-fft-qwen  l4-0  1     2        11m16s  44s        6%
```

Write a scenario for the cluster you are about to buy, or the burst you are
about to send, and see what it costs before it costs anything.

**2. kind, with synthetic ResourceSlices.** Exercises the Kubernetes half — CRD,
claims, pods, status — without real accelerators.

**3. One real box.** `scenarios/box.yaml` is the dev box: one node, 2× L4 24Gi,
94Gi of host RAM.

**4. The big cluster.**

## The switch-cost model

Fitted to two measured points on 2× RTX PRO 6000 Blackwell (driver 580.159.04)
under `cuda-checkpoint`:

| parked | checkpoint | restore | total |
| --- | --- | --- | --- |
| ~1GiB over 1 device | 1.849s | 0.521s | 2.37s |
| ~128GiB over 2 devices | 21.523s | 8.648s | 30.17s |

Two points only fit a line, and it is not claimed to hold outside that range. It
exists so an operator can see, before setting `openrl.io/max-residents`, that the
same knob costs 2s on a 0.5B model and 30s on an 8B one.

Host memory, not that label, is the harder bound: `cuda-checkpoint` does not
spill to disk, it parks a process's device memory in that process's own host
address space. Exceeding device memory degrades; exceeding host memory
OOM-kills the node.

## Node labels

Policy the operator sets. They express intent, never hardware — the DRA driver
reports what the devices actually are.

| label | meaning |
| --- | --- |
| `openrl.io/enabled=true` | the scheduler may use this node at all |
| `openrl.io/trainer=true` | trainers may land here |
| `openrl.io/sampler=true` | samplers may land here |
| `openrl.io/max-residents=N` | how many workers may share one claim here (default 1: no sharing) |

## Deploying

```
kubectl apply -k ../k8s/deploy/scheduler
```

Applying it changes nothing about a running cluster: the scheduler only acts on
OpenRLWorker objects, and nothing creates those yet.

## What is deliberately not here

Priority, preemption, quota, gang scheduling, and real sharding are all
additive: they need new fields or new objects, not different ones. What would be
expensive to change later, and is therefore frozen now, is the runtime protocol
(baked into every image), the cohort key, and the split between memory in the
spec and device count in the status.
