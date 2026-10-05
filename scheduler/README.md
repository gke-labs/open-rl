# The GPU scheduler

A workload says how much accelerator memory it needs and brings its own pod
template. The scheduler decides which GPU it lands on and who it takes turns
with. That is the whole contract.

```yaml
apiVersion: openrl.io/v1alpha1
kind: Workload
metadata:
  name: fft-job-a-trainer
spec:
  role: trainer                # which node pools may host it
  trainingKind: fft            # informational; placement reads exclusive
  exclusive: false             # true keeps it alone on its GPU
  modelID: job-a               # its identity everywhere
  ownerID: Qwen/Qwen3-0.6B     # optional: the unit of fairness it belongs to
  accelerator:
    memory: 28Gi               # peak accelerator memory, from the estimator
  template:                    # the complete worker pod, inline
    spec:
      containers:
        - name: worker
          image: ghcr.io/gke-labs/open-rl/worker:latest
```

Everything else — device, tier, claim, node — is derived and reported back in
`status`.

## Placement policy

The default `binpack` strategy shares existing claims between workers that are
not exclusive, which participate in time slicing. Exclusive workers, every
LoRA worker and any model that asked to share nothing, get claims of their own
and wait for a free GPU. Multiple LoRA
adapters can still reuse the same worker; that happens before placement.

`spread` tries a new claim first and falls back to sharing only between
workers that are not exclusive. Both policies read `exclusive`, never
`trainingKind`. An omitted `exclusive` is exclusive.
The LoRA release sets no placement timeout.

DRA allocates the device from the claim's ordered alternatives. A ClaimLedger
records each worker's seat using compare-and-swap, so concurrent reconciles
cannot double-book. Node role labels select eligible pools; hardware capacity
comes from ResourceSlices.

## Layout

| path | what it is |
| --- | --- |
| `api/v1alpha1` | the CRDs: Workload (the request) and ClaimLedger (the seat ledger) |
| `internal/placement` | the decision. Pure functions, no Kubernetes imports |
| `internal/controller` | the part that reads and writes Kubernetes objects |
| `docs/design.md` | the design |

## Try it

The behaviors live in `internal/placement/behavior_test.go`: workers arriving
and leaving, with the estimator's real tier figures on the hardware we run,
played through the same decisions the controller makes.

```
go test ./...
make stress    # the placement storm; slow, not part of go test ./...
```

For the pipeline — real API server, real kube-scheduler, real DRA — there is
a kind smoke test that needs no hardware (the DRA example driver publishes
fake GPUs):

```
make smoke                                # kind + fake GPUs
USE_EXISTING_CLUSTER=1 DEVICE_CLASS=gpu.nvidia.com ./hack/kind-smoke.sh   # real GPUs
```

The controller only ever reads ResourceSlices and node labels, so fake and
real devices exercise the identical path; only the two env values differ.

## Deploy

For the complete LoRA stack, use the [LoRA deployment overlays](../docs/setup/lora-dra.md).
The standalone scheduler base includes its service account, CRDs, and RBAC:

```
kubectl apply --server-side -k scheduler/deploy/base
kubectl label node <node> openrl.io/enabled=true openrl.io/trainer=true
```

Applying it changes nothing about a running cluster: the scheduler only acts
on Workload objects. Node labels are policy, never hardware — the DRA
driver's ResourceSlices say what devices actually exist.

Labeling a node opts its GPUs in **exclusively**: the scheduler assumes its
own claims are the only GPU consumers there, so other GPU workloads on an
enabled node will collide with it. Give OpenRL whole nodes.

## Session cleanup

The tinker client heartbeats its session every ten seconds. The API server
records which sessions use which owners, where an owner is the trainer and
sampler pair behind a Workload's `ownerID`. Every 30 seconds it drops
sessions silent for 120 seconds and deletes the workloads of any owner no
live session uses anymore. An FFT job has its own owner, so its pair goes
when the job's session does. LoRA jobs on one base model share an owner, so
the pair stays until the last of their sessions is gone. The registry lives
in Redis, so an API server restart keeps it. Run one API server replica. The lock
that keeps a session from attaching to an owner mid-teardown is in-process.

## Everything else

Assumptions and caveats, the estimator, worker identity, claim lifecycle,
and the future optimizations all live in
[`docs/design.md`](docs/design.md); a file-by-file tour with a suggested
reading order is [`docs/layout.md`](docs/layout.md). If the code and any
document disagree, the code is right.
