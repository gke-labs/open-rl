# OpenRL on GKE TPU nodes

How to run OpenRL trainer and sampler workers on Cloud TPU nodes in a GKE
cluster deployed with the scheduler stack (see
[GKE Setup Guide](gke-setup.md) for the GPU version of the same cluster). The
control plane is the one every overlay installs: API server, scheduler, Redis
and a shared Filestore volume. What changes is the node pool, the DRA driver,
two worker images, and the `k8s/deploy/tpu` overlay that names them.

What runs on TPU:

| Job | Runs on TPU | Notes |
| --- | --- | --- |
| LoRA, shared or exclusive | yes | Several LoRA jobs on one base model share a trainer and a sampler, as on GPU. |
| Full fine-tuning | with `openrl.exclusive=true` only | The trainer and the sampler each take a node of their own; the sampler reloads the full checkpoint after every step. |
| `openrl.trainer_backend=automodel` | no | GPU only. |

Qwen3-0.6B (LoRA and FFT) and Gemma 4 E2B (LoRA) have run 100-step GSM8K
jobs on v6e and matched the GPU learning curves; Gemma 4 E2B FFT ran on v7x,
since its full model does not fit one v6e chip. Each worker uses one chip. The
design is described in
[Design Doc 018](../designs/018-tpu-support.md); this page is the operator's
view.

## What the cluster needs

| Component | Used here | Why |
| --- | --- | --- |
| GKE version | `1.36` rapid channel | DRA in `resource.k8s.io/v1`, which the TPU DRA driver and the OpenRL scheduler use. |
| CPU node pool | `2 x n2-standard-4` | API server, scheduler, Redis, client jobs. |
| TPU node pool | `ct6e-standard-4t` (v6e), one node per worker | A LoRA job needs two nodes (trainer and sampler); an FFT job needs two more. |
| TPU DRA driver | [kubernetes-sigs/dra-driver-google-tpu](https://github.com/kubernetes-sigs/dra-driver-google-tpu), staging image | Publishes the chips as DRA devices. The scheduler places TPU workers through it, not through GKE's `google.com/tpu` device plugin. |
| Shared storage | `1Ti standard-rwx` Filestore PVC | Adapter snapshots, FFT checkpoints and the Hugging Face cache. Same as the GPU setup. |
| Images | one TPU trainer image, one TPU sampler image | Built from `src/server/Dockerfile.tpu` and `src/server/Dockerfile.tpu_sampler`. They are not published. |

Each TPU worker takes a whole node. The driver prepares no partial-node claim,
so the scheduler asks for every chip on the node and the worker uses the first
one. On a four-chip `ct6e-standard-4t` host three chips idle; a one-chip host
type such as `ct6e-standard-1t`, where the zone has it, avoids that.

TPU generations the scheduler sizes today: v6e (32 GiB HBM per chip) and v7x
(192 GiB). It keys the per-chip memory on the driver's `tpuGen` attribute. The
table also lists `v5e` and `v5p`, but the driver publishes v5e nodes as
`v5litepod`, so they are not placed, and v5p has not been tried.

## 1. Create the TPU node pool

Create the cluster as in the [GKE Setup Guide](gke-setup.md), with the
Filestore CSI add-on and `--cluster-ipv4-cidr=/19` on the default network.
Then add the TPU pool. v6e capacity usually needs a reservation; without the
reservation flags the create fails on the `TPUS_PER_TPU_FAMILY` quota.

```bash
gcloud container node-pools create tpu-v6e --project <project> --cluster <cluster> --zone <zone> \
  --machine-type ct6e-standard-4t --num-nodes 2 --disk-size 200 \
  --node-labels=cloud.google.com/gke-tpu-dra-driver=true,openrl.io/enabled=true \
  --reservation-affinity=specific --reservation=<reservation>
```

Two labels do the work:

- `cloud.google.com/gke-tpu-dra-driver=true` tells GKE not to run its TPU
  device plugin on these nodes and lets the DRA driver's kubelet plugin
  schedule there. A pool serves `google.com/tpu` requests or DRA claims, never
  both.
- `openrl.io/enabled=true` enrols the node with the OpenRL scheduler. Put it
  in `--node-labels`, not on the nodes by hand: a node recreated after a scale
  to zero comes back with the pool's labels only, and the scheduler stops
  seeing it.

Do not pass `--tpu-topology` with more than one node: GKE then treats the pool
as one multi-host slice and rejects it.

Install the driver. The kubelet plugin needs the TPU toleration and an affinity
to the labelled nodes, since the chart ships without them:

```bash
git clone https://github.com/kubernetes-sigs/dra-driver-google-tpu
helm upgrade -i --create-namespace -n dra-driver-google-tpu dra-driver-google-tpu \
  dra-driver-google-tpu/deployments/helm/dra-driver-google-tpu \
  --set image.repository=us-central1-docker.pkg.dev/k8s-staging-images/dra-driver-google/dra-driver-google-tpu \
  --set image.tag=<tag> \
  --set kubeletPlugin.priorityClassName="" \
  --set 'kubeletPlugin.tolerations[0].key=google.com/tpu' \
  --set 'kubeletPlugin.tolerations[0].operator=Exists' \
  --set 'kubeletPlugin.tolerations[0].effect=NoSchedule' \
  --set 'kubeletPlugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key=cloud.google.com/gke-tpu-dra-driver' \
  --set 'kubeletPlugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator=Exists'
```

The driver is ready when every TPU node has a ResourceSlice:

```bash
kubectl get resourceslices
kubectl get resourceslices -o jsonpath='{range .items[*]}{.spec.nodeName}{" "}{.spec.devices[*].attributes.tpuGen.string}{"\n"}{end}'
```

The slice must show `tpuGen: v6e` (or `v7x`) on each device, because that is
what the scheduler sizes the chip by. It also installs the `tpu.google.com`
DeviceClass the scheduler requests; nothing in `k8s/` adds one.

## 2. Build the images

The TPU images are built on every pull request but not published. Build them
into your registry with Cloud Build:

```bash
make cloud-build-tpu-trainer cloud-build-tpu-sampler GCP_PROJECT=<project>
```

The trainer image installs the `tpu` extra (`torch-tpu` and a CPU `torch` from
PyPI); the sampler image installs the `tpu-sampler` extra (`vllm-torchtpu`,
vLLM's TPU platform on the same `torch-tpu`). Both are plain Ubuntu images
with no CUDA.

The released `vllm-torchtpu` serves full fine-tuning but not LoRA: its
multi-LoRA support is still in open upstream pull requests (vllm-torchtpu
#2376, #2359, #2378 and #2360). Until a release carries them, a sampler image
for LoRA jobs has to be built from a `vllm-torchtpu` checkout of `main` with
those changes applied, with OpenRL's `tpu-sampler` extra installed against that
build instead of the PyPI pin. FFT jobs run on the image above as is.

## 3. Deploy

The `k8s/deploy/tpu` overlay is `k8s/deploy/fft` plus the two TPU image
variables on the API server. It serves TPU-only clusters and clusters that
also have GPU nodes; the GPU time slicer, snapshot agent and DCGM DaemonSets
select `nvidia.com/gpu.present` and run no pods on TPU nodes.

Write a small overlay of your own that points the image placeholders at your
builds and carries the worker knobs:

```yaml
# my-tpu/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: openrl-system
resources:
  - ../open-rl/k8s/deploy/tpu
images:
  - name: TPU-TRAINER-IMAGE
    newName: <registry>/open-rl-tpu-trainer
    newTag: <tag>
  - name: TPU-SAMPLER-IMAGE
    newName: <registry>/open-rl-tpu-sampler
    newTag: <tag>
generatorOptions:
  disableNameSuffixHash: true
configMapGenerator:
  - name: open-rl-tpu-worker-env
    literals:
      - VLLM_MAX_MODEL_LEN=4096
      - VLLM_MAX_NUM_SEQS=64
patches:
  - patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: open-rl-api-server
      spec:
        template:
          spec:
            containers:
              - name: api-server
                env:
                  - name: OPEN_RL_TPU_WORKER_ENV_CONFIGMAP
                    value: open-rl-tpu-worker-env
```

```bash
kustomize build my-tpu | kubectl apply --server-side -f -
kubectl -n openrl-system wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
kubectl -n openrl-system rollout status deploy/open-rl-scheduler deploy/open-rl-api-server
```

Server-side apply is required: the Workload CRD exceeds the client-side apply
annotation limit. The API server, scheduler and GPU worker images come from the
base overlay's defaults (`ghcr.io/gke-labs/open-rl/...:latest`); pin them with
`images:` entries too if you want a fixed version.

### Worker knobs

Every entry of the ConfigMap named by `OPEN_RL_TPU_WORKER_ENV_CONFIGMAP`
becomes an environment variable on each TPU worker pod. Pods read it when they
are created, so change it before a run or delete the Workloads so the next run
recreates them. The entries that matter on TPU:

| Entry | Default on TPU | What it does |
| --- | --- | --- |
| `VLLM_MAX_MODEL_LEN` | `8192` | The sampler compiles a ladder of shapes up to this length at start. Shorter is faster to start and leaves more HBM for the KV cache. |
| `VLLM_MAX_NUM_SEQS` | `64` | Sequences the sampler batches together. The GPU default scales with detected device memory; on TPU nothing is detected, so set it by hand. |
| `OPEN_RL_TRAIN_TOKEN_BUDGET` | `auto` = `16384` | Padded tokens per trainer pass. On TPU `auto` does not probe free memory and uses the 16384 ceiling. The compiled graph materialises activations for the whole padded batch, so a budget that fits on GPU can run out of HBM here. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | The HBM share vLLM reserves. The name says GPU, but the TPU platform reads it too. |

Knobs that ran, all with one chip per worker:

| Model | Chip | `VLLM_MAX_MODEL_LEN` | `VLLM_MAX_NUM_SEQS` | `OPEN_RL_TRAIN_TOKEN_BUDGET` | Other |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B, LoRA and FFT | v6e | 4096 | 64 | 16384 | |
| Gemma 4 E2B, LoRA | v6e | 2048 | 16 | 8192 | 262k-token vocabulary: 16384 runs out of HBM. |
| Gemma 4 E2B, FFT | v7x | 2048 | 16 | 8192 | `VLLM_GPU_MEMORY_UTILIZATION=0.75`; the full reload stages the embedding table per layer and runs out at 0.90. |

## 4. Run a job

A model asks for TPU with the accelerator preference settings, in the model's
`user_metadata` or in `TINKER_TAGS` (a tag cannot hold a comma, so entries are
separated with `|` there):

```bash
export TINKER_TAGS="openrl.trainer_accel_prefs=tpu,openrl.sampler_accel_prefs=tpu"
```

Full fine-tuning adds `openrl.exclusive=true`; the API server refuses a TPU
FFT model without it. Only the first entry of each list is used today, and a
model with no preference runs on GPU. The settings are fixed when the model is
created. See [Choosing an accelerator](../configuration.md#choosing-an-accelerator).

The e2e script sets both tags from one argument:

```bash
uv run --extra cpu python scripts/run_training_e2e.py scenario=lora-gsm8k-rl \
  base_url=http://127.0.0.1:9003 accelerator=tpu steps=10
```

and `k8s/eval/e2e-client-job.yaml` runs it in the cluster with the TPU
toleration already set. The first `create_model` creates a trainer Workload,
the first sampling session a sampler Workload:

```bash
kubectl -n openrl-system get workloads,resourceclaims,pods -o wide
```

A Workload whose claim cannot be built stays `Pending` with a reason:
`NoCapacity: no enabled node has TPU devices` when no enrolled node has a
ResourceSlice from the TPU driver, `WaitingForCapacity` when every TPU node is
taken. The release overlays set no placement timeout, so the Workload waits
until a node appears or the client gives up.

### What to expect

Numbers from Qwen3-0.6B on v6e, GSM8K, one chip per worker:

| | Start-up | Step |
| --- | --- | --- |
| Trainer | 2-3 min (image on the node, model in the shared cache); the first few batches compile | |
| FFT sampler, vLLM in the sampler's process | about 3 min, most of it compiling the shape ladder | 35 s at 16x8 (includes the checkpoint write and full reload) |
| LoRA sampler, `vllm serve` replica | about 6-7 min; the LoRA kernels compile per shape too | 39 s at 8x8, three jobs sharing the runtime |

Sampling requests wait in the pool until a sampler answers, up to
`OPEN_RL_SAMPLE_DEADLINE_SECONDS` (3600 by default), so the first step absorbs
the sampler start without failing. A new `(batch, length)` shape costs the
trainer a compile of about a minute; shapes are bucketed to powers of two, so
the set is small and the cost is paid once per shape.

Host memory: the scheduler requests 23 GiB plus 2 bytes per parameter for a
TPU trainer and 7 GiB plus 2 bytes per parameter for a sampler (14 bytes per
parameter for an FFT trainer), and the limit equals the request. The trainer's
resident memory grows about 0.1 GiB per step; the request covers runs of about
100 steps on the models above. Longer runs on a bigger host need the trainer
overhead raised in `src/server/estimator.py`.

## 5. TPU v7 (tpu7x)

v7 runs through the same DRA path, with three differences:

- **Driver.** The upstream driver does not list the v7x PCI id and publishes
  nothing on a `tpu7x` node. Use the fork `hodaaaaaaaaaa/dra-driver-google-tpu`,
  branch `v7x`: build its image, push it to a registry the cluster can pull
  from, and pass it as `image.repository` and `image.tag` in the helm command
  above. The fork publishes one device per chip (a v7x chip is two PCI
  functions, which it folds by PCI slot), so the HBM table's `v7x: 192Gi` row
  applies per chip.
- **Pool.** `tpu7x-standard-1t` (one chip per node) with the same two labels
  and the v7 reservation:

  ```bash
  gcloud container node-pools create tpu7x-1t --project <project> --cluster <cluster> --zone <zone> \
    --machine-type tpu7x-standard-1t --num-nodes 2 --disk-size 200 \
    --node-labels=cloud.google.com/gke-tpu-dra-driver=true,openrl.io/enabled=true \
    --reservation-affinity=specific --reservation=<reservation> \
    --no-enable-autorepair
  ```

- **Auto-repair off.** GKE 1.37 repairs a `tpu7x` node that advertises no
  `google.com/tpu` allocatable about every 15 minutes
  (`AUTO_REPAIR_TPU_UNALLOCATABLE`), which is every DRA node. The
  `gke-tpu-dra-driver` label suppresses that on v6e pools but not on tpu7x
  yet. Keep auto-repair off on the pool until it does.

## 6. GPU and TPU nodes in one cluster

The `tpu` overlay also serves a cluster with both kinds of node. Add a GPU pool
with the NVIDIA DRA driver as in
[OpenRL on an existing DRA cluster](lora-dra.md), enrol both pools with
`openrl.io/enabled=true`, and deploy once. Then:

- A model whose preference lists `tpu` first runs its workers on TPU nodes;
  every other model runs on GPU. The scheduler never places a TPU request on a
  GPU pool or the reverse.
- TPU and GPU LoRA jobs on the same base model get separate shared runtimes
  (the TPU one is named `tpu-<base model>`), so a GPU trainer never serves an
  adapter that a TPU sampler reads.
- A preference that no node can serve is not rejected at `create_model`: the
  Workload waits for placement, since the TPU pool may be scaled to zero or
  still provisioning. The client's own timeout ends the job.

A zone with v6e capacity may have no GPU machine types (us-central2-b has
none); a regional cluster can hold node pools in other zones of the region.

## 7. Limitations

- **Whole node per worker.** The DRA driver grants every chip on the node or
  none, and the worker uses one. Multi-chip and multi-host workers are not
  built; an FFT model must fit one chip (about 1.7B parameters on v6e).
- **HBM is a table.** The driver publishes no memory capacity, so the
  scheduler sizes chips by `tpuGen`: v6e and v7x work, v5e is published under
  a different key and v5p is untried.
- **First preference only.** Later entries of an accelerator preference list
  are not used, and there is no "whichever is free" placement.
- **FFT is exclusive and reloads in full.** No time slicer exists for TPU, so an
  FFT trainer and sampler never share a chip, and the sampler reloads the whole
  checkpoint after every step: sparse deltas cannot patch the TPU platform's
  transposed weights. `openrl.weight_sync` is forced to `full`.
- **Scheduler mode only.** TPU workers run through the Kubernetes scheduler.
  The local worker manager has no TPU path.
- **LoRA sampling needs a from-main vllm-torchtpu** until a release carries the
  multi-LoRA pull requests (section 2).
- **v7 needs the forked driver** and auto-repair off (section 5).
- **Trainer host memory grows per step.** The default request covers about 100
  steps (section 4).

## Cleanup

LoRA Workloads outlive the run, since the runtime is shared per base model.
They are named `lora-<owner>-0-trainer` and `lora-<owner>-0-sampler`:

```bash
kubectl -n openrl-system get workloads
kubectl -n openrl-system delete workload <names>
```

Remove the deployment before the cluster, or the Filestore instance behind the
PVC outlives it:

```bash
kustomize build my-tpu | kubectl delete -f -
gcloud container clusters delete <cluster> --project <project> --zone <zone>
```
