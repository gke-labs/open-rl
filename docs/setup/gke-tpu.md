# LoRA on GKE TPUs

This guide runs OpenRL LoRA training on Cloud TPU v6e nodes in GKE. It uses the
same control plane as the GPU setup in [GKE Setup Guide](gke-setup.md): the API
server, the scheduler, Redis and a shared Filestore PVC in `openrl-system`. The
`k8s/deploy/lora-tpu` overlay adds what TPU workers need:

- the TPU trainer and sampler images;
- `open-rl-tpu-worker-env`, a ConfigMap of env vars for TPU workers only:
  vLLM's maximum model length and number of sequences, and a training token
  budget that fit one 32 GB v6e chip;
- a sampler-ready timeout of 30 minutes, since a TPU sampler compiles for
  several minutes before it reports ready.

The deployment does not decide which models run on TPU. Each client asks for
TPU workers with the `openrl.trainer_accel_prefs` and
`openrl.sampler_accel_prefs` settings (see
[Choosing an accelerator](../configuration.md#choosing-an-accelerator)).
Models that don't ask get GPU workers.

## Limits

- **LoRA only.** The API server refuses a full fine-tuning model that asks for TPU.
- **Each TPU worker takes a whole node.** The TPU DRA driver only hands out all
  of a node's chips at once, so a worker uses one chip and the others sit idle.
  One LoRA model needs two TPU nodes, one for its trainer and one for its
  sampler. Workers for more models wait until a node frees up.
- **One chip per worker.** The model, its adapters and activations, or the
  sampler's KV cache, must fit one chip's memory.
- **Only the first entry of each list is used.** A model asking for `tpu|gpu`
  on a cluster without TPU nodes waits for a TPU node rather than falling back
  to GPU.
- **Each new input shape compiles once.** The trainer pads batches to
  power-of-two shapes to limit this, but the first steps are slow.

## Shape

| Component | Minimum used here | Why |
| --- | --- | --- |
| GKE version | `1.34` or newer | The TPU DRA driver uses the `resource.k8s.io/v1` API. |
| CPU node pool | `1 x e2-standard-4` | API server, scheduler, Redis, system pods. |
| TPU node pool | `2 x ct6e-standard-4t` | Four v6e chips per node. One node per worker: a trainer and a sampler. |
| Shared storage | `1Ti standard-rwx` Filestore PVC | Adapter snapshots, checkpoints and the Hugging Face cache. |

## 1. Create the cluster

Choose a zone with v6e capacity. v6e quota is often zero by default, so you may
need a quota increase or a reservation.

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central2"
export ZONE="us-central2-b"
export CLUSTER="open-rl-tpu"
export REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/open-rl"
```

Create the cluster with a CPU pool and the Filestore CSI driver. Add
`--cluster-version` if the release channel's default is older than 1.34:

```bash
gcloud container clusters create "${CLUSTER}" \
  --location="${ZONE}" \
  --release-channel=rapid \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --disk-size=100 \
  --addons=GcpFilestoreCsiDriver
```

Add the TPU pool. Two labels matter:

- `cloud.google.com/gke-tpu-dra-driver=true` turns off GKE's TPU device plugin
  on these nodes, so the DRA driver can manage the chips.
- `openrl.io/enabled=true` opts the nodes in to the OpenRL scheduler. Set it on
  the pool, not on nodes, so a recreated node keeps it.

```bash
gcloud container node-pools create tpu-v6e \
  --cluster="${CLUSTER}" \
  --location="${ZONE}" \
  --machine-type=ct6e-standard-4t \
  --num-nodes=2 \
  --disk-size=200 \
  --node-labels=cloud.google.com/gke-tpu-dra-driver=true,openrl.io/enabled=true
```

Don't pass `--tpu-topology`: GKE then treats the pool as a single one-node
slice and refuses more than one node. To use a reservation, add
`--reservation-affinity=specific --reservation=<name>`.

```bash
gcloud container clusters get-credentials "${CLUSTER}" --location="${ZONE}"
```

## 2. Install the TPU DRA driver

The [TPU DRA driver](https://github.com/kubernetes-sigs/dra-driver-google-tpu)
publishes each node's chips as a `ResourceSlice` and prepares the
`ResourceClaim`s the OpenRL scheduler creates for TPU workers. It is a
community driver in alpha, installed from a checkout with Helm. Build and push
its image as its README describes, then install it:

```bash
git clone https://github.com/kubernetes-sigs/dra-driver-google-tpu
cd dra-driver-google-tpu
helm upgrade -i --create-namespace -n dra-driver-google-tpu dra-driver-google-tpu \
  deployments/helm/dra-driver-google-tpu \
  --set image.repository=<driver-image> \
  --set image.tag=<driver-tag> \
  --set kubeletPlugin.priorityClassName="" \
  --set 'kubeletPlugin.tolerations[0].key=google.com/tpu' \
  --set 'kubeletPlugin.tolerations[0].operator=Exists' \
  --set 'kubeletPlugin.tolerations[0].effect=NoSchedule' \
  --set 'kubeletPlugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key=cloud.google.com/gke-tpu-dra-driver' \
  --set 'kubeletPlugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator=Exists'
cd -
```

The tolerations let the driver's kubelet plugin run on the tainted TPU nodes,
and the affinity keeps it off nodes that still use GKE's device plugin. Leave
the driver in its default mode. Its shares mode lets several claims hold the
same node, so several workers would land on one node and fail to open the chips
another worker already holds.

Check that each TPU node has a slice from driver `tpu.google.com`:

```bash
kubectl get resourceslices
```

## 3. Build the TPU images

TPU workers need two images of their own. Neither is published, so build them
into your registry:

- **Trainer:** PyTorch with TorchTPU. TorchTPU is not on a public package index
  yet, so put its wheel under `wheels/` first (exactly one). Build it from
  commit `85628c81` or later; earlier builds compute wrong gradients for
  Gemma 4.
- **Sampler:** vLLM for TPU (`vllm-tpu`), installed from `uv.lock`.

```bash
gcloud artifacts repositories create open-rl --repository-format=docker --location="${REGION}"
make cloud-build-tpu-trainer GCP_PROJECT="${PROJECT_ID}" CLOUD_REGISTRY="${REGISTRY}" CLOUD_IMAGE_TAG=v1
make cloud-build-tpu-sampler GCP_PROJECT="${PROJECT_ID}" CLOUD_REGISTRY="${REGISTRY}" CLOUD_IMAGE_TAG=v1
```

The cluster's nodes need read access to the registry.

## 4. Deploy OpenRL

The overlay names the TPU images with the placeholders `TPU-TRAINER-IMAGE` and
`TPU-SAMPLER-IMAGE`. Until they are replaced, TPU worker pods fail with
`InvalidImageName`. Replace them in an overlay of your own:

```bash
mkdir -p k8s/deploy/my-lora-tpu
cat > k8s/deploy/my-lora-tpu/kustomization.yaml <<EOF
resources:
  - ../lora-tpu
images:
  - name: TPU-TRAINER-IMAGE
    newName: ${REGISTRY}/open-rl-tpu-trainer
    newTag: v1
  - name: TPU-SAMPLER-IMAGE
    newName: ${REGISTRY}/open-rl-tpu-sampler
    newTag: v1
EOF
make render OVERLAY=k8s/deploy/my-lora-tpu VERSION=latest | kubectl apply --server-side -f -
```

`VERSION` pins the control plane images as in [GKE Setup Guide](gke-setup.md).
The same overlay is the place to change the worker env, for example a
`configMapGenerator` entry for `open-rl-tpu-worker-env` with `behavior: replace`.

Wait for the PVC and the control plane:

```bash
kubectl -n openrl-system wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
kubectl -n openrl-system rollout status deploy/redis-store
kubectl -n openrl-system rollout status deploy/open-rl-scheduler
kubectl -n openrl-system rollout status deploy/open-rl-api-server
```

## 5. Train on TPU

Forward the API server's port:

```bash
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000
```

Ask for TPU workers through `TINKER_TAGS`, or pass the same settings in the
model's `user_metadata`:

```bash
TINKER_TAGS="openrl.trainer_accel_prefs=tpu,openrl.sampler_accel_prefs=tpu" \
  uv --project examples run python examples/tiny/tiny_sft.py \
  base_model=Qwen/Qwen3-0.6B base_url=http://127.0.0.1:9003 sample_after_train=true
```

Or run an e2e scenario in the cluster. `accelerator=tpu` sets the same tags:

```bash
make cluster-e2e E2E_SCENARIO=tiny-lora E2E_ARGS="accelerator=tpu base_model=Qwen/Qwen3-0.6B"
```

Watch the workers and their claims:

```bash
kubectl -n openrl-system get workloads,resourceclaims,pods -o wide
```

## 6. Clean up

Delete OpenRL before the cluster. The PVC's Filestore instance is deleted with
the PVC; deleting the cluster first leaves the instance behind, and it keeps
billing.

```bash
kubectl delete -k k8s/deploy/my-lora-tpu
kubectl get pv   # wait until the shared volume is gone
gcloud container clusters delete "${CLUSTER}" --location="${ZONE}"
```
