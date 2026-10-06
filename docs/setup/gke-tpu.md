# LoRA on GKE TPUs

This guide runs OpenRL LoRA training on Cloud TPU v6e nodes in GKE. It uses the
same control plane as [Getting started](../getting-started.md): the API server,
the scheduler, Redis and a shared volume in `openrl-system`. The
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

You need `gcloud`, `kubectl`, `helm`, `kustomize`, `git` and
[uv](https://docs.astral.sh/uv/) on your machine, and a checkout of this repo.

## 1. Create the cluster

This creates a GKE Standard cluster with one CPU node (`e2-standard-4`) for the
API server, scheduler and Redis, and two TPU nodes (`ct6e-standard-4t`, four
v6e chips each) for one trainer and one sampler. Pick a region and zone with
v6e capacity. v6e quota is often zero by default, so you may need a quota
increase or a reservation:

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central2"
export ZONE="us-central2-b"
export CLUSTER="openrl-tpu"
export REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/open-rl"
```

Enable the APIs, create the cluster, and enable the Filestore CSI driver, which
provides the shared volume OpenRL uses. The TPU DRA driver needs GKE 1.34 or
newer; add `--cluster-version` if the release channel default is older:

```bash
gcloud services enable compute.googleapis.com container.googleapis.com file.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

gcloud container clusters create "${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --release-channel=regular \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --disk-size=100

gcloud container clusters update "${CLUSTER}" \
  --location="${REGION}" \
  --update-addons=GcpFilestoreCsiDriver=ENABLED
```

If your project has no `default` VPC network, the Filestore storage classes
cannot provision; create a `StorageClass` that names your network and use it as
described in [Use different storage](lora-dra.md#use-different-storage).

Add the TPU node pool. Two labels matter:

- `cloud.google.com/gke-tpu-dra-driver=true` turns off GKE's TPU device plugin
  on these nodes, so the DRA driver can manage the chips.
- `openrl.io/enabled=true` lets the OpenRL scheduler use the nodes. Set it on
  the pool, not on nodes, so a recreated node keeps it.

```bash
gcloud container node-pools create openrl-v6e \
  --cluster="${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=ct6e-standard-4t \
  --node-labels=openrl.io/enabled=true,cloud.google.com/gke-tpu-dra-driver=true \
  --num-nodes=2 \
  --disk-size=200

gcloud container clusters get-credentials "${CLUSTER}" --location="${REGION}"
```

Don't pass `--tpu-topology`: GKE then treats the pool as a single one-node
slice and refuses more than one node. To use a reservation, add
`--reservation-affinity=specific --reservation=<name>`.

## 2. Install the TPU DRA driver

The [TPU DRA driver](https://github.com/kubernetes-sigs/dra-driver-google-tpu)
publishes each node's chips as a `ResourceSlice` and prepares the
`ResourceClaim`s the OpenRL scheduler creates for TPU workers. It is a
community driver in alpha, installed from a checkout with Helm. Build and push
its image as its README describes, then install it:

```bash
git clone https://github.com/kubernetes-sigs/dra-driver-google-tpu
helm upgrade -i --create-namespace -n dra-driver-google-tpu dra-driver-google-tpu \
  dra-driver-google-tpu/deployments/helm/dra-driver-google-tpu \
  --set image.repository=<driver-image> \
  --set image.tag=<driver-tag> \
  --set kubeletPlugin.priorityClassName="" \
  --set 'kubeletPlugin.tolerations[0].key=google.com/tpu' \
  --set 'kubeletPlugin.tolerations[0].operator=Exists' \
  --set 'kubeletPlugin.tolerations[0].effect=NoSchedule' \
  --set 'kubeletPlugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key=cloud.google.com/gke-tpu-dra-driver' \
  --set 'kubeletPlugin.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator=Exists'
```

The tolerations let the driver's kubelet plugin run on the tainted TPU nodes,
and the affinity keeps it off nodes that still use GKE's device plugin. Leave
the driver in its default mode. Its shares mode lets several claims hold the
same node, so several workers would land on one node and fail to open the chips
another worker already holds.

When the driver is up, each TPU node has a `ResourceSlice` from driver
`tpu.google.com`:

```bash
kubectl get resourceslices
```

## 3. Build the TPU images

TPU workers need two images of their own. Neither is published, so build them
into your registry from the root of this repo:

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
```

Render it with the control plane images pinned and apply it. `latest` follows
`main`; a release tag pins that release. Server-side apply is required because
the Workload CRD is larger than the client-side apply limit:

```bash
make render OVERLAY=k8s/deploy/my-lora-tpu VERSION=latest | kubectl apply --server-side -f -
```

The same overlay is the place to change the worker env, for example a
`configMapGenerator` entry for `open-rl-tpu-worker-env` with `behavior: replace`.

Wait for the shared volume to bind and for each component to become ready:

```bash
kubectl -n openrl-system wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
kubectl -n openrl-system rollout status deploy/redis-store
kubectl -n openrl-system rollout status deploy/open-rl-scheduler
kubectl -n openrl-system rollout status deploy/open-rl-api-server
```

## 5. Connect

Forward the API server to your machine and leave this running:

```bash
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000
```

In another terminal, check that the API server answers:

```bash
curl http://127.0.0.1:9003/api/v1/healthz
```

## 6. Train on TPU

Ask for TPU workers through `TINKER_TAGS`, or pass the same settings in the
model's `user_metadata`. Teach `gemma-4-e2b` one answer with a LoRA adapter,
then sample from the trained adapter:

```bash
uv --project examples sync
TINKER_TAGS="openrl.trainer_accel_prefs=tpu,openrl.sampler_accel_prefs=tpu" \
  uv --project examples run python examples/tiny/tiny_sft.py base_model=google/gemma-4-e2b sample_after_train=true
```

The first run can take 15 minutes or more while the cluster starts a trainer
and a sampler, downloads the model and compiles. Check where the workers landed:

```bash
kubectl -n openrl-system get workloads,resourceclaims,pods -o wide
```

To run an e2e scenario in the cluster instead, pass `accelerator=tpu`, which
sets the same tags:

```bash
make cluster-e2e E2E_SCENARIO=tiny-lora E2E_ARGS="accelerator=tpu base_model=google/gemma-4-e2b"
```

## Troubleshooting

- **A Workload stays `NoCapacity`.** No labeled node with a free TPU fits. Check
  that the TPU nodes carry `openrl.io/enabled=true` and have a `ResourceSlice`
  from `tpu.google.com`. Each worker takes a whole node, so a second model's
  workers wait until the first model's are released.
- **A worker pod fails with `InvalidImageName`.** The image placeholders were
  not replaced; see [Deploy OpenRL](#4-deploy-openrl).
- **The client gets an error back.** The API server log has the details:
  `kubectl -n openrl-system logs deploy/open-rl-api-server`.

## Clean up

To remove OpenRL, delete the Workloads first, while the scheduler is still
running to release their TPUs, then everything the overlay created. This also
deletes the shared volume and its data:

```bash
kubectl -n openrl-system delete workloads --all
kubectl delete -k k8s/deploy/my-lora-tpu
```

Do this before deleting the cluster. The shared volume's Filestore instance is
deleted with the volume; deleting the cluster first leaves the instance behind,
and it keeps billing. Once `kubectl get pv` no longer lists the volume, delete
the cluster:

```bash
gcloud container clusters delete "${CLUSTER}" --location="${REGION}"
```
