# Deploy OpenRL on Kubernetes

OpenRL installs into the `openrl-system` namespace: the API server, the
scheduler, Redis, and a shared volume. Trainer and sampler workers are not
deployed up front. When a client creates a model, the API server asks the
scheduler for workers, and the scheduler places each one on a GPU through a DRA
`ResourceClaim`.

For a first run end to end, follow [Getting started](../getting-started.md). This
guide covers the cluster in more detail.

## Requirements

| | |
| --- | --- |
| Kubernetes | 1.35 or newer, for [DRA for GPUs](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/set-up-dra). |
| GPU driver | The [NVIDIA DRA driver](https://github.com/NVIDIA/k8s-dra-driver-gpu), publishing DeviceClass `gpu.nvidia.com` and a `ResourceSlice` per GPU. |
| GPUs | At least two that fit the model. A LoRA trainer and sampler each need a GPU of their own. |
| Shared storage | A `ReadWriteMany` volume that every node can mount. On GKE the bundles use Filestore (`standard-rwx`). |
| Tools | `kubectl`, and `helm` to install the DRA driver. |

If you already have such a cluster, skip to [Enroll the GPU nodes](#enroll-the-gpu-nodes).

## Create a GKE cluster

This creates a GKE Standard cluster with one CPU node and one GPU node with two
NVIDIA L4 GPUs. Choose a region and zone with L4 capacity and quota:

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER="openrl"

gcloud services enable compute.googleapis.com container.googleapis.com file.googleapis.com
```

Create the cluster with a small CPU pool for the API server, scheduler, and
Redis. Add `--cluster-version` if the release channel's default is older than
1.35:

```bash
gcloud container clusters create "${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --release-channel=regular \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --disk-size=100 \
  --addons=GcpFilestoreCsiDriver
```

> [!TIP]
> If your project has no `default` VPC network, GKE's Filestore storage classes
> cannot provision volumes. Create a `StorageClass` that names your network and
> point the shared volume at it, as in [Use different storage](#use-different-storage).

Add the GPU pool. Workers get GPUs through DRA instead of the GKE device plugin,
so the pool turns off the device plugin and the automatic driver install. The
labels opt the nodes in to OpenRL (`openrl.io/enabled=true`) and to the FFT
DaemonSets (`nvidia.com/gpu.present=true`):

```bash
gcloud container node-pools create gpu \
  --cluster="${CLUSTER}" \
  --location="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=g2-standard-24 \
  --accelerator=type=nvidia-l4,count=2,gpu-driver-version=disabled \
  --node-labels=openrl.io/enabled=true,gke-no-default-nvidia-gpu-device-plugin=true,nvidia.com/gpu.present=true \
  --image-type=COS_CONTAINERD \
  --num-nodes=1 \
  --disk-size=200

gcloud container clusters get-credentials "${CLUSTER}" --location="${REGION}"
```

Install the NVIDIA GPU driver and the NVIDIA DRA driver. The `latest` driver
installer also provides the CUDA checkpoint support that full fine-tuning needs:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.8.0" --create-namespace --namespace nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot="/home/kubernetes/bin/nvidia/"
```

Check that the GPU node publishes a `ResourceSlice` from driver `gpu.nvidia.com`
listing both GPUs:

```bash
kubectl get resourceslices
```

The pool above is already enrolled; continue with [Deploy OpenRL](#deploy-openrl).

## Enroll the GPU nodes

On an existing cluster, label the GPU nodes OpenRL may use:

```bash
kubectl label nodes <gpu-node> openrl.io/enabled=true
```

On autoscaled or Spot pools, set labels on the node pool instead: a recreated
node comes back with only the pool's labels, and the scheduler stops seeing it.

A node with neither `openrl.io/trainer` nor `openrl.io/sampler` accepts both
roles. Once either label is present, only roles labeled `true` can use that
node. To keep a role off a node, set its label to `false`; removing the label
makes the node accept every role again.

## Deploy OpenRL

Apply one release bundle. Both install the same API server, scheduler, Redis,
and shared volume:

| Bundle | Workers |
| --- | --- |
| `openrl-lora.yaml` | LoRA. Each trainer and sampler gets a GPU of its own. |
| `openrl-fft.yaml` | LoRA and full fine-tuning (FFT). FFT workers can share a GPU with other FFT workers, never with a LoRA worker. |

```bash
kubectl apply --server-side -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```

Server-side apply is required: the Workload CRD is larger than client-side
apply's annotation limit. Replace `latest/download` with `download/<tag>` to pin
a release.

The FFT bundle adds two DaemonSets on nodes labeled `nvidia.com/gpu.present=true`:
the accelerator time-slicer and the llm-d snapshot agent. They run privileged,
with host PID and host network, and mount the GKE driver directory
`/home/kubernetes/bin/nvidia`. The snapshot agent checkpoints CUDA processes and
needs NVIDIA driver r570 or newer. [FFT time slicing](../fft/time-slicing.md)
explains how the sharing works.

To deploy unreleased code from a checkout, render an overlay instead
(`k8s/deploy/lora` or `k8s/deploy/fft`). `VERSION=latest` follows `main`:

```bash
make render OVERLAY=k8s/deploy/lora VERSION=latest | kubectl apply --server-side -f -
```

Wait for the shared volume and the deployments:

```bash
kubectl -n openrl-system wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
kubectl -n openrl-system rollout status deploy/redis-store
kubectl -n openrl-system rollout status deploy/open-rl-scheduler
kubectl -n openrl-system rollout status deploy/open-rl-api-server
```

With the FFT bundle, also wait for the DaemonSets:

```bash
kubectl -n openrl-system rollout status daemonset/open-rl-accel-timeslicer
kubectl -n openrl-system rollout status daemonset/snapshot-agent
```

## Connect

The API server's service is `ClusterIP`. From your machine:

```bash
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000
curl http://127.0.0.1:9003/api/v1/healthz
```

Clients use `http://127.0.0.1:9003` as the base URL. Any Tinker SDK from 0.23
onward works. Worker pods appear once a client creates a model:

```bash
kubectl -n openrl-system get workloads,resourceclaims,pods
```

Workers download model weights on first use, so the first request for a model
waits for that.

## Use different storage

Both bundles create one PersistentVolumeClaim, `open-rl-shared-pvc`, on the GKE
Filestore class `standard-rwx`, requesting 1 TiB. The API server and every
worker mount it: it holds adapter snapshots, checkpoints, FFT weight snapshots
(about 28 GiB per 8B model), and the Hugging Face cache.

To use another backend, such as
[Managed Lustre](https://cloud.google.com/kubernetes-engine/docs/concepts/managed-lustre),
download the bundle and edit the claim before the first apply. Keep the name and
change the storage class and size:

```bash
curl -fsSLO https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: open-rl-shared-pvc
  namespace: openrl-system
spec:
  accessModes: [ReadWriteMany]
  storageClassName: lustre-rwx-1000mbps-per-tib  # was standard-rwx
  resources:
    requests:
      storage: 1200Gi                            # Lustre's size steps differ from Filestore's
```

```bash
kubectl apply --server-side -f openrl-lora.yaml
```

The storage class must already exist. A bound claim's storage class cannot
change, so edit it before the first apply, or delete the old claim and its data
first.

To use an existing claim with another name, delete `open-rl-shared-pvc` from the
bundle, point the API server's `shared-storage` volume at your claim, and set
`OPEN_RL_SHARED_PVC` on the API server container to its name. The API server
mounts that claim in every worker it creates.

From a checkout, make the same changes as a Kustomize overlay over
`k8s/deploy/lora` or `k8s/deploy/fft` and pass it to `make render`.

## Upgrade

Finish active runs and delete their Workloads before upgrading. A Workload's
spec and GPU assignment are immutable, and applying a new release does not
migrate running workers. Deleting a Workload makes the scheduler remove its pod
and release its claim. Keep the shared volume to preserve saved adapters.

```bash
kubectl -n openrl-system delete workloads --all
kubectl apply --server-side -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```

## Troubleshooting

- **A Workload stays Pending.** `kubectl -n openrl-system get workloads` shows
  the reason. `NoCapacity` means no enrolled node accepting the role has a GPU
  large enough; `WaitingForCapacity` means one exists but is busy.
  `kubectl describe resourceclaim <claim>` and
  `kubectl get pods -n nvidia-dra-driver-gpu` show whether the DRA driver is
  allocating at all.
- **A worker pod stays Pending under a placed Workload.** Check the pod's events
  for volume attach limits, taints, image pulls, or host memory.
- **The first request after creating a model is slow.** Pod scheduling, the
  image pull, and model loading all happen first.
- **A `create_model` future fails with a pod error.** Check the API server log
  (`kubectl -n openrl-system logs deploy/open-rl-api-server`); the error is
  returned in the `RequestFailedResponse`.
- **FFT: no snapshot agent on a GPU node.**
  `kubectl -n openrl-system get pods -l app.kubernetes.io/name=snapshot-agent -o wide`
  should show a running pod on every GPU node. The DaemonSet selects
  `nvidia.com/gpu.present=true`.
- **FFT: a trainer fails on its first CUDA batch with snapshot errors.** Check
  the trainer pod's log, then the `open-rl-accel-timeslicer` and `snapshot-agent`
  logs on that node.

## Clean up

On GKE, delete the cluster:

```bash
gcloud container clusters delete "${CLUSTER}" --location="${REGION}"
```

Elsewhere, delete the Workloads first, while the scheduler is still running to
release their claims, then everything the bundle created. This also deletes the
shared volume and its data:

```bash
kubectl -n openrl-system delete workloads --all
kubectl delete -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```
