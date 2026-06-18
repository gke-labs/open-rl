# GKE FFT DRA Worker Manager Setup Guide

This guide describes the core cluster shape introduced by this PR. It separates
three ideas that build on each other:

1. **DRA pinning:** one manually created `ResourceClaim` allocates one physical
   GPU, and every trainer worker pod that references that claim is scheduled onto
   the node that can access that same device.
2. **Kubernetes worker manager:** the gateway creates one trainer worker pod per
   `model_id`, instead of relying on a static trainer Deployment.
3. **OpenRL GPU coordination:** a node-local snapshot-agent DaemonSet
   coordinates checkpoint/restore so only one trainer worker enters a CUDA batch at
   a time on that node.

## Architecture at a glance

There are three separate responsibilities in this PR.

First, DRA is used only for GPU allocation and placement. The deployment creates
two static `ResourceClaims` named `open-rl-trainer-gpu-1` and `open-rl-sampler-gpu-1`. Trainer worker pods reference `open-rl-trainer-gpu-1`, while Sampler worker pods reference `open-rl-sampler-gpu-1`. Kubernetes allocates one matching NVIDIA GPU to each claim and schedules those pods onto separate physical nodes where those devices are available.

Second, the Kubernetes worker manager is the deployment launcher. It runs inside
the gateway process today. When the gateway receives `create_model` in FFT mode, it creates a trainer pod for that `model_id`. When it receives `create_sampling_client`, it creates a dedicated vLLM sampler pod for that `model_id` from the sampler worker pod template. It enqueues the request
on the model-specific Redis queue. It is idempotent: if the trainer worker pod
for a model is already running, it reuses it.

Third, the OpenRL snapshot agent is the runtime GPU coordinator. It runs as a
node-local DaemonSet on GPU nodes, with `hostPID` and `hostNetwork` enabled.
Trainer worker pods connect to the agent on their node with
`OPEN_RL_SNAPSHOT_AGENT_HOST=status.hostIP` and
`OPEN_RL_SNAPSHOT_AGENT_PORT=9753`. The training processor registers its PID
with the agent and wraps GPU work in acquire/release calls. The agent keeps a
FIFO queue, allows one active process at a time, checkpoints on release, and
restores on acquire.

The request flow is:

1. A client calls `create_model`.
2. The gateway creates a unique `model_id`.
3. The Kubernetes worker manager ensures a trainer worker pod exists for that
   model.
4. The trainer worker pod references `open-rl-trainer-gpu-1`, so Kubernetes
   places it on the DRA GPU node.
5. The gateway enqueues the create request on the model's Redis queue.
6. The trainer worker drains that queue and uses the node-local snapshot agent
   before entering CUDA sections.

The whole shape has two layers. The top layer creates pods, places them, and
moves requests through Redis. The bottom layer runs on the GPU node and
coordinates which colocated trainer worker may enter CUDA.

```mermaid
flowchart TD
    subgraph launch["Layer 1: launch and placement"]
        client["Client"]
        gateway["OpenRL gateway\nworker manager lives here"]
        kube["Kubernetes API"]
        redis["Redis\nper-model queue + future"]
        claim["DRA ResourceClaim\nopen-rl-trainer-gpu-1"]
    end

    subgraph node["Layer 2: node-local GPU coordination"]
        workerA["trainer worker pod\nmodel A"]
        workerB["trainer worker pod\nmodel B"]
        agent["snapshot-agent DaemonSet\none per GPU node"]
        gpu["Physical GPU"]
    end

    client -->|"create_model / retrieve_future"| gateway
    gateway -->|"create or reuse pod"| kube
    gateway -->|"enqueue request / read future"| redis
    kube -->|"schedule pods that reference claim"| workerA
    kube -->|"schedule pods that reference claim"| workerB
    claim -.->|"pins placement to one device"| workerA
    claim -.->|"pins placement to one device"| workerB
    claim -.->|"allocates physical device"| gpu

    workerA <-->|"pop request / write result"| redis
    workerB <-->|"pop request / write result"| redis
    workerA -->|"acquire / release PID"| agent
    workerB -->|"acquire / release PID"| agent
    agent -->|"checkpoint / restore"| workerA
    agent -->|"checkpoint / restore"| workerB
    agent -->|"one active CUDA section"| gpu
```

The orchestration is currently cooperative: worker code calls acquire/release,
and the OpenRL snapshot agent serializes those calls with a FIFO lock. The
Kubernetes worker manager only launches pods; it is not the runtime time-slice
scheduler.

## 1. DRA pins the GPU allocation

`k8s/deploy/distributed-fft-timeslice/06-gpu-resourceclaim.yaml` and `08-sampler-resourceclaim.yaml` create dedicated namespace-scoped `ResourceClaims` for Trainers and Samplers:

```yaml
apiVersion: resource.k8s.io/v1
kind: ResourceClaim
metadata:
  name: open-rl-trainer-gpu-1 # (and open-rl-sampler-gpu-1)
spec:
  devices:
    requests:
    - name: gpu
      exactly:
        deviceClassName: gpu.nvidia.com
```

Trainer worker pods reference `open-rl-trainer-gpu-1`, while Sampler worker pods reference `open-rl-sampler-gpu-1`:

```yaml
resourceClaims:
- name: trainer-gpu # (or sampler-gpu)
  resourceClaimName: open-rl-trainer-gpu-1 # (or open-rl-sampler-gpu-1)
```

Because these are shared `ResourceClaims`, Kubernetes allocates a single matching device to each claim and schedules referencing pods onto the dedicated nodes where those claims reside (`group.timeslice.io/trainers` vs `samplers`).

DRA is the allocation and placement layer. It does not serialize CUDA execution
by itself. This PR is intentionally an oversubscription model: multiple trainer
worker pods can reference the same GPU claim, and OpenRL decides which trainer
worker may touch CUDA at a given time.

## 2. The gateway creates one trainer worker pod per model

The per-model Redis queues and future protocol are unchanged. The new behavior
is only how dedicated trainer workers are launched:

```mermaid
sequenceDiagram
    participant C as Client
    participant G as Gateway
    participant K as Kubernetes API
    participant R as Redis
    participant W as trainer worker pod

    C->>G: POST /create_model
    G->>G: model_id = uuid4
    G->>K: create Pod open-rl-trainer-<model_id>
    G->>R: RPUSH open_rl:queue:<model_id>
    G-->>C: request_id = model_id
    W->>R: BLPOP open_rl:queue:<model_id>
    W->>W: load base model and process request
    W->>R: resolve open_rl:future:<request_id>
```

`server/k8s_worker_manager.py` renders the trainer worker pod from the ConfigMap
template in `05-worker-pod-template.yaml`. It stamps:

- the pod name, derived from the model id
- trainer worker labels, including `app=open-rl-trainer-worker`
- time-slicing labels (`snapshot-agent=true`, `timeslice.io/group`,
  `timeslice.io/job-id`) that preserve the shape of a future coordinator
  integration
- `OPEN_RL_TIME_SLICE_JOB_ID`, aligned with the `timeslice.io/job-id` label
- `--model-id <model_id>`, so the worker drains only its own queue

The gateway still has a local subprocess launcher for VM development. Select the
cluster launcher with `OPEN_RL_WORKER_MANAGER=kubernetes`.

## 3. A node-local snapshot agent coordinates GPU windows

The deployment includes `07-snapshot-agent-daemonset.yaml`, which runs one
OpenRL snapshot agent on each trainer GPU node:

```yaml
hostPID: true
hostNetwork: true
command: ["uv", "run", "python", "-m", "snapshot_agent.serve"]
args: ["--listen-host", "0.0.0.0", "--port", "9753"]
```

The dynamically launched trainer worker pods run the normal training processor:

```yaml
command: ["uv", "run", "python", "-m", "server.training_requests_processor"]
```

The training processor uses:

- `OPEN_RL_SNAPSHOT_AGENT_HOST` from the pod's `status.hostIP`
- `OPEN_RL_SNAPSHOT_AGENT_PORT=9753`
- `cuda-checkpoint` from the server image for process-level CUDA
  checkpoint/restore inside the node-local agent

Trainer workers talk to a host-level agent on their node, and that agent owns the
in-memory queue and checkpoint/restore state for processes sharing the physical
GPU. The trainer worker pod currently sets `hostPID: true` so the PID it
registers is valid from the node-local agent's host PID namespace; replacing that with
Kubernetes PID discovery is a later hardening step.

## Requirements

- GKE Standard cluster on **1.35 or newer** ([DRA for GPUs](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/set-up-dra)
  needs it) with the Filestore CSI driver enabled (see
  [gke-setup.md](gke-setup.md) for the base cluster, CPU pool, and PVC details).
- A working NVIDIA GPU driver on the DRA node. The OpenRL snapshot path uses
  `cuda-checkpoint --action lock/checkpoint/restore/unlock`, so use driver
  **r570 or newer**.
- The **NVIDIA DRA GPU driver** (Helm chart `nvidia-dra-driver-gpu` >= 25.8.0)
  so all trainer worker pods can share one GPU through a single `ResourceClaim`.
- Helm v3 for the DRA-driver chart.

## Setup 1: Create the DRA GPU node pool

Trainer worker pods share the GPU through the `open-rl-trainer-gpu-1`
`ResourceClaim` (`06-gpu-resourceclaim.yaml`) instead of device-plugin time
sharing, so the node pool disables the default device plugin and automatic
driver install (per the
[GKE DRA setup guide](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/set-up-dra);
follow it if these flags have drifted):

```bash
gcloud container node-pools create gpu-dra \
  --cluster "${CLUSTER}" --zone "${ZONE}" \
  --machine-type g2-standard-24 \
  --accelerator "type=nvidia-l4,count=1,gpu-driver-version=disabled" \
  --node-labels="group.timeslice.io/trainers=true,group.timeslice.io/samplers=true,gke-no-default-nvidia-gpu-device-plugin=true,nvidia.com/gpu.present=true" \
  --num-nodes 2
```

Install the GPU driver manually. Use the `latest` installer so the
`cuda-checkpoint` command set is available:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

Then install the NVIDIA DRA driver, which is the kubelet plugin that discovers
GPUs and serves `ResourceClaim` allocations:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.8.0" --create-namespace --namespace nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot="/home/kubernetes/bin/nvidia/"
```

Notes:

- `group.timeslice.io/trainers=true` is the node label used by these manifests
  for the `trainers` group. It also matches the trainer worker pod template's
  `OPEN_RL_TIME_SLICE_GROUP` value.
- The trainer GPU claim does not bound how many trainer jobs reference the GPU.
  The node-local OpenRL snapshot agent decides which trainer worker may run a
  CUDA batch at a time.
- Upstream caveat: DRA for GPUs is a supported GKE path on 1.35+, but GPU
  allocation is still marked experimental in the upstream
  [k8s-dra-driver-gpu](https://github.com/NVIDIA/k8s-dra-driver-gpu) repo. If
  it misbehaves or the cluster is pre-1.35, the fallback is device-plugin
  **time sharing**: create the pool with
  `--accelerator "type=nvidia-l4,count=1,gpu-sharing-strategy=time-sharing,max-shared-clients-per-gpu=2" --gpu-driver-version=latest`
  and only the `group.timeslice.io/trainers=true` label, skip the DRA driver
  install, and in `05-worker-pod-template.yaml` replace the
  `resources.claims`/`resourceClaims` stanzas with `nvidia.com/gpu: "1"`
  requests/limits. On non-GKE clusters the equivalent is the NVIDIA device
  plugin's [time-slicing config](https://github.com/NVIDIA/k8s-device-plugin#shared-access-to-gpus-with-cuda-time-slicing)
  (`replicas: 2`) plus the node label.

## Setup 2: Build, push, and deploy OpenRL

```bash
make build-images push-images
make deploy-fft-timeslice
```

`k8s/deploy/distributed-fft-timeslice/` deploys Redis, the shared PVC, the
shared GPU `ResourceClaim`, the node-local snapshot-agent DaemonSet, and the
gateway with `OPEN_RL_ENABLE_FFT=true` and
`OPEN_RL_WORKER_MANAGER=kubernetes`.
The deployment assumes one base model per rollout: set `BASE_MODEL` in
`kustomization.yaml`, and the gateway uses that value for `get_info` and
`create_model` requests that do not explicitly pass a base model.

There are no static worker deployments. Every `create_model` call makes the gateway create a trainer pod named `open-rl-trainer-<model-id>`, and every `create_sampling_client` call makes the gateway create a dedicated vLLM sampler pod named `open-rl-sampler-<model-id>`. Both are labeled:

```yaml
snapshot-agent: "true"          # OpenRL coordinator marker
timeslice.io/group: trainers    # (or samplers for vLLM rollout workers)
timeslice.io/job-id: <model-id> # per-worker identity
```

The gateway's `open-rl-sa` service account has a Role allowing pod CRUD in the workload namespace (`03-rbac.yaml`). When weight updates occur during FFT training, Trainers write checkpoints to NFS `/mnt/shared`, and Samplers dynamically reload those checkpoint safetensors in-place in ~1.1 seconds while yielding GPU VRAM via cooperative sleep.

## Setup 3: Run training on the cluster

```bash
kubectl port-forward svc/open-rl-gateway-service 8000:8000 &
make test e2e fft-gsm8k BASE_URL=http://127.0.0.1:8000
```

## Troubleshooting

- **Trainer worker pod Pending**: the `open-rl-trainer-gpu-1` `ResourceClaim`
  could not be allocated; the DRA driver isn't running, or no GPU node carries the
  `group.timeslice.io/trainers` label. `kubectl describe resourceclaim
  open-rl-trainer-gpu-1` and `kubectl get pods -n nvidia-dra-driver-gpu` show why.
- **Additional trainer worker pod Pending**: all trainer workers reference the same
  `ResourceClaim`, so they should be schedulable onto the node that owns that
  claim. If only later pods are pending, check pod events for PVC attach limits,
  node selectors, taints, image pull errors, or an unallocated claim.
- **Trainer worker fails on first CUDA batch with snapshot errors**: check the
  trainer worker pod logs and the `open-rl-snapshot-agent` DaemonSet logs. The
  worker should connect to `OPEN_RL_SNAPSHOT_AGENT_HOST:OPEN_RL_SNAPSHOT_AGENT_PORT`,
  `cuda-checkpoint` must be on the agent `PATH`, and the node driver must support
  the requested checkpoint operations.
- **`create_model` future fails with a pod-create error**: check gateway logs and
  RBAC; the error message is propagated into the `RequestFailedResponse`.
- **First request after `create_model` is slow**: pod scheduling, image pull, and
  model load all happen before the worker drains its queue; pre-pull the server
  image on the GPU node to cut this down.
