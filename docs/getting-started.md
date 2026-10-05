# Getting started

Deploy OpenRL on a Kubernetes cluster, then fine-tune a small model with
supervised learning and with reinforcement learning from your own machine. The
training loops run on your machine; the GPUs do the work on the cluster.

You need:

- `kubectl` pointed at a Kubernetes 1.35+ cluster with the NVIDIA DRA driver
  and at least two GPUs. If you don't have one, expand the first section below
  to create one on GKE.
- `git` and [uv](https://docs.astral.sh/uv/) on your machine. No GPU needed.

<details>
<summary><b>No cluster yet? Create one on GKE</b></summary>

This creates a GKE Standard cluster with one CPU node (`e2-standard-4`) for the
API server, scheduler and Redis, and one GPU node (`g2-standard-24`, two NVIDIA
L4s) for the workers. Pick a region and zone with L4 capacity and quota:

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER="openrl"

gcloud compute accelerator-types list --filter="zone:(${ZONE}) AND name:nvidia-l4"
```

Enable the APIs, create the cluster, and enable the Filestore CSI driver, which
provides the shared volume OpenRL uses. DRA for GPUs needs GKE 1.35 or newer;
add `--cluster-version` if the release channel default is older:

```bash
gcloud services enable compute.googleapis.com container.googleapis.com file.googleapis.com

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
described in [Use different storage](setup/lora-dra.md#use-different-storage).

Add the GPU node pool. Workers get GPUs through DRA instead of the GKE device
plugin, so the pool turns off the device plugin and the automatic driver
install. The `openrl.io/enabled=true` label lets the OpenRL scheduler use the
nodes:

```bash
gcloud container node-pools create openrl-l4 \
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

Install the NVIDIA GPU driver and the NVIDIA DRA driver. The DRA driver
publishes each GPU as a `ResourceSlice`, and the OpenRL scheduler places every
worker by claiming one:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.8.0" --create-namespace --namespace nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot="/home/kubernetes/bin/nvidia/"
```

When the drivers are up, the GPU node has a `ResourceSlice` from driver
`gpu.nvidia.com` listing both GPUs:

```bash
kubectl get resourceslices
```

The GPU nodes are already labeled, so skip the labeling step below.

</details>

<details>
<summary><b>Bringing your own cluster? Check what OpenRL needs</b></summary>

OpenRL needs:

- The NVIDIA DRA driver, with DeviceClass `gpu.nvidia.com` and published
  `ResourceSlice`s (`kubectl get resourceslices`).
- At least two GPUs that fit the model. LoRA trainer and sampler workers each
  get a GPU of their own.
- A `ReadWriteMany` storage class. The release requests a 1TiB volume on GKE's
  `standard-rwx` Filestore class; to use your own class, edit the claim before
  applying, as described in [Use different storage](setup/lora-dra.md#use-different-storage).

On autoscaled or spot node pools, set the `openrl.io/enabled=true` label on the
pool rather than on nodes: a recreated node comes back with only the pool's
labels.

</details>

## 1. Deploy OpenRL

Label the GPU nodes OpenRL may use:

```bash
kubectl label nodes <gpu-node> openrl.io/enabled=true
```

Apply the release. Server-side apply is required because the Workload CRD is
larger than the client-side apply limit:

```bash
kubectl apply --server-side -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```

This installs the API server, the scheduler, Redis, and a shared volume into
`openrl-system`. Trainer and sampler workers start on demand, when a client
creates a model.

<details>
<summary><b>Optional: full fine-tuning (FFT)</b></summary>

Skip this unless you want to train all of a model's weights. The FFT release
runs LoRA workers as above and adds full fine-tuning workers, which can take
turns on a shared GPU. Apply it instead of the LoRA release:

```bash
kubectl apply --server-side -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-fft.yaml
```

It adds two DaemonSets that coordinate GPU sharing on each GPU node, the
accelerator time-slicer and the llm-d snapshot agent. They run privileged, with
host PID and host network, on nodes labeled `nvidia.com/gpu.present=true`, and
need NVIDIA driver r570 or newer and the GKE driver directory
`/home/kubernetes/bin/nvidia`. The GKE cluster above meets all of these. FFT
workers never share a GPU with a LoRA worker.

Wait for the DaemonSets along with the checks below:

```bash
kubectl -n openrl-system rollout status daemonset/open-rl-accel-timeslicer
kubectl -n openrl-system rollout status daemonset/snapshot-agent
```

The examples in this guide train LoRA adapters and run unchanged on either
release.

</details>

Wait for the shared volume to bind and for each component to become ready:

```bash
kubectl -n openrl-system wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
kubectl -n openrl-system rollout status deploy/redis-store
kubectl -n openrl-system rollout status deploy/open-rl-scheduler
kubectl -n openrl-system rollout status deploy/open-rl-api-server
```

To pin a release, replace `latest/download` with `download/<tag>`, for example
`download/v0.0.5`.

## 2. Connect

Forward the API server to your machine and leave this running:

```bash
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000
```

In another terminal, check that the API server answers:

```bash
curl http://127.0.0.1:9003/api/v1/healthz
curl http://127.0.0.1:9003/api/v1/get_server_capabilities
```

## 3. Get the examples

```bash
git clone https://github.com/gke-labs/open-rl.git
cd open-rl
uv --project examples sync
```

The examples talk to `http://127.0.0.1:9003` by default.

## 4. Supervised fine-tuning

Teach Google's `gemma-4-e2b` one answer with a LoRA adapter, then sample from
the trained adapter:

```bash
uv --project examples run python examples/tiny/tiny_sft.py base_model=google/gemma-4-e2b sample_after_train=true
```

The first run can take up to ten minutes while the cluster starts a trainer and
a sampler and downloads the model. Loss drops to zero and the adapter answers
the prompt:

```text
[tiny-sft] initial_loss=4.541667
[tiny-sft] step=01/10 loss=4.541667
[tiny-sft] step=02/10 loss=1.190104
[tiny-sft] step=03/10 loss=0.605286
[tiny-sft] step=04/10 loss=0.044357
...
[tiny-sft] final_loss=0.000000
[tiny-sft] loss_drop=100.0%
[tiny-sft] sampled_saved_adapter=' 4'
```

## 5. Reinforcement learning

Now train with a reward instead of a label. Each step samples 8 answers to
"What is 2 + 2?", rewards the ones that contain `4`, and updates the policy:

```bash
uv --project examples run python examples/tiny/tiny_rl.py base_model=google/gemma-4-e2b steps=10 learning_rate=1e-4
```

If you start it within a couple of minutes of the previous step, the workers
are still running and it finishes in about two minutes; otherwise they start
again first. Mean reward climbs to 1.0 as the samples settle on the answer:

```text
[tiny-rl] step=01 sample[0]=' Four.\nQuestion: What cried out “Fire!“?\nAnswer: Bamb'
[tiny-rl] step=01/10 loss=-0.024138 mean_reward=0.38 datums=8
...
[tiny-rl] step=04 sample[0]=' 1/2 | 2 + 2 = 1.3\n'
[tiny-rl] step=04/10 loss=-0.002864 mean_reward=0.50 datums=8
[tiny-rl] step=05 sample[0]=' 4\n\nWhat a stupid question! Of course, when you read a question'
[tiny-rl] step=05/10 loss=-16.117848 mean_reward=1.00 datums=8
...
[tiny-rl] step=10 sample[0]=' 4\nQuestion: What is 2 xx 2?\nAnswer:'
[tiny-rl] step=10/10 loss=-15.967099 mean_reward=1.00 datums=8
```

Exact numbers vary from run to run. `gemma-4-e2b` is a base model, not a chat
model, so it keeps writing after the answer; the reward only checks for `4`.

## 6. Write your own RL loop

[`examples/tiny/tiny_rl.py`](../examples/tiny/tiny_rl.py) is a complete RL loop
in one short file, written against the [Tinker](https://tinker-docs.thinkingmachines.ai/)
API that OpenRL serves. Each step:

1. **Sample** from the current policy: `save_weights_and_get_sampling_client`,
   then `sample`.
2. **Reward** each completion; here, whether it contains the target answer.
3. **Compute gradients** from the rewards: `forward_backward` with the
   `importance_sampling` loss.
4. **Update** the weights: `optim_step`.

Change the prompt, the reward, or the loss, and you have your own experiment.
Any Tinker SDK client works against OpenRL, so the larger examples carry over:

- [Text-to-SQL RL recipe](../examples/text-to-sql/README.md): SFT then RL with
  SQL execution rewards on Gemma 4.
- [Pig Latin](../examples/sft/pig-latin/piglatin_sft_notebook.ipynb) and
  [Text-to-SQL](../examples/sft/text-to-sql/texttosql_sft_notebook.ipynb) SFT
  notebooks.
- [Tinker Cookbook recipes](../examples/tinker-cookbook/README.md) run against
  OpenRL unchanged.

## Troubleshooting

- **A step hangs before the first loss line.** The cluster is starting a
  worker. `kubectl -n openrl-system get workloads,pods` shows its progress. A
  Workload stuck with `NoCapacity` means no labeled node has a free GPU large
  enough; `WaitingForCapacity` means one exists but is busy.
- **A worker pod stays Pending.** `kubectl -n openrl-system describe pod <pod>`
  shows why: image pulls, volume attach limits, taints, or host memory.
- **The client gets an error back.** The API server log has the details:
  `kubectl -n openrl-system logs deploy/open-rl-api-server`.

## Clean up

OpenRL releases the workers and their GPUs about two minutes after your client
exits. To release them right away:

```bash
kubectl -n openrl-system delete workloads --all
```

To remove OpenRL, delete the Workloads first, as above, while the scheduler is
still running to release their GPUs, then everything the release created. This
also deletes the shared volume and its data:

```bash
kubectl delete -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```

Use `openrl-fft.yaml` if you applied the FFT release. If you created the GKE
cluster above, delete it instead:

```bash
gcloud container clusters delete "${CLUSTER}" --location="${REGION}"
```
