# Getting started

Deploy OpenRL on a Kubernetes cluster, then fine-tune a small model with
supervised learning and with reinforcement learning from your own machine. The
training loops run on your machine; the GPUs do the work on the cluster.

You need:

- A Kubernetes 1.35+ cluster with the NVIDIA DRA driver and at least two GPUs.
  To create one on GKE, follow [Create a GKE cluster](setup/kubernetes.md#create-a-gke-cluster).
- `kubectl` pointed at that cluster.
- `git` and [uv](https://docs.astral.sh/uv/) on your machine. No GPU needed.

## 1. Deploy OpenRL

Label the GPU nodes OpenRL may use. (The GKE pool in the setup guide is already
labeled.)

```bash
kubectl label nodes <gpu-node> openrl.io/enabled=true
```

Apply the release and wait for it:

```bash
kubectl apply --server-side -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml

kubectl -n openrl-system wait --for=jsonpath='{.status.phase}'=Bound pvc/open-rl-shared-pvc --timeout=5m
kubectl -n openrl-system rollout status deploy/open-rl-api-server
```

This installs the API server, the scheduler, Redis, and a shared volume into
`openrl-system`. Trainer and sampler workers start on demand, when a client
creates a model. For full fine-tuning, other storage, and troubleshooting, see
[Deploy OpenRL on Kubernetes](setup/kubernetes.md).

## 2. Connect

Forward the API server to your machine and leave this running:

```bash
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000
```

In another terminal:

```bash
curl http://127.0.0.1:9003/api/v1/healthz
```

## 3. Get the examples

```bash
git clone https://github.com/gke-labs/open-rl.git
cd open-rl
uv --project examples sync
```

The examples talk to `http://127.0.0.1:9003` by default.

## 4. Supervised fine-tuning

Teach `Qwen/Qwen2.5-0.5B` one answer with a LoRA adapter, then sample from the
trained adapter:

```bash
uv --project examples run python examples/tiny/tiny_sft.py sample_after_train=true
```

The first run waits a few minutes while the cluster starts a trainer and a
sampler and downloads the model. Loss drops to about zero and the adapter
answers the prompt:

```text
[tiny-sft] initial_loss=0.850260
[tiny-sft] step=01/10 loss=0.850260
[tiny-sft] step=02/10 loss=0.003099
...
[tiny-sft] final_loss=0.000001
[tiny-sft] loss_drop=100.0%
[tiny-sft] sampled_saved_adapter=' 4'
```

## 5. Reinforcement learning

Now train with a reward instead of a label. Each step samples 8 answers to
"What is 2 + 2?", rewards the ones that contain `4`, and updates the policy:

```bash
uv --project examples run python examples/tiny/tiny_rl.py steps=10 learning_rate=1e-4
```

If you start it within a couple of minutes of the previous step, the workers
are still running and it finishes in under a minute; otherwise they start
again first. Mean reward climbs to 1.0 as the samples settle on the answer:

```text
[tiny-rl] step=01/10 loss=1.507891 mean_reward=0.50 datums=8
...
[tiny-rl] step=05 sample[0]=' The answer is 4.\nYou are a helpful assistant, ...'
[tiny-rl] step=05/10 loss=0.719542 mean_reward=0.88 datums=8
...
[tiny-rl] step=10 sample[0]=' 4<|endoftext|>'
[tiny-rl] step=10/10 loss=-4.624406 mean_reward=1.00 datums=8
```

Exact numbers vary from run to run.

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

## Clean up

OpenRL releases the workers and their GPUs about two minutes after your client
exits. To release them right away:

```bash
kubectl -n openrl-system delete workloads --all
```

To remove OpenRL, see [Clean up](setup/kubernetes.md#clean-up).
