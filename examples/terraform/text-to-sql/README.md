# Text-to-SQL on GKE with Terraform

A small GKE cluster for the [text-to-SQL recipe](../../text-to-sql/README.md) on the recipe's model, `google/gemma-4-e2b`: one node with two NVIDIA L4 GPUs, one for the LoRA trainer and one for the vLLM sampler. It is the [GKE Setup Guide](../../../docs/setup/gke-setup.md) as code, kept to the minimum needed to read and run the recipe (default network, default node service account), so it is not a production setup.

The project only needs a billing account. Terraform enables the APIs it uses.

## Layout

Two stages, applied in order. The second is separate because its Helm and kubectl providers can only be configured once the cluster exists.

| Stage | Creates |
| --- | --- |
| [`1-infra/`](1-infra/main.tf) | The `compute`, `container` and `file` APIs; a zonal GKE Standard cluster on the `default` network with the Filestore CSI driver on; one `g2-standard-24` node (2× L4) with GKE's GPU driver install and device plugin off, so the GPUs can be handed out by DRA |
| [`2-cluster/`](2-cluster/main.tf) | The NVIDIA driver (Google's COS installer DaemonSet), the NVIDIA DRA driver (Helm), and the OpenRL LoRA control plane: the recipe's own overlay [`examples/text-to-sql`](../../text-to-sql/kustomization.yaml), which is [`k8s/deploy/lora`](../../../k8s/deploy/lora) with Gemma 4 E2B as the default model |

> [!IMPORTANT]
> **Stage 2 uses a third-party Terraform provider, [`kbst/kustomization`](https://registry.terraform.io/providers/kbst/kustomization), to keep the source of truth in the repo and in sync.** It only renders the `examples/text-to-sql` overlay, at plan time, from the commit you have checked out. Terraform writes the result to `charts/openrl/.tmp/` (gitignored), and the local chart [`charts/openrl`](2-cluster/charts/openrl) prints that file verbatim. There is no rendered or vendored copy of the bundle to drift, so a `git pull` that changes the overlay or anything under `k8s/` shows up in your next `terraform plan`. Check out a release tag if you want a fixed deployment.

The bundle's shared volume is a `standard-rwx` PVC, which GKE turns into a 1 TiB Filestore instance on the `default` network. To preview what stage 2 applies, run `kubectl kustomize examples/text-to-sql` from the repo root.

## Deploy

Prerequisites: `terraform`, `gcloud` (logged in with `gcloud auth application-default login`), `kubectl`, `uv`, and quota for 2 L4 GPUs in the zone.

```bash
cd examples/terraform/text-to-sql
cp 1-infra/terraform.tfvars.example 1-infra/terraform.tfvars   # set project_id
terraform -chdir=1-infra init && terraform -chdir=1-infra apply
terraform -chdir=2-cluster init && terraform -chdir=2-cluster apply
```

If the zone has no L4s for you, set `zone` or `gpu_spot = true` in `1-infra/terraform.tfvars`.

## Check it

```bash
$(terraform -chdir=1-infra output -raw get_credentials)
kubectl get resourceslices                   # gpu.nvidia.com lists gpu-0, gpu-1
kubectl -n openrl-system get pods,pvc        # API server, scheduler, Redis running; PVC Bound
```

## Run the recipe

In one terminal, forward the API server to the port the recipe uses:

```bash
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000
```

With that running, follow [Running the Training](../../text-to-sql/README.md#running-the-training) in the recipe README from another terminal. Its default `TINKER_BASE_URL` (`http://127.0.0.1:9003`) is the port-forward above.

The first step waits while the scheduler starts the trainer and sampler, each on its own L4. The first time, that includes pulling the ~24 GB worker image and downloading the model, which takes several minutes.

## Tear down

Destroy stage 2 first: removing the OpenRL bundle deletes its PVC, which deletes the Filestore instance before the cluster goes.

```bash
terraform -chdir=2-cluster destroy
terraform -chdir=1-infra destroy
```
