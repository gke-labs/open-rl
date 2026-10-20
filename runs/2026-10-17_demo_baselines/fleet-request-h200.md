# Fleet request: H200 pool for the fleet demo

Project `cdrollouts-sunilarora`, cluster `open-rl-dra` (us-central1, GKE 1.35).

## What we ask for

- 1 x `a3-ultragpu-8g` (8 x H200 141 GB), zone `us-central1-b`. The H200 has no
  smaller shape; 8 GPUs per node is the minimum step.
- Quota: `GPUS-PER-GPU-FAMILY` in `us-central1`: `NVIDIA_H200` to 8 and `NVIDIA_H100` to
  at least 10. The H100 limit for new node pools is 0 today: a 1-node a3-highgpu-4g pool
  failed with `Quota 'GPUS_PER_GPU_FAMILY' exceeded. Limit: 0.0 in region us-central1`
  (2026-10-18 00:33 UTC), while resizing the existing 1-GPU pools from 2 to 3 nodes
  worked minutes earlier. The H200 attempt (2026-10-17 23:41 UTC) never reached the
  quota check: eight attempts in 35 minutes, all `ZONE_RESOURCE_POOL_EXHAUSTED` in
  us-central1-b.
- Also useful: 1 x `a3-highgpu-4g` (4 x H100 on one node, us-central1-a) as the first
  multi-GPU node to prove placement across devices before the H200 arrives.
- Capacity: a reservation for 1 machine for the demo window, or approval to use a
  flex-start node pool (DWS) if a reservation is not possible. The project has no
  reservations today.
- The 4 H100s stay. The L4 pools (8 GPUs) stay until the H200 node is up and has run
  one drill. Only then do they go away. Target fleet: 8 H200 + 4 H100 = 12 GPUs.

## Why H200 and why us-central1-b

- Same Hopper architecture as the H100s we run today. No new kernels, no new compile
  paths. The only new work is one compile-cache warm-up per model.
- us-central1-b is the only zone in the region with `a3-ultragpu-8g`. The Filestore
  volume that carries every checkpoint and adapter is also in us-central1-b.
- 2.9 TB of host memory per node. A gemma FFT trainer asks for 91 GiB of host memory,
  which limits us to 2 per node on the 1-GPU H100 machines today.

## Node pool command (run after quota and capacity are confirmed)

```bash
gcloud container node-pools create h200-8x \
  --cluster open-rl-dra --region us-central1 --node-locations us-central1-b \
  --project cdrollouts-sunilarora \
  --machine-type a3-ultragpu-8g \
  --accelerator type=nvidia-h200-141gb,count=8,gpu-driver-version=disabled \
  --num-nodes 1 \
  --image-type COS_CONTAINERD --disk-type pd-balanced --disk-size 200 \
  --ephemeral-storage-local-ssd count=12 \
  --node-labels openrl.io/enabled=true,openrl.io/trainer=true,openrl.io/sampler=true,nvidia.com/gpu.present=true,gke-no-default-nvidia-gpu-device-plugin=true \
  --node-taints nvidia.com/gpu=present:NoSchedule \
  --scopes default
# With a reservation:  add  --reservation-affinity specific --reservation <name>
# With flex-start:     add  --flex-start --enable-autoscaling --min-nodes 0 --max-nodes 1 --num-nodes 0
```

The node carries both role labels; the scheduler lets trainers and samplers share a
node that accepts both. The `gpu-driver-version=disabled` setting matches the existing
pools, where the cluster's own `nvidia-driver-installer` DaemonSet (driver 580) and
the NVIDIA DRA driver v25.8.0 handle the GPUs.

## After the pool is up

1. Confirm the DRA ResourceSlice for the node lists 8 H200 devices.
2. Confirm the time-slicer and snapshot-agent DaemonSets run on the node.
3. The pre-pull DaemonSet pulls the worker image on the node (about 3 minutes).
4. Warm the compile cache: gemma-4-e2b and Qwen3-0.6B, LoRA and FFT, on H200. Four
   sampler starts.
5. Drill 3: the 12-job drill, then the 24-job drill, with the L4s still in the fleet.
6. If the drills pass, remove the two L4 pools and run the 24-job drill once more.

## Sizing on 8 H200 + 4 H100

The warm sizing asked for about 18 H100 GPUs for 24 jobs. One H200 node is not that,
so the 24-job drill will tell us the real number. What the H200 node changes: 141 GB
per GPU holds more LoRA runtimes and more time-sliced FFT pairs per GPU than an 80 GB
H100, and host memory no longer caps gemma FFT trainers. If 12 GPUs come up short, the
fallback is to keep the L4s for the Qwen3-0.6B LoRA jobs.

## Quota requests filed 2026-10-19 06:35 UTC (Cloud Quotas API, project cdrollouts-sunilarora)

All for `GPUS-PER-GPU-FAMILY-per-project-region`, preferred value 16, contact
sunilarora@google.com. The Spot attempt (`h200-spot`, 06:22 UTC) failed on this same quota
(`Limit: 0.0`), so Spot and on-demand share it for these families.

| Family | Regions | Request IDs |
|---|---|---|
| NVIDIA_H200 | us-central1, us-east4, us-east5, us-east7, us-south1, us-west1, europe-west1, europe-west4, asia-south1, asia-south2 | c3ec12f3, 4c7f0d3c, dbfca2c4, 9bdf2a1a, b2d47b08, 6d7276a1, efd957bb, 07a6ea5e, 60d3071b, aa775f3d |
| NVIDIA_B200 | us-central1, us-east1, us-east4, us-west3, europe-west4, asia-southeast1 | e13d2bc7, 7496ed03, 2748ee31, 5642a24a, 80941b4c, 66973796 |
| NVIDIA_H100 | us-central1 | c640a7da |

Check status:
`gcloud beta quotas preferences list --project cdrollouts-sunilarora` or the console quota
page. H200 machines (a3-ultragpu-8g) exist in: asia-south1-b, asia-south2-c, europe-west1-b,
europe-west4-a, us-central1-b, us-east4-b, us-east5-a, us-east7-c, us-south1-b, us-west1-c.
B200 (a4-highgpu-8g) in 17 zones including us-east1-b/d, us-east4-b, us-west3-b/c,
europe-west4-b, asia-southeast1-b.

## A second cell in another region

A GKE cluster's node pools must be in the cluster's region, so capacity in another region
means a second OpenRL cell there: a regional cluster, a Filestore instance, the bundle
with the dev images, and the client Jobs, all in that region. One control plane across
regions is not possible with GKE node pools; a GKE Fleet groups the clusters for one view
but each keeps its own control plane.

## Result of the CLI requests (checked 06:50 UTC, about 15 minutes after filing)

Every one of the 17 requests came back `Quota request denied` by the automatic review, in
every region, for H200, B200 and the H100 family alike. The one approved entry in the
project's history is an older request without a family dimension, granted to 10. So the
self-service path does not open these families for this project. The request has to go
through the account team or a support case, which is a person, not the API.

What to send them: project `cdrollouts-sunilarora`, quota `GPUS-PER-GPU-FAMILY-per-project-region`,
family `NVIDIA_H200` (and `NVIDIA_B200`), value 16, in us-central1 plus two or three of
us-east4, us-east5, us-south1, us-west1, and the Spot use for drills with a reservation for
the demo date. The request IDs above are the evidence of the automatic denial.

## Cell dry run, us-east4 (2026-10-19 06:40-07:10 UTC)

`scripts/cell_up.sh cell-use4 us-east4 us-east4-b` created the regional cluster and the
Spot a3-ultragpu-8g pool; the pool failed on `GPUS_PER_GPU_FAMILY ... Limit: 0.0 in region
us-east4`, as expected, and the script reported it every 20 s until stopped. The cluster
was then deleted with `scripts/cell_down.sh`; no Filestore instance was left behind. The
bundle apply on a fresh cluster is still unverified (my check ran from zsh, where the
script's `$K` variable does not word-split; the script itself is bash). Verify it the
first time a cell gets a GPU node.

