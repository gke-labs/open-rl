# kind cluster on a GPU VM

Runs the Open-RL DRA workload scheduler against real GPUs inside a kind cluster
on a single Linux GPU VM, instead of against GKE.

The point is the image loop. On GKE every code change means build → push →
pull across the internet, and the CUDA server image is multiple GB. Here the
build, the registry, and the node all live on one machine: `docker push` moves
only the layer that changed, over the local docker network. Cluster teardown and
rebuild is a minute, which makes claim-leak and reconciliation behavior cheap to
test from a clean slate.

Not `kind load docker-image`. That path is `docker save` piped into
`ctr images import`, with no layer-level dedup against what the node already
holds — a one-line source change re-shipped the whole 36 GB server image, about
24 minutes per iteration. A registry dedups by digest.

## Requirements

- A Linux VM with NVIDIA GPUs and the kernel driver installed (`nvidia-smi` works).
- **Kubernetes 1.34+.** `k8s_worker_manager.py` talks to `resource.k8s.io/v1`,
  which reached GA in 1.34. On a 1.33 node image the API server serves `v1beta1`
  and every claim call 404s. `kind-cluster.yaml` pins `kindest/node:v1.35.5`.
- Enough VRAM for the model. `Qwen/Qwen2.5-0.5B` (the overlay default) fits a
  23 GB L4 alongside a sampler; an 8B FFT does not.

## Setup

```bash
# From your laptop, against a host aliased in ~/.ssh/config:
ssh l4dev 'bash -s' < dev/kind/host-setup.sh     # one time; installs docker,
                                                 # nvidia-container-toolkit, kind,
                                                 # kubectl, helm
ssh l4dev                                        # re-login picks up the docker group

# On the VM, in a checkout of this repo:
./dev/kind/create-cluster.sh                     # also starts the registry
./dev/kind/load-images.sh
kubectl apply -k k8s/deploy/kind-dra/
```

## Iterating

```bash
./dev/kind/load-images.sh gateway                # ~1 min; covers gateway.py,
                                                 # k8s_worker_manager.py, store.py
kubectl rollout restart deployment/open-rl-gateway
```

Rebuild the `server` image only when the trainer or sampler code changes — it is
the expensive one.

## The cluster-local registry

`registry.sh` runs a `registry:2` container published on `127.0.0.1:5001` and
attached to the `kind` docker network. `create-cluster.sh` calls it, so there is
usually nothing to do by hand; run it directly if the container was removed.

The host pushes to `localhost:5001`. The node cannot reach the host's loopback,
so `kind-cluster.yaml` gives containerd a mirror rewriting that same reference to
`http://kind-registry:5000` on the docker network. One image reference works from
both sides, which is why the manifests can name `localhost:5001` directly.

Two consequences worth knowing:

- **The mirror is baked in at cluster creation.** Adding it to
  `kind-cluster.yaml` does nothing for a cluster that already exists; recreate
  the cluster.
- **`imagePullPolicy: Always` everywhere in the kind overlay.** The `:kind-dev`
  tag is mutable and republished every iteration, so the base manifests'
  `IfNotPresent` would leave the kubelet on the build it already cached and
  silently test the previous code. The worker pod templates live inside
  ConfigMaps as YAML strings where no kustomize patch can reach them, so the
  gateway stamps their image and pull policy at render time from
  `OPEN_RL_WORKER_IMAGE` / `OPEN_RL_WORKER_IMAGE_PULL_POLICY`.

### Reclaiming the disk it accumulates

A moved tag strands its old blobs in the registry and leaves the superseded
image on the node as an untagged entry. Nothing collects either: a day of
rebuilding the server image measured 24.3GB of registry backing 11.9GB of live
images. `make kind-prune` (or `./dev/kind/prune.sh`) reclaims both, in about a
minute, and leaves the BuildKit cache alone — that cache is what makes a
one-line source change rebuild in seconds.

It rebuilds the registry and re-pushes rather than running `registry
garbage-collect`, which is **not safe here**: distribution 2.8 does not
traverse OCI image indexes, and an index is what Docker's containerd store
pushes. Collecting deletes the child manifests under the tagged index and
leaves a registry that answers 200 on `:kind-dev` and 404 on every layer it
points at, breaking images that were never stale. Re-pushing from the host's
local images is the repair, so `prune.sh` refuses to touch the registry unless
all three are present to put back.

## How GPUs reach the pods

kind nodes are Docker containers, so a GPU has to be injected twice: once into
the node container, then from the node into the pod.

1. `host-setup.sh` makes `nvidia` the default Docker runtime, because kind gives
   no way to pass `--runtime` when it creates node containers.
2. It sets `accept-nvidia-visible-devices-as-volume-mounts=true`, which lets
   `kind-cluster.yaml`'s mount at `/var/run/nvidia-container-devices/all` act as
   a request for every GPU on the host.
3. It runs `nvidia-ctk system create-dev-char-symlinks`, because the kubelet
   inside the node resolves GPUs through `/dev/char/<major>:<minor>` and a
   bare-metal host does not create those.
4. `create-cluster.sh` installs the [DRA driver for NVIDIA
   GPUs](https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/install/), which
   publishes the `gpu.nvidia.com` DeviceClass and the ResourceSlices the
   scheduler scans.

`kind-cluster.yaml` also labels the node `nvidia.com/gpu.present=true`. The
driver's kubelet-plugin DaemonSet has a *required* nodeAffinity matching one of
several Node Feature Discovery labels, and NFD is not installed on kind. Without
the label the DaemonSet sits at `desired: 0` while the DeviceClasses still
register, so the cluster looks fine and every claim silently stays Pending.

ComputeDomains are disabled — they target Multi-Node NVLink, which L4s do not
have. Pass `KEEP_COMPUTE_DOMAINS=1` on NVLink hardware.

## Testing the 80gb tier without an 80 GB GPU

`_discover_cluster_gpu_products()` buckets devices by `capacity.memory`: over
40000Mi is `80gb`, everything else `24gb`. Two L4s only ever yield a `24gb`
tier, so the 80gb branch never runs.

```bash
kubectl apply -f dev/kind/synthetic-h100-resourceslice.yaml
kubectl rollout restart deployment/open-rl-gateway   # tier map is cached 15 min
```

That slice names a node that does not exist. The gateway only *lists* slices, so
the tier appears in its map and 80gb claims get created with the right CEL
`productName` selector — but kube-scheduler will never allocate against a
phantom node, so those claims sit Pending and no real pod is ever sent to a
device that isn't there. Enough to exercise tier selection, claim naming, and
reconciliation; not enough to run anything.

## Known gaps versus GKE

| | GKE | kind |
|---|---|---|
| Shared storage | RWX Filestore (`enterprise-rwx`) | RWO local-path, single node |
| GPU tiers | L4 + H100 | L4 only (plus synthetic slice) |
| `compute-domain.nvidia.com` | present | not installed |
| DCGM / Managed Prometheus | deployed | omitted |
| Nodes | multi-node, real scheduling spread | one node |

The single node is the biggest semantic difference: cross-node scheduling,
node-affinity behavior, and anything that depends on workers landing apart
cannot be reproduced here.

## Troubleshooting

**Kubelet plugin stuck in `Init:0/1`.** It cannot find `libnvidia-ml.so.1`.
Either the container toolkit is not injecting the driver into the node
(re-run `host-setup.sh` and confirm `docker info | grep -i 'default runtime'`
says `nvidia`), or `nvidiaDriverRoot` is wrong — it should be `/` for a
host-installed driver.

```bash
docker exec open-rl-dra-control-plane nvidia-smi -L     # GPUs in the node?
kubectl get resourceslices                              # driver publishing?
kubectl -n dra-driver-nvidia-gpu logs -l app.kubernetes.io/name=dra-driver-nvidia-gpu --all-containers
```

**Pods stuck in `ContainerCreating` with a DRA error.** Usually the `/dev/char`
symlinks are missing; re-run
`sudo nvidia-ctk system create-dev-char-symlinks --create-all`.

**Claims stay Pending.** No slice matches the CEL selector. Compare the claim's
`productName` expression against what the driver actually published:

```bash
kubectl get resourceslices -o jsonpath='{.items[*].spec.devices[*].attributes.productName.string}'
```
