#!/usr/bin/env bash
# Bring up an OpenRL drill cell in any GCP region: a GKE cluster, one GPU node
# pool (Spot by default), the NVIDIA driver and DRA driver, and the OpenRL
# bundle from k8s/deploy/gke-cell. Trainer, sampler, shared storage and the
# client Jobs all live in this region, which is what locality needs.
#
#   scripts/cell_up.sh <name> <region> <zone> [machine] [accelerator] [count] [local-ssds]
#
#   scripts/cell_up.sh cell-use4 us-east4 us-east4-b                                  # 1 x a3-ultragpu-8g (8 x H200), Spot
#   scripts/cell_up.sh cell-use1 us-east1 us-east1-b a4-highgpu-8g nvidia-b200 8 32   # 1 x a4-highgpu-8g (8 x B200), Spot
#   SPOT=0 scripts/cell_up.sh ...                                                      # on-demand (the demo, with a reservation)
#   GPU_NODES=2 scripts/cell_up.sh ...                                                 # more than one GPU node
#
# Idempotent: every step checks for what already exists, so a run that stopped
# at a quota or stockout error can be rerun as is. The GPU pool is created
# asynchronously; the script waits for a node and reports the pool's errors
# (GCE_QUOTA_EXCEEDED, ZONE_RESOURCE_POOL_EXHAUSTED) instead of hanging.
#
# After it finishes: scripts/cell_seed_hf.sh copies the model cache from the
# main cluster, then the warm-up round and the drills run against this cell's
# kubectl context (printed at the end).
set -euo pipefail
NAME=${1:?name}; REGION=${2:?region}; ZONE=${3:?zone}
MACHINE=${4:-a3-ultragpu-8g}; ACCEL=${5:-nvidia-h200-141gb}; COUNT=${6:-8}; SSDS=${7:-32}
PROJECT=${PROJECT:-cdrollouts-sunilarora}
SPOT=${SPOT:-1}
GPU_NODES=${GPU_NODES:-1}
OVERLAY=${OVERLAY:-k8s/deploy/gke-cell}
cd "$(dirname "$0")/.."
CTX="gke_${PROJECT}_${REGION}_${NAME}"
log() { printf '\n== %s\n' "$*"; }

log "cluster $NAME in $REGION ($ZONE)"
if ! gcloud container clusters describe "$NAME" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud container clusters create "$NAME" --location "$REGION" --node-locations "$ZONE" --project "$PROJECT" \
    --release-channel regular --machine-type e2-standard-4 --num-nodes 1 --disk-size 100 \
    --addons GcpFilestoreCsiDriver,GcePersistentDiskCsiDriver \
    --logging SYSTEM,WORKLOAD --monitoring SYSTEM --quiet
fi
# get-credentials switches the current context; put it back so other shells keep theirs.
prev_ctx=$(kubectl config current-context 2>/dev/null || true)
gcloud container clusters get-credentials "$NAME" --location "$REGION" --project "$PROJECT" >/dev/null
[ -n "$prev_ctx" ] && kubectl config use-context "$prev_ctx" >/dev/null
K="kubectl --context $CTX"

log "GPU node pool gpu-8x ($MACHINE, $COUNT x $ACCEL, spot=$SPOT)"
if ! gcloud container node-pools describe gpu-8x --cluster "$NAME" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  spot_flag=""; [ "$SPOT" = 1 ] && spot_flag="--spot"
  disk_flags="--disk-type hyperdisk-balanced --disk-size 200 --ephemeral-storage-local-ssd count=$SSDS"
  case "$MACHINE" in a3-highgpu-*|g2-*) disk_flags="--disk-type pd-balanced --disk-size 200" ;; esac
  # shellcheck disable=SC2086
  gcloud container node-pools create gpu-8x --cluster "$NAME" --location "$REGION" --node-locations "$ZONE" --project "$PROJECT" \
    --machine-type "$MACHINE" --accelerator "type=$ACCEL,count=$COUNT,gpu-driver-version=disabled" \
    --num-nodes "$GPU_NODES" --image-type COS_CONTAINERD $disk_flags $spot_flag \
    --node-labels openrl.io/enabled=true,openrl.io/trainer=true,openrl.io/sampler=true,nvidia.com/gpu.present=true,gke-no-default-nvidia-gpu-device-plugin=true \
    --node-taints nvidia.com/gpu=present:NoSchedule --async --quiet
fi

log "waiting for a GPU node (pool errors are reported below)"
mig=""
for _ in $(seq 1 60); do
  if [ "$($K get nodes -l cloud.google.com/gke-nodepool=gpu-8x --no-headers 2>/dev/null | grep -c ' Ready')" -ge 1 ]; then break; fi
  [ -z "$mig" ] && mig=$(gcloud compute instance-groups managed list --project "$PROJECT" --filter="name~gke-$NAME-gpu-8x" --format='value(name)' 2>/dev/null | head -1 || true)
  if [ -n "$mig" ]; then
    err=$(gcloud compute instance-groups managed list-errors "$mig" --zone "$ZONE" --project "$PROJECT" --format='value(error.code,error.message)' 2>/dev/null | head -1 || true)
    [ -n "$err" ] && echo "  pool: $err" | cut -c1-200
  fi
  sleep 20
done
if [ "$($K get nodes -l cloud.google.com/gke-nodepool=gpu-8x --no-headers 2>/dev/null | grep -c ' Ready')" -lt 1 ]; then
  echo "no GPU node after 20 minutes; the pool keeps retrying. Rerun this script when capacity or quota arrives." >&2
  exit 2
fi

log "NVIDIA kernel driver (installer DaemonSet) and DRA driver (Helm)"
$K apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml >/dev/null
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
helm repo update >/dev/null
helm --kube-context "$CTX" upgrade --install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version 25.8.0 --create-namespace --namespace nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot=/home/kubernetes/bin/nvidia/ >/dev/null
for _ in $(seq 1 60); do
  n=$($K get resourceslices -o json 2>/dev/null | python3 -c 'import json,sys; print(sum(len(s["spec"].get("devices",[])) for s in json.load(sys.stdin)["items"]))' 2>/dev/null || echo 0)
  [ "$n" -ge "$COUNT" ] && break
  sleep 10
done
echo "  ResourceSlice devices: $n"

log "OpenRL bundle from $OVERLAY"
$K create namespace openrl-system --dry-run=client -o yaml | $K apply -f - >/dev/null
$K apply --server-side --force-conflicts -k "$OVERLAY" >/dev/null
$K -n openrl-system rollout status deployment/open-rl-api-server --timeout=10m
$K -n openrl-system rollout status deployment/open-rl-scheduler --timeout=5m
$K -n openrl-system rollout status daemonset/open-rl-worker-prepull --timeout=15m || true

log "cell $NAME is up"
$K get nodes -L cloud.google.com/gke-accelerator,openrl.io/trainer,openrl.io/sampler
$K -n openrl-system get pods
cat <<EOF

Next:
  scripts/cell_seed_hf.sh $CTX            # copy the model and dataset cache into this cell's volume
  export KCTX=$CTX                          # then run warm-ups and drills with kubectl --context \$KCTX
  scripts/cell_down.sh $NAME $REGION      # when done
EOF
