#!/usr/bin/env bash
#
# Create the kind cluster and install the DRA driver for NVIDIA GPUs.
#
# Run ./dev/kind/host-setup.sh first. Re-running this deletes and recreates the
# cluster, which is the fastest way to get back to a clean claim/slice state.
#
# Usage:
#   ./dev/kind/create-cluster.sh
#   KEEP_COMPUTE_DOMAINS=1 ./dev/kind/create-cluster.sh   # on NVLink hardware
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLUSTER_NAME="${KIND_CLUSTER_NAME:-open-rl-dra}"

# Pinned so a cluster rebuild does not silently pick up a new driver. Chart and
# docs: https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/install/
DRA_CHART="${DRA_CHART:-oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu}"
DRA_VERSION="${DRA_VERSION:-0.4.1}"
DRA_NAMESPACE="dra-driver-nvidia-gpu"

log() { echo "[create-cluster] $*"; }

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  log "Deleting existing cluster '$CLUSTER_NAME'..."
  kind delete cluster --name "$CLUSTER_NAME"
fi

log "Creating cluster '$CLUSTER_NAME'..."
kind create cluster --config "$SCRIPT_DIR/kind-cluster.yaml" --name "$CLUSTER_NAME"
kubectl config use-context "kind-${CLUSTER_NAME}"

log "Confirming GPUs reached the node container..."
docker exec "${CLUSTER_NAME}-control-plane" nvidia-smi -L

# ComputeDomains target Multi-Node NVLink. L4s have none, so the ComputeDomain
# resources never allocate and the extra plugin container only adds noise.
COMPUTE_DOMAINS="${KEEP_COMPUTE_DOMAINS:+true}"
COMPUTE_DOMAINS="${COMPUTE_DOMAINS:-false}"

log "Installing the DRA driver (chart $DRA_VERSION, computeDomains=$COMPUTE_DOMAINS)..."
helm install dra-driver-nvidia-gpu "$DRA_CHART" \
  --version "$DRA_VERSION" \
  --create-namespace \
  --namespace "$DRA_NAMESPACE" \
  --set gpuResourcesEnabledOverride=true \
  --set resources.gpus.enabled=true \
  --set "resources.computeDomains.enabled=${COMPUTE_DOMAINS}" \
  --set nvidiaDriverRoot=/ \
  --wait --timeout 5m

log "Waiting for the kubelet plugin to publish ResourceSlices..."
slices_ready=""
for _ in $(seq 1 60); do
  if [[ -n "$(kubectl get resourceslices -o name 2>/dev/null)" ]]; then
    slices_ready=yes
    break
  fi
  sleep 5
done

# Fail loudly. A cluster with zero slices looks healthy -- the DeviceClasses are
# registered and the pods, if any, are Running -- but every claim the gateway
# creates will sit Pending with nothing in its logs to say why.
if [[ -z "$slices_ready" ]]; then
  echo "No ResourceSlices published after 5m." >&2
  kubectl -n "$DRA_NAMESPACE" get ds,pods >&2
  echo >&2
  echo "If the kubelet-plugin DaemonSet shows DESIRED 0, its required nodeAffinity" >&2
  echo "matched nothing: it wants a Node Feature Discovery label, and NFD is not" >&2
  echo "installed here. kind-cluster.yaml sets nvidia.com/gpu.present=true for this" >&2
  echo "reason -- confirm it survived with:" >&2
  echo "  kubectl get node ${CLUSTER_NAME}-control-plane -o jsonpath='{.metadata.labels}'" >&2
  exit 1
fi

echo
log "DeviceClasses:"
kubectl get deviceclasses
echo
log "ResourceSlices:"
kubectl get resourceslices -o custom-columns=NAME:.metadata.name,DRIVER:.spec.driver,NODE:.spec.nodeName

if ! kubectl get deviceclass gpu.nvidia.com >/dev/null 2>&1; then
  echo "DeviceClass gpu.nvidia.com is missing -- the scheduler hardcodes it. Check:" >&2
  echo "  kubectl -n $DRA_NAMESPACE get pods" >&2
  echo "  kubectl -n $DRA_NAMESPACE logs -l app.kubernetes.io/name=dra-driver-nvidia-gpu --all-containers" >&2
  exit 1
fi

cat <<EOF

Cluster ready. Next:
  ./dev/kind/load-images.sh          # build + side-load the open-rl images
  kubectl apply -k k8s/deploy/kind-dra/
  kubectl apply -f dev/kind/synthetic-h100-resourceslice.yaml   # optional, for 80gb-tier tests
EOF
