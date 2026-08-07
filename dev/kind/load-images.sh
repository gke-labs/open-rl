#!/usr/bin/env bash
#
# Build the open-rl images on this host and side-load them into the kind cluster.
#
# This is the reason to run kind on the GPU VM at all: the multi-GB CUDA server
# image never touches a registry. `kind load` copies it straight from the local
# Docker daemon into the node's containerd store, so an edit-to-running-pod loop
# costs a docker build layer, not a push plus a pull.
#
# The manifests already set imagePullPolicy: IfNotPresent, so a side-loaded image
# is used as-is and the kubelet never reaches out to gcr.io.
#
# Usage:
#   ./dev/kind/load-images.sh                # gateway + server
#   ./dev/kind/load-images.sh gateway        # just the fast one
#   IMAGE_TAG=wip ./dev/kind/load-images.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLUSTER_NAME="${KIND_CLUSTER_NAME:-open-rl-dra}"
IMAGE_TAG="${IMAGE_TAG:-kind-dev}"
# Local-only names. Kept distinct from the gcr.io names so a side-loaded dev
# build can never be confused with something that was actually pushed.
REGISTRY="${REGISTRY:-open-rl.local}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(gateway server)
fi

log() { echo "[load-images] $*"; }

for target in "${TARGETS[@]}"; do
  case "$target" in
    gateway) dockerfile="src/server/Dockerfile.gateway" ;;
    server) dockerfile="src/server/Dockerfile" ;;
    client) dockerfile="src/server/Dockerfile.client" ;;
    *)
      echo "Unknown target '$target'. Expected gateway, server, or client." >&2
      exit 2
      ;;
  esac

  image="${REGISTRY}/open-rl-${target}:${IMAGE_TAG}"
  log "Building $image..."
  DOCKER_BUILDKIT=1 docker build -t "$image" -f "$REPO_ROOT/$dockerfile" "$REPO_ROOT"

  log "Loading $image into kind cluster '$CLUSTER_NAME'..."
  kind load docker-image "$image" --name "$CLUSTER_NAME"
done

cat <<EOF

Loaded. Deploy with:
  kubectl apply -k k8s/deploy/kind-dra/

If the cluster is already running, restart to pick up the new build:
  kubectl rollout restart deployment/open-rl-gateway
EOF
