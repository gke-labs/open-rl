#!/usr/bin/env bash
#
# Build the open-rl images on this host and publish them to the cluster-local
# registry.
#
# This is the reason to run kind on the GPU VM at all: the multi-GB CUDA server
# image never leaves the machine. It goes to a registry container sitting on the
# kind docker network, so an edit-to-running-pod loop costs a docker build layer
# plus a layer push, not a push plus a pull across the internet.
#
# It used to use `kind load docker-image`, which was the wrong tool for an edit
# loop. That path is `docker save` streamed into `ctr images import` on the node,
# and it has no layer-level dedup against what the node already holds -- so a
# one-line source change re-shipped the whole 36GB server image and containerd
# re-unpacked every layer, about 24 minutes per iteration. `docker push` to a
# registry dedups by digest: only the changed source layer moves, and the
# kubelet pulls only that. Same iteration is now seconds.
#
# Usage:
#   ./dev/kind/load-images.sh                # gateway + server
#   ./dev/kind/load-images.sh gateway        # just the fast one
#   IMAGE_TAG=wip ./dev/kind/load-images.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-kind-dev}"
# The registry is published on the host loopback and mirrored into the node by
# create-cluster.sh, so this one reference resolves from both sides. Kept
# distinct from the gcr.io names so a local dev build can never be confused with
# something that was actually pushed.
REGISTRY="${REGISTRY:-localhost:5001}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(gateway server)
fi

log() { echo "[load-images] $*"; }

if ! curl -fsS "http://${REGISTRY}/v2/" >/dev/null 2>&1; then
  echo "No registry answering at ${REGISTRY}." >&2
  echo "It is started by dev/kind/create-cluster.sh; bring it back with:" >&2
  echo "  ./dev/kind/registry.sh" >&2
  exit 1
fi

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

  log "Pushing $image..."
  docker push "$image"
done

cat <<EOF

Pushed. Deploy with:
  kubectl apply -k k8s/deploy/kind-dra/

The manifests pin imagePullPolicy: Always against this registry. The tag does not
change between iterations, so IfNotPresent would leave the kubelet sitting on the
build it already has and silently test the previous code. Pulling from a registry
one docker network away costs nothing.

To pick up a new build in an already-running deployment:
  kubectl rollout restart deployment/open-rl-gateway
EOF
