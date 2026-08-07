#!/usr/bin/env bash
#
# Start the cluster-local image registry and attach it to the kind network.
#
# Idempotent: safe to re-run, and create-cluster.sh calls it. Split into its own
# script because the registry outlives the cluster -- deleting and recreating
# kind is the fastest way back to a clean claim state, and it would be a waste
# to re-push 36GB of CUDA image every time it happens.
set -euo pipefail

REGISTRY_NAME="${KIND_REGISTRY_NAME:-kind-registry}"
REGISTRY_PORT="${KIND_REGISTRY_PORT:-5001}"
KIND_NETWORK="${KIND_NETWORK:-kind}"

log() { echo "[registry] $*"; }

if [[ "$(docker inspect -f '{{.State.Running}}' "$REGISTRY_NAME" 2>/dev/null || true)" != "true" ]]; then
  log "Starting $REGISTRY_NAME on 127.0.0.1:${REGISTRY_PORT}..."
  # Bound to loopback: this holds unreviewed dev builds and the VM has a public
  # interface. Nothing outside the host needs to reach it -- the node container
  # reaches it over the kind docker network, not through this published port.
  docker run -d --restart=always \
    -p "127.0.0.1:${REGISTRY_PORT}:5000" \
    --name "$REGISTRY_NAME" \
    registry:2 >/dev/null
else
  log "$REGISTRY_NAME already running."
fi

# The network only exists once kind has created a cluster. Connecting is what
# lets the node resolve the mirror endpoint by container name.
if docker network inspect "$KIND_NETWORK" >/dev/null 2>&1; then
  if ! docker network inspect "$KIND_NETWORK" -f '{{range .Containers}}{{.Name}} {{end}}' | grep -qw "$REGISTRY_NAME"; then
    log "Connecting $REGISTRY_NAME to the '$KIND_NETWORK' network..."
    docker network connect "$KIND_NETWORK" "$REGISTRY_NAME"
  fi
else
  log "Network '$KIND_NETWORK' does not exist yet; it is created with the cluster."
fi

log "Ready at localhost:${REGISTRY_PORT}"
