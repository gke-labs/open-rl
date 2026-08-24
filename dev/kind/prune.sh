#!/usr/bin/env bash
#
# Reclaim the disk a long iteration streak leaves behind.
#
# The kind loop republishes :kind-dev on every build. The tag moves, the blobs
# it used to point at do not: they stay in the registry, and the superseded
# image stays resident on the node as an untagged entry. Neither is collected
# on its own, so a day of rebuilding the 24GB server image quietly costs tens
# of gigabytes -- 24.3GB of registry for 11.9GB of live images, measured.
#
# The registry is rebuilt rather than garbage-collected. `registry
# garbage-collect` in distribution 2.8 does not traverse OCI image indexes, and
# an index is exactly what Docker's containerd store pushes: it keeps the
# tagged index, deletes the child manifests underneath it as unreferenced, and
# leaves a registry that answers 200 on the tag and 404 on everything it points
# at. Every pull then fails, including for images that were never stale. A
# rebuild plus re-push costs about 45s and cannot half-succeed.
#
# Deliberately does NOT touch the BuildKit cache. That cache is what makes a
# one-line source change rebuild in seconds instead of re-running `uv sync`.
# Pass PRUNE_BUILD_CACHE=1 if the disk is genuinely full and you want it anyway.
#
# Usage:
#   ./dev/kind/prune.sh
#   PRUNE_BUILD_CACHE=1 ./dev/kind/prune.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGISTRY_NAME="${KIND_REGISTRY_NAME:-kind-registry}"
REGISTRY="${REGISTRY:-localhost:5001}"
IMAGE_TAG="${IMAGE_TAG:-kind-dev}"
CLUSTER_NAME="${KIND_CLUSTER_NAME:-open-rl-dra}"
NODE="${CLUSTER_NAME}-control-plane"

log() { echo "[prune] $*"; }

registry_size() {
  docker exec "$REGISTRY_NAME" du -sh /var/lib/registry 2>/dev/null | cut -f1 || echo "?"
}

# --- registry ----------------------------------------------------------------
if [[ "$(docker inspect -f '{{.State.Running}}' "$REGISTRY_NAME" 2>/dev/null || true)" == "true" ]]; then
  # Only images the host can put back. Anything else would be destroyed with no
  # way to restore it, so leave the registry untouched instead.
  restorable=()
  missing=()
  for target in server gateway client; do
    image="${REGISTRY}/open-rl-${target}:${IMAGE_TAG}"
    if docker image inspect "$image" >/dev/null 2>&1; then
      restorable+=("$image")
    else
      missing+=("$image")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    log "Not rebuilding the registry: ${missing[*]} not present locally to re-push."
    log "Build them first (./dev/kind/load-images.sh) or accept the stale blobs."
  else
    before="$(registry_size)"
    log "Rebuilding the registry (was $before)..."
    # -v matters: registry:2 declares VOLUME /var/lib/registry, so the data sits
    # in an anonymous volume. Without it the old volume is orphaned with the
    # storage still allocated and the prune costs disk instead of reclaiming it.
    docker rm -f -v "$REGISTRY_NAME" >/dev/null
    "$SCRIPT_DIR/registry.sh" >/dev/null
    for image in "${restorable[@]}"; do
      log "Re-pushing $image..."
      docker push "$image" >/dev/null
    done
    log "Registry now $(registry_size) (was $before)."
  fi
else
  log "No $REGISTRY_NAME container running; skipping registry."
fi

# --- node images -------------------------------------------------------------
# Only untagged entries. `crictl rmi --prune` would also take images no pod
# happens to reference right now -- the client image between e2e runs, for
# instance -- and buy a multi-GB re-pull on the next one.
if docker inspect "$NODE" >/dev/null 2>&1; then
  untagged="$(docker exec "$NODE" crictl images -o json 2>/dev/null \
    | python3 -c 'import json,sys; print(" ".join(i["id"] for i in json.load(sys.stdin)["images"] if not i.get("repoTags")))' 2>/dev/null || true)"
  if [[ -n "${untagged// /}" ]]; then
    log "Removing $(wc -w <<<"$untagged") untagged image(s) from the node..."
    # shellcheck disable=SC2086
    docker exec "$NODE" crictl rmi $untagged >/dev/null 2>&1 || log "some node images were still in use; left them."
  else
    log "No untagged images on the node."
  fi
else
  log "No $NODE container; skipping node images."
fi

# --- host ---------------------------------------------------------------------
log "Removing dangling images on the host..."
docker image prune -f >/dev/null

if [[ -n "${PRUNE_BUILD_CACHE:-}" ]]; then
  log "Dropping the BuildKit cache -- the next build re-runs uv sync."
  docker builder prune -af >/dev/null
fi

log "Done. Host disk:"
df -h / | tail -1
