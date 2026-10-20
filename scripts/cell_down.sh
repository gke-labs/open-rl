#!/usr/bin/env bash
# Tear down a drill cell: the OpenRL namespace first (so the Filestore
# instance behind open-rl-shared-pvc is released), then the cluster.
#
#   scripts/cell_down.sh <name> <region>
set -euo pipefail
NAME=${1:?name}; REGION=${2:?region}
PROJECT=${PROJECT:-cdrollouts-sunilarora}
CTX="gke_${PROJECT}_${REGION}_${NAME}"
if kubectl --context "$CTX" get ns openrl-system >/dev/null 2>&1; then
  kubectl --context "$CTX" delete ns openrl-system --wait=true --timeout=10m || true
fi
gcloud container clusters delete "$NAME" --location "$REGION" --project "$PROJECT" --quiet
kubectl config delete-context "$CTX" >/dev/null 2>&1 || true
echo "cell $NAME removed. Check for a leftover Filestore instance:"
echo "  gcloud filestore instances list --project $PROJECT --filter=\"zone~$REGION\""
