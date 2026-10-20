#!/usr/bin/env bash
# Copy the Hugging Face cache (models and datasets) from the main cluster's
# shared volume into a drill cell's volume, through a GCS bucket. The cell's
# nodes can read GCS (devstorage.read_only) but not write it, so the upload
# streams through this machine; the download runs inside the cell.
#
#   scripts/cell_seed_hf.sh <cell kube context> [source context] [models...]
#
# Defaults: source gke_cdrollouts-sunilarora_us-central1_open-rl-dra; models
# google/gemma-4-e2b and Qwen/Qwen3-0.6B plus the gsm8k dataset. Reruns skip
# the upload when the archive is already in the bucket.
set -euo pipefail
DST=${1:?cell context}
SRC=${2:-gke_cdrollouts-sunilarora_us-central1_open-rl-dra}
shift 2 2>/dev/null || shift $#
MODELS=${*:-"google/gemma-4-e2b Qwen/Qwen3-0.6B"}
PROJECT=${PROJECT:-cdrollouts-sunilarora}
BUCKET=${BUCKET:-gs://$PROJECT-openrl-cache}
HF=/mnt/shared/open-rl/huggingface
cd "$(dirname "$0")/.."

gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1 || gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location US --uniform-bucket-level-access
paths="datasets hub/datasets--openai--gsm8k"
for m in $MODELS; do paths="$paths hub/models--${m//\//--}"; done

archive="$BUCKET/hf-cache.tgz"
if gcloud storage ls "$archive" >/dev/null 2>&1; then
  echo "archive exists: $archive (delete it to re-upload)"
else
  echo "streaming $paths from $SRC to $archive"
  # shellcheck disable=SC2086
  kubectl --context "$SRC" -n openrl-system exec deploy/open-rl-api-server -- sh -c "cd $HF && tar czf - $paths" \
    | gcloud storage cp - "$archive"
fi

echo "unpacking inside $DST"
kubectl --context "$DST" -n openrl-system delete job hf-seed --ignore-not-found >/dev/null
cat <<EOF | kubectl --context "$DST" -n openrl-system apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: hf-seed
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: seed
        image: google/cloud-sdk:slim
        command: ["sh", "-c"]
        args:
        - |
          set -e
          mkdir -p $HF && cd $HF
          gcloud storage cp $archive - | tar xzf -
          du -sh $HF/hub/* $HF/datasets 2>/dev/null
        volumeMounts:
        - name: shared
          mountPath: /mnt/shared
      volumes:
      - name: shared
        persistentVolumeClaim:
          claimName: open-rl-shared-pvc
EOF
kubectl --context "$DST" -n openrl-system wait --for=condition=complete job/hf-seed --timeout=60m
kubectl --context "$DST" -n openrl-system logs job/hf-seed | tail -8
