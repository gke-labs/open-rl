#!/usr/bin/env bash
# One cold-start measurement, end to end: steer placement with node labels,
# clear old workers, record, run the demo client, collect, build the timeline.
#
#   scripts/cold_start_case.sh <case-dir> <base_model> <trainer-node-suffixes> <sampler-node-suffixes>
#   scripts/cold_start_case.sh runs/2026-10-15_cold_start/A-0.5b-l4-cold Qwen/Qwen2.5-0.5B lvw8 2208
#
# Node suffixes name the nodes allowed for each role; every other node in the
# trainer or sampler pools gets openrl.io/<role>=false for the run. Labels are
# restored to the values in <case-dir>/../node-labels-before.json afterwards
# only by cold_start_restore_labels.sh, so consecutive cases can reuse them.
set -euo pipefail
CASE_DIR=$1; MODEL=$2; TRAINERS=$3; SAMPLERS=$4
NS=${OPENRL_NAMESPACE:-openrl-system}
BASE_URL=${BASE_URL:-http://127.0.0.1:9003}
cd "$(dirname "$0")/.."
mkdir -p "$CASE_DIR"

allow() { # role, allowed suffixes
  local role=$1 allowed=$2
  for node in $(kubectl get nodes -l "openrl.io/enabled=true" -o name | sed 's|node/||'); do
    kubectl get node "$node" -o jsonpath="{.metadata.labels.openrl\.io/$role}" | grep -q . || continue
    local want=false
    for s in $allowed; do case "$node" in *"$s") want=true;; esac; done
    kubectl label node "$node" "openrl.io/$role=$want" --overwrite >/dev/null
  done
}
allow trainer "$TRAINERS"
allow sampler "$SAMPLERS"
kubectl get nodes -l openrl.io/enabled=true -L openrl.io/trainer,openrl.io/sampler --no-headers > "$CASE_DIR/node-labels-during.txt"

# Fresh pods: release whatever the previous case left, wait for the pods to go.
kubectl -n "$NS" delete workloads --all --ignore-not-found >/dev/null
for _ in $(seq 1 60); do
  kubectl -n "$NS" get pods --no-headers 2>/dev/null | grep -q '^orw-' || break
  sleep 2
done
# The API server forgets the owner on its next reaper sweep; give it one.
sleep 35

python3 scripts/cold_start_timeline.py watch --out "$CASE_DIR"
sleep 3
set +e
uv --project examples run python scripts/cold_start_client.py base_model="$MODEL" base_url="$BASE_URL" out="$CASE_DIR/client.jsonl" > "$CASE_DIR/client.log" 2>&1
echo "EXIT=$?" >> "$CASE_DIR/client.log"
set -e
python3 scripts/cold_start_timeline.py collect --out "$CASE_DIR"
python3 scripts/cold_start_timeline.py timeline --out "$CASE_DIR" > /dev/null
sed -n '/Critical path/,$p' "$CASE_DIR/timeline.md"
