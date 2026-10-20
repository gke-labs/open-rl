#!/usr/bin/env bash
# One command for a demo rehearsal: check the fleet is idle, trim the shared
# volume, optionally bring the shared LoRA runtimes up first, kick off the job
# set, and watch.
#
#   scripts/demo_rehearsal.sh <run-name> [--warm] [--jobs "<kind:model:count> ..."] [--steps N] [--context CTX]
#
#   scripts/demo_rehearsal.sh rehearsal-1 --warm
#   scripts/demo_rehearsal.sh big-1 --jobs "lora:Qwen/Qwen3-0.6B:6 fft:Qwen/Qwen3-0.6B:6 lora:google/gemma-4-e2b:6 fft:google/gemma-4-e2b:6"
#
# --warm starts one short job per LoRA base model and launches the set as soon
# as those finish, while the runtimes are still up (the API server reaps a
# runtime about two minutes after its last session ends). The warm-up jobs are
# labeled <run>-warm so the report can tell them apart.
#
# Storage: finished FFT jobs leave sampler weights and checkpoints behind, 2-10
# GB each. Entries under sampler_full and checkpoints older than RETAIN_MIN
# minutes (default 180) are removed before the run.
set -euo pipefail
RUN=${1:?run name}; shift
WARM=0; STEPS=3; CTX=${KCTX:-$(kubectl config current-context)}
JOBS="lora:Qwen/Qwen3-0.6B:3 fft:Qwen/Qwen3-0.6B:4 lora:google/gemma-4-e2b:3 fft:google/gemma-4-e2b:2"
while [ $# -gt 0 ]; do
  case $1 in
    --warm) WARM=1 ;;
    --jobs) JOBS=$2; shift ;;
    --steps) STEPS=$2; shift ;;
    --context) CTX=$2; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
IMAGE=${IMAGE:-us-central1-docker.pkg.dev/cdrollouts-sunilarora/open-rl/open-rl-client:demo1-18409bd}
NS=${OPENRL_NAMESPACE:-openrl-system}
RETAIN_MIN=${RETAIN_MIN:-180}
cd "$(dirname "$0")/.."
K="kubectl --context $CTX -n $NS"
log() { printf '\n== %s  %s\n' "$(date -u +%T)" "$*"; }

log "fleet on $CTX"
kubectl --context "$CTX" get nodes -l openrl.io/enabled=true -L cloud.google.com/gke-accelerator,openrl.io/trainer,openrl.io/sampler --no-headers | awk '{print "  " $1, $6, $7, $8}' | sed 's/gke-open-rl-dra-//'
busy=$($K get workloads --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [ "$busy" != 0 ]; then
  echo "  $busy workloads are still running; wait for the fleet to be idle:" >&2
  $K get workloads >&2
  exit 1
fi

log "shared volume"
$K exec deploy/open-rl-api-server -- sh -c "
  for d in /mnt/shared/open-rl/sampler_full /mnt/shared/open-rl/checkpoints; do
    [ -d \$d ] && find \$d -mindepth 1 -maxdepth 1 -mmin +$RETAIN_MIN -exec rm -rf {} + 2>/dev/null
  done
  df -h /mnt/shared | tail -1"

jobargs=()
for j in $JOBS; do jobargs+=(--job "$j"); done

if [ "$WARM" = 1 ]; then
  log "warm-up: one short job per LoRA base model"
  warmargs=()
  for j in $JOBS; do
    kind=${j%%:*}; rest=${j#*:}; model=${rest%:*}
    [ "$kind" = lora ] && warmargs+=(--job "lora:$model:1")
  done
  if [ ${#warmargs[@]} -gt 0 ]; then
    python3 scripts/demo_jobs.py --image "$IMAGE" --run "$RUN-warm" "${warmargs[@]}" --steps 1 | $K apply -f - >/dev/null
    for _ in $(seq 1 60); do
      left=$($K get jobs -l "openrl.io/demo-run=$RUN-warm" -o json | python3 -c 'import json,sys; j=json.load(sys.stdin)["items"]; print(sum(1 for x in j if not (x["status"].get("succeeded") or x["status"].get("failed"))))')
      [ "$left" = 0 ] && break
      sleep 10
    done
    echo "  runtimes up:"; $K get workloads --no-headers | awk '{print "  " $1, $2}' | grep lora || true
  fi
fi

log "kickoff $RUN: $JOBS, $STEPS steps"
python3 scripts/demo_jobs.py --image "$IMAGE" --run "$RUN" "${jobargs[@]}" --steps "$STEPS" | $K apply -f - | grep -c created
echo "  watch:  python3 scripts/demo_watch.py --run $RUN"
echo "  report: python3 scripts/demo_report.py --run $RUN --steps $STEPS --budget 180"
exec python3 scripts/demo_watch.py --run "$RUN" --namespace "$NS"
