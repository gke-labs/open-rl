#!/usr/bin/env bash
# Bring up the Harvey-LAB RL stack in one tmux session: vLLM sampler on the
# non-trainer GPUs, gateway (+ trainer), and typed train/eval commands.
#
#   MODEL=9b  ./scripts/launch_work.sh                # Qwen3.5-9B (default)
#   TRAIN_GPUS=4 MODEL=27b ./scripts/launch_work.sh   # data-parallel LoRA trainer on 4 GPUs
#
# Overridable: MODEL, TRAIN_GPUS, RUN_LABEL, GEN_TOKENS, JUDGE_MODEL.
set -euo pipefail

SESSION=work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB_ROOT="$REPO/examples/harvey_labs/harvey-labs"
LOGS="$REPO/artifacts/box-logs"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running — attaching (tmux kill-session -t $SESSION to reset)"
  if [ -t 1 ]; then
    exec tmux attach -t "$SESSION"
  else
    exit 0
  fi
fi

JUDGE_MODEL=${JUDGE_MODEL:-gemini-3.5-flash}
JUDGE_ENV=""
case "$JUDGE_MODEL" in
  glm*)
    if [ -z "${VERTEX_JUDGE_ENDPOINT:-}" ]; then
      echo "ERROR: JUDGE_MODEL=$JUDGE_MODEL needs the Vertex endpoint:" >&2
      echo "  export VERTEX_JUDGE_ENDPOINT=projects/<project>/locations/<region>/endpoints/<id>" >&2
      exit 1
    fi
    VERTEX_JUDGE_TOKENIZER=${VERTEX_JUDGE_TOKENIZER:-zai-org/GLM-5.2-FP8}
    JUDGE_ENV="VERTEX_JUDGE_ENDPOINT=$VERTEX_JUDGE_ENDPOINT VERTEX_JUDGE_TOKENIZER=$VERTEX_JUDGE_TOKENIZER"
    ;;
  *)
    if [ -z "${GEMINI_API_KEY:-}" ]; then
      echo "WARNING: GEMINI_API_KEY is not set — rubric grading will fail without it." >&2
    fi
    ;;
esac
mkdir -p "$LOGS"
cd "$REPO"

# Podman layers + per-episode results exhaust small disks mid-run.
GRAPHROOT=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo "$HOME/.local/share/containers/storage")
for path in "$REPO" "$GRAPHROOT"; do
  [ -e "$path" ] || path=$(dirname "$path")
  AVAIL_GB=$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ "${AVAIL_GB:-0}" -lt 20 ]; then
    echo "ERROR: only ${AVAIL_GB:-?}G free on $path (< 20G). Free space before training:" >&2
    echo "  old runs:      rm -rf $LAB_ROOT/results/<old-run-id>" >&2
    echo "  podman layers: move graphroot in ~/.config/containers/storage.conf to a big disk" >&2
    exit 1
  fi
done

# Reap sandbox containers leaked by crashed episodes.
LEAKED=$(podman ps -a --filter name=lab-sandbox --format '{{.Names}}' 2>/dev/null)
if [ -n "$LEAKED" ]; then
  echo "[work] removing leaked sandbox containers: $LEAKED"
  echo "$LEAKED" | xargs -r podman rm -f >/dev/null
fi

if [ ! -d "$LAB_ROOT" ]; then
  echo "[work] LAB checkout missing — running setup_lab.sh (clones the fork, installs pandoc/podman)..."
  ./examples/harvey_labs/setup_lab.sh
fi
echo "[work] LAB judge at: $(git -C "$LAB_ROOT" log --oneline -1 -- evaluation/judge.py 2>/dev/null || echo 'unknown')"

if [[ "$JUDGE_MODEL" == glm* ]]; then
  if ! "$LAB_ROOT/.venv/bin/python" -c "import google.cloud.aiplatform, transformers" 2>/dev/null; then
    echo "[work] installing GLM judge deps into the LAB venv..."
    "$LAB_ROOT/.venv/bin/pip" install -q google-cloud-aiplatform transformers
  fi
  if ! "$LAB_ROOT/.venv/bin/python" -c "import google.auth; google.auth.default()" 2>/dev/null; then
    echo "ERROR: no Application Default Credentials — the GLM judge cannot authenticate to Vertex." >&2
    echo "  Fix one of:" >&2
    echo "    gcloud auth application-default login --no-launch-browser" >&2
    echo "    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json" >&2
    echo "    (on a GCP VM: attach a service account with the Vertex AI User role)" >&2
    exit 1
  fi
  echo "[work] JUDGE_MODEL=$JUDGE_MODEL via $VERTEX_JUDGE_ENDPOINT (ADC ok)"
fi

FP=$(uv run --no-sync python -c "from transformers.models.qwen3_5 import modeling_qwen3_5 as m; print(m.is_fast_path_available)" 2>/dev/null)
if [ "$FP" != "True" ]; then
  echo "WARNING: Qwen deltanet fast path is NOT available — training runs the eager" >&2
  echo "         fallback (2-5x slower). Run ./scripts/setup_vm.sh to build causal-conv1d." >&2
fi

MODEL=${MODEL:-9b}
case "$MODEL" in
  9b)
    MODEL_NAME=Qwen/Qwen3.5-9B
    CONTEXT=262144
    GEN_TOKENS=${GEN_TOKENS:-32768}
    TASK_SET=${TASK_SET:-random}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen9b}
    ;;
  9b-128k)
    # Signal-hunting shape: big groups for GRPO contrast, the seeded random
    # 300/50 split for task diversity — and the 50-task eval's ~3,150
    # criteria cut eval noise to ~±1%, so small gains are detectable.
    MODEL_NAME=Qwen/Qwen3.5-9B
    CONTEXT=131072
    GEN_TOKENS=${GEN_TOKENS:-16384}
    TASK_SET=${TASK_SET:-random}
    BATCH_SIZE=${BATCH_SIZE:-8}
    ROLLOUTS=${ROLLOUTS:-6}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen9b-128k}
    ;;
  27b)
    MODEL_NAME=Qwen/Qwen3.5-27B
    CONTEXT=98304
    # 32K tool results in a 98K window would let a few parallel document
    # reads overflow the whole trajectory budget; 16K is the proven value.
    GEN_TOKENS=${GEN_TOKENS:-16384}
    # Curated run-3/4 task lists so 27B numbers stay comparable across runs.
    TASK_SET=${TASK_SET:-bootstrap}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen27b}
    ;;
  *)
    echo "Unknown MODEL=$MODEL (use 9b, 9b-128k, or 27b)" >&2
    exit 1
    ;;
esac
BATCH_SIZE=${BATCH_SIZE:-5}
ROLLOUTS=${ROLLOUTS:-2}
echo "[work] MODEL=$MODEL -> $MODEL_NAME, context $CONTEXT, batch ${BATCH_SIZE}x${ROLLOUTS}, log $RUN_LABEL"

# fla backend by arch: Hopper needs TileLang; Blackwell needs Triton.
GPU0=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU0" in
  *H100*|*H200*) FLA_TILELANG=${FLA_TILELANG:-1} ;;
  *)             FLA_TILELANG=${FLA_TILELANG:-0} ;;
esac
echo "[work] GPU: $GPU0 -> FLA_TILELANG=$FLA_TILELANG"

# TRAIN_GPUS>1: dedicated torchrun data-parallel trainer on GPUs 0..N-1.
TRAIN_GPUS=${TRAIN_GPUS:-1}
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); NUM_GPUS=${NUM_GPUS:-8}
SAMPLER_DP=$((NUM_GPUS - TRAIN_GPUS))
SAMPLER_DEV=$(seq -s, "$TRAIN_GPUS" $((NUM_GPUS - 1)))
TRAIN_DEV=$(seq -s, 0 $((TRAIN_GPUS - 1)))
QUEUE_ENV=""
if [ "$TRAIN_GPUS" -gt 1 ]; then
  # Ephemeral queue: no RDB snapshots or AOF — background persistence of
  # multi-MB training payloads is pure stall risk for zero value. Raise the
  # fd limit before daemonizing: redis derives maxclients from it, and every
  # pending gateway request holds one BLPOP connection.
  ulimit -n 65535 2>/dev/null || true
  pgrep -x redis-server >/dev/null || redis-server --daemonize yes --save '' --appendonly no --maxclients 8192
  QUEUE_ENV="REDIS_URL=redis://127.0.0.1:6379 OPEN_RL_EXTERNAL_TRAINER=1"
  echo "[work] TRAIN_GPUS=$TRAIN_GPUS -> torchrun trainer on GPUs $TRAIN_DEV, sampler DP$SAMPLER_DP on $SAMPLER_DEV"
fi

# AFFINITY=1: one vllm serve per sampler GPU with prefix-hash routing.
AFFINITY=${AFFINITY:-0}
if [ "$AFFINITY" = "1" ]; then
  SAMPLER_CMD=""
  SAMPLER_URLS=""
  for i in $(seq 0 $((SAMPLER_DP - 1))); do
    GPU_ID=$((TRAIN_GPUS + i))
    PORT=$((8000 + i))
    SAMPLER_URLS="$SAMPLER_URLS,http://127.0.0.1:$PORT"
    SAMPLER_CMD="$SAMPLER_CMD CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
uv run --extra gpu --extra vllm --extra fastpath vllm serve $MODEL_NAME \
--port $PORT --enable-lora --max-lora-rank 64 --max-loras 2 --enable-prefix-caching \
--max-model-len $CONTEXT --gpu-memory-utilization 0.92 \
--language-model-only |& tee -a $LOGS/sampler-$i.log &"
  done
  SAMPLER_CMD="${SAMPLER_CMD} wait"
  SAMPLER_URLS="${SAMPLER_URLS#,}"
  SAMPLER_ENV="SAMPLER_BASE_URLS=$SAMPLER_URLS"
  LAST_PORT=$((8000 + SAMPLER_DP - 1))
  SAMPLER_WAIT="until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && curl -sf http://127.0.0.1:$LAST_PORT/v1/models >/dev/null 2>&1; do echo 'waiting for samplers...'; sleep 10; done"
  echo "[work] AFFINITY=1 -> $SAMPLER_DP single-GPU samplers on ports 8000-$LAST_PORT"
else
  SAMPLER_ENV="SAMPLER_BASE_URL=http://127.0.0.1:8000"
  SAMPLER_WAIT="until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do echo 'waiting for sampler...'; sleep 10; done"
  SAMPLER_CMD="CUDA_VISIBLE_DEVICES=$SAMPLER_DEV VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
uv run --extra gpu --extra vllm --extra fastpath vllm serve $MODEL_NAME \
--port 8000 --enable-lora --max-lora-rank 64 --max-loras 2 --enable-prefix-caching \
--data-parallel-size $SAMPLER_DP --api-server-count 1 \
--max-model-len $CONTEXT --gpu-memory-utilization 0.92 \
--language-model-only |& tee -a $LOGS/sampler.log"
fi

GATEWAY_DEV=0
[ "$TRAIN_GPUS" -gt 1 ] && GATEWAY_DEV=""
GATEWAY_CMD="$SAMPLER_WAIT; \
CUDA_VISIBLE_DEVICES=$GATEWAY_DEV $QUEUE_ENV FLA_TILELANG=$FLA_TILELANG BASE_MODEL=$MODEL_NAME $SAMPLER_ENV \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OPEN_RL_TRAIN_TOKEN_BUDGET=$CONTEXT OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 \
OPEN_RL_LOG_CUDA_MEMORY=1 \
uv run --extra gpu --extra vllm --extra fastpath python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003 |& tee -a $LOGS/gateway.log"

TRAIN_CMD="TINKER_API_KEY=tml-dummy $JUDGE_ENV uv --project examples run python examples/harvey_labs/train.py \
model_name=$MODEL_NAME renderer_name=qwen3_5 base_url=http://127.0.0.1:9003 \
learning_rate=2e-4 lora_rank=32 \
batch_size=$BATCH_SIZE rollouts_per_example=$ROLLOUTS max_steps=20 eval_every=5 \
task_set=$TASK_SET judge_model=$JUDGE_MODEL \
max_tokens=$GEN_TOKENS max_trajectory_tokens=$CONTEXT max_tool_result_tokens=$GEN_TOKENS \
log_path=artifacts/harvey-labs/$RUN_LABEL"

EVAL_CMD="TINKER_API_KEY=tml-dummy $JUDGE_ENV uv --project examples run python examples/harvey_labs/eval_checkpoint.py \
checkpoint=/tmp/open-rl/peft/CHANGE-ME/final model_name=$MODEL_NAME renderer_name=qwen3_5 \
base_url=http://127.0.0.1:9003 task_set=$TASK_SET judge_model=$JUDGE_MODEL \
max_tokens=$GEN_TOKENS max_trajectory_tokens=$CONTEXT max_tool_result_tokens=$GEN_TOKENS"

# set-option needs a running tmux server, so the session must exist first.
tmux new-session -d -s "$SESSION" -n sampler -c "$REPO"
tmux set-option -t "$SESSION" history-limit 100000
tmux send-keys -t "$SESSION:sampler" "$SAMPLER_CMD" C-m

tmux new-window -t "$SESSION" -n gateway -c "$REPO"
tmux send-keys -t "$SESSION:gateway" "$GATEWAY_CMD" C-m

if [ "$TRAIN_GPUS" -gt 1 ]; then
  TRAINER_CMD="CUDA_VISIBLE_DEVICES=$TRAIN_DEV FLA_TILELANG=$FLA_TILELANG REDIS_URL=redis://127.0.0.1:6379 \
OPEN_RL_FSDP_WORLD_SIZE=$TRAIN_GPUS OPEN_RL_WORKER_PROBE_PORT=8090 \
BASE_MODEL=$MODEL_NAME PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OPEN_RL_TRAIN_TOKEN_BUDGET=$CONTEXT OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 \
OPEN_RL_LOG_CUDA_MEMORY=1 \
uv run --extra gpu --extra fastpath torchrun --standalone --nproc-per-node=$TRAIN_GPUS -m server.training_requests_processor |& tee -a $LOGS/trainer.log"
  tmux new-window -t "$SESSION" -n trainer -c "$REPO"
  tmux send-keys -t "$SESSION:trainer" "$TRAINER_CMD" C-m
fi

tmux new-window -t "$SESSION" -n train -c "$REPO"
tmux send-keys -t "$SESSION:train" "$TRAIN_CMD"          # typed, NOT run

tmux new-window -t "$SESSION" -n eval -c "$REPO"
tmux send-keys -t "$SESSION:eval" "$EVAL_CMD"            # typed, NOT run

tmux new-window -t "$SESSION" -n gpu -c "$REPO"
tmux send-keys -t "$SESSION:gpu" "watch -n 5 nvidia-smi" C-m

tmux select-window -t "$SESSION:train"
echo "[work] up. sampler+gateway starting; train/eval commands are typed and waiting."
if [ -t 1 ]; then
  exec tmux attach -t "$SESSION"
fi
