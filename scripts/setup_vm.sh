#!/usr/bin/env bash
# One-command setup: bare Ubuntu GPU VM -> ready to run scripts/launch_work.sh.
#
#   ./scripts/setup_vm.sh
#
# Idempotent — safe to re-run any time; it doubles as a health check. Ends with
# a green/red checklist. Handles: apt build deps, uv, CUDA toolchain discovery,
# the Python env (core sync is compiler-proof; the causal-conv1d fast path is
# attempted separately and warns instead of failing), the LAB checkout with
# sandbox + pandoc + podman, and disk-space sanity for podman storage.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB_ROOT="$REPO/examples/harvey_labs/harvey-labs"
cd "$REPO"

FAILED=()
step() { echo; echo "==> $*"; }
ok()   { echo "    [ok] $*"; }
bad()  { echo "    [MISSING] $*"; FAILED+=("$*"); }

SUDO=""
if [ "$(id -u)" != "0" ]; then SUDO="sudo"; fi

# --- 1. apt build deps -------------------------------------------------------
step "apt packages (build-essential, python3-dev, ninja-build, tmux, git, curl, redis-server)"
PKGS="build-essential python3-dev ninja-build tmux git curl redis-server"
MISSING_PKGS=""
for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING_PKGS="$MISSING_PKGS $p"; done
if [ -n "$MISSING_PKGS" ]; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $MISSING_PKGS && ok "installed:$MISSING_PKGS" || bad "apt install$MISSING_PKGS"
else
  ok "all present"
fi

# --- 2. uv -------------------------------------------------------------------
step "uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 && ok "$(uv --version)" || bad "uv install"
grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# --- 3. CUDA toolchain -------------------------------------------------------
step "CUDA toolchain (nvcc >= 12.6 for kernel JIT and the conv1d fast path)"
NVCC=""
for cand in "${CUDA_HOME:-}/bin/nvcc" "$(command -v nvcc 2>/dev/null)" /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
  [ -x "$cand" ] && NVCC="$cand" && break
done
if [ -n "$NVCC" ]; then
  export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
  export PATH="$CUDA_HOME/bin:$PATH"
  NVCC_VER=$("$NVCC" --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
  MAJOR=${NVCC_VER%%.*}; MINOR=${NVCC_VER##*.}
  if [ "$MAJOR" -gt 12 ] || { [ "$MAJOR" -eq 12 ] && [ "$MINOR" -ge 6 ]; }; then
    ok "nvcc $NVCC_VER at $NVCC (CUDA_HOME=$CUDA_HOME)"
    grep -q "CUDA_HOME=$CUDA_HOME" ~/.bashrc 2>/dev/null || {
      echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
      echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
    }
  else
    NVCC=""  # too old — fall through to the auto-install below
    echo "    nvcc $NVCC_VER is too old (< 12.6) — installing a current toolkit alongside it..."
  fi
fi
if [ -z "$NVCC" ]; then
  # Install the CUDA toolkit (only — the driver is untouched; toolkits
  # coexist and CUDA_HOME picks). The fast path is a hard requirement.
  UBU=$(. /etc/os-release && echo "${VERSION_ID%%.*}04" | sed 's/^/ubuntu/;s/04$/04/')
  case "$UBU" in ubuntu2204|ubuntu2404) ;; *) UBU=ubuntu2404 ;; esac
  echo "    installing cuda-toolkit-12-9 for $UBU (toolkit only, driver untouched)..."
  TMPDEB=$(mktemp /tmp/cuda-keyring-XXXX.deb)
  if curl -fsSL -o "$TMPDEB" "https://developer.download.nvidia.com/compute/cuda/repos/$UBU/x86_64/cuda-keyring_1.1-1_all.deb" \
    && $SUDO dpkg -i "$TMPDEB" >/dev/null \
    && $SUDO apt-get update -qq \
    && $SUDO apt-get install -y -qq cuda-toolkit-12-9 > /tmp/cuda-install.log 2>&1; then
    export CUDA_HOME=/usr/local/cuda-12.9
    export PATH="$CUDA_HOME/bin:$PATH"
    NVCC="$CUDA_HOME/bin/nvcc"
    ok "installed nvcc $("$NVCC" --version | grep -oE 'release [0-9.]+' ) at $NVCC"
    grep -q "CUDA_HOME=$CUDA_HOME" ~/.bashrc 2>/dev/null || {
      echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
      echo 'export PATH="$CUDA_HOME/bin:$PATH"' >> ~/.bashrc
    }
  else
    bad "CUDA toolkit install failed (log: /tmp/cuda-install.log) — the deltanet fast path cannot build without nvcc >= 12.6"
  fi
fi

# --- 4. python env: core sync (must succeed), fast path (may not) ------------
step "Python env sync (gpu + vllm + fastpath)"
if ! grep -q '^fastpath' pyproject.toml; then
  if uv sync --frozen --exact --extra gpu --extra vllm > /tmp/uv-sync.log 2>&1; then
    ok "core sync (conv1d ships inside the gpu extra on this branch)"
  else
    tail -5 /tmp/uv-sync.log; bad "core uv sync (full log: /tmp/uv-sync.log)"
  fi
elif uv sync --frozen --exact --extra gpu --extra vllm --extra fastpath > /tmp/uv-sync.log 2>&1; then
  ok "full sync incl. conv1d fast path"
elif uv sync --frozen --exact --extra gpu --extra vllm > /tmp/uv-sync.log 2>&1; then
  ok "core sync"
  bad "causal-conv1d build (log: /tmp/uv-sync.log) — without it training runs the eager deltanet fallback (2-5x slower)"
else
  tail -5 /tmp/uv-sync.log; bad "core uv sync (full log: /tmp/uv-sync.log)"
fi

# --- 5. LAB checkout + sandbox -----------------------------------------------
step "LAB harness (checkout, venv, pandoc, podman, sandbox image)"
if [ -d "$LAB_ROOT/.git" ]; then
  git -C "$LAB_ROOT" pull --ff-only >/dev/null 2>&1 || echo "    [WARN] LAB checkout pull failed (local changes?)"
  ok "LAB checkout at $(git -C "$LAB_ROOT" log --oneline -1 -- evaluation/judge.py)"
else
  ./examples/harvey_labs/setup_lab.sh && ok "LAB bootstrapped" || bad "setup_lab.sh"
fi

# --- 6. disk space for podman + runs -----------------------------------------
step "disk space"
GRAPHROOT=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo "$HOME/.local/share/containers/storage")
for path in "$REPO" "$GRAPHROOT"; do
  [ -e "$path" ] || path=$(dirname "$path")
  AVAIL_GB=$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ "${AVAIL_GB:-0}" -ge 30 ]; then
    ok "$path: ${AVAIL_GB}G free"
  else
    bad "$path has only ${AVAIL_GB:-?}G free (< 30G). Podman layers + run results will exhaust it.
      Move container storage to a big disk: set graphroot in ~/.config/containers/storage.conf, then podman system reset.
      Prune old runs: rm -rf <lab_root>/results/<old-run-ids>"
  fi
done

# --- 7. final checklist ------------------------------------------------------
step "verification"
uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && ok "torch sees the GPU" || bad "torch.cuda.is_available()"
FP=$(uv run --no-sync python -c "from transformers.models.qwen3_5 import modeling_qwen3_5 as m; print(m.is_fast_path_available)" 2>/dev/null)
if [ "$FP" = "True" ]; then ok "Qwen deltanet fast path available"; else bad "Qwen deltanet fast path unavailable — training would run 2-5x slower"; fi
timeout 90 podman run --rm ghcr.io/harveyai/lab-sandbox:latest echo ok >/dev/null 2>&1 && ok "podman sandbox runs" || bad "podman run lab-sandbox (try: podman system migrate; check /etc/subuid)"
[ -x "$LAB_ROOT/.venv/bin/python" ] && "$LAB_ROOT/.venv/bin/python" -c "import sys; sys.path.insert(0,'$LAB_ROOT'); from evaluation.judge import Judge; Judge._salvage_verdict" 2>/dev/null && ok "LAB grading env (judge + salvage fix)" || bad "LAB venv judge import — rerun setup_lab.sh / git pull the LAB checkout"
if [ -n "${VERTEX_JUDGE_ENDPOINT:-}" ]; then
  "$LAB_ROOT/.venv/bin/python" -c "import google.cloud.aiplatform, transformers" 2>/dev/null \
    || "$LAB_ROOT/.venv/bin/pip" install -q google-cloud-aiplatform transformers
  "$LAB_ROOT/.venv/bin/python" -c "import google.cloud.aiplatform, transformers, google.auth; google.auth.default()" 2>/dev/null \
    && ok "GLM judge ready (Vertex deps + ADC)" \
    || bad "GLM judge — ADC missing or deps failed (gcloud auth application-default login, or GOOGLE_APPLICATION_CREDENTIALS)"
else
  [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ] && ok "Gemini judge key present" || echo "    [WARN] export GEMINI_API_KEY before training — grading needs it (or use the GLM judge: set VERTEX_JUDGE_ENDPOINT + JUDGE_MODEL=glm-5.2)"
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "SETUP COMPLETE — launch the stack with: ./scripts/launch_work.sh  (MODEL=9b|27b)"
else
  echo "SETUP INCOMPLETE — fix the [MISSING] items above and re-run. Failed: ${#FAILED[@]}"
  exit 1
fi
