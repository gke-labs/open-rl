#!/usr/bin/env bash
# Fetch the Harvey LAB benchmark and bootstrap everything the recipe needs.
#
# Clones the LAB fork (upstream + harness fixes) into lab_root, then
# delegates to LAB's own scripts/setup.sh, which idempotently installs
# uv, the harness's Python deps, pandoc, podman (including the VM on
# macOS/Windows), and the sandbox image.
#
# Runnable from any directory; the checkout lands next to this script
# (matching the recipe's lab_root default) unless LAB_ROOT overrides it.
#   examples/harvey_labs/setup_lab.sh
#   LAB_ROOT=/data/harvey-labs examples/harvey_labs/setup_lab.sh
#   LAB_REPO=https://github.com/<you>/harvey-labs LAB_REF=main examples/harvey_labs/setup_lab.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_REPO=${LAB_REPO:-https://github.com/ShubyM/harvey-labs}
LAB_REF=${LAB_REF:-main}
LAB_ROOT=${LAB_ROOT:-${SCRIPT_DIR}/harvey-labs}

if [ -d "${LAB_ROOT}/.git" ]; then
  echo "[setup-lab] existing checkout at ${LAB_ROOT} — leaving its revision alone"
  echo "[setup-lab] (update with: git -C ${LAB_ROOT} pull)"
else
  echo "[setup-lab] cloning ${LAB_REPO}@${LAB_REF} -> ${LAB_ROOT}"
  git clone --branch "${LAB_REF}" "${LAB_REPO}" "${LAB_ROOT}"
fi

cd "${LAB_ROOT}"
exec ./scripts/setup.sh
