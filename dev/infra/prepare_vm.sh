#!/bin/bash
# prepare_vm.sh - Automates setup of a fresh GCP VM

set -e

if [ -z "$1" ]; then
    echo "Usage: ./prepare_vm.sh <REMOTE_HOST>"
    echo "Example: ./prepare_vm.sh b3"
    exit 1
fi

REMOTE_HOST=$1

echo "--- Configuring VM: $REMOTE_HOST ---"

# 1. Sync local .tmux.conf
if [ -f "$HOME/.tmux.conf" ]; then
    echo "[1/4] Syncing .tmux.conf..."
    scp "$HOME/.tmux.conf" "${REMOTE_HOST}:~/.tmux.conf"
else
    echo "[1/4] Skipping .tmux.conf (not found locally)."
fi

# 2. Install tmux, build-essential, and uv
echo "[2/4] Installing tmux, build-essential, and uv on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "sudo apt-get update && sudo apt-get install -y tmux build-essential && curl -LsSf https://astral.sh/uv/install.sh | sh"

# 3. Update PATH in .profile
echo "[3/4] Updating PATH in ~/.profile on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "grep -q \"/.local/bin\" ~/.profile || echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.profile"

# 4. Sync Ghostty terminfo (if applicable)
GHOSTTY_TERMINFO="/Applications/Ghostty.app/Contents/Resources/terminfo"
if [ -d "$GHOSTTY_TERMINFO" ]; then
    echo "[4/4] Syncing Ghostty terminfo..."
    TERMINFO="$GHOSTTY_TERMINFO" infocmp -x xterm-ghostty | ssh "$REMOTE_HOST" tic -x -
else
    echo "[4/4] Skipping Ghostty terminfo sync (Ghostty not found locally)."
fi

echo "--- Setup Complete! ---"
echo "You may need to run 'source ~/.profile' or log out and back in to use 'uv'."
