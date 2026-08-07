#!/usr/bin/env bash
#
# One-time host prep for running a GPU-enabled kind cluster on a Linux GPU VM.
#
# Installs Docker, the NVIDIA Container Toolkit, kind, kubectl, and helm, then
# configures the toolkit so GPUs are visible *inside* kind's node containers.
# That last part is the whole trick: kind nodes are Docker containers, so the
# GPUs have to be injected into the node before the DRA driver running in the
# node can hand them to pods.
#
# Assumes the NVIDIA kernel driver is already installed (check with nvidia-smi).
#
# Usage:
#   ./dev/kind/host-setup.sh          # run ON the GPU VM
#   ssh l4dev 'bash -s' < dev/kind/host-setup.sh
#
# Idempotent: safe to re-run.
set -euo pipefail

log() { echo "[host-setup] $*"; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA kernel driver first." >&2
  exit 1
fi
log "GPUs detected:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'

# --- Docker -----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker..."
  curl -fsSL https://get.docker.com | sudo sh
fi
if ! id -nG "$USER" | grep -qw docker; then
  log "Adding $USER to the docker group (takes effect on next login)."
  sudo usermod -aG docker "$USER"
fi

# --- NVIDIA Container Toolkit -----------------------------------------------
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  log "Installing nvidia-container-toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
    sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

# Make "nvidia" the default Docker runtime. kind does not let us pass
# --runtime=nvidia when it creates node containers, so the default has to be it.
log "Configuring Docker to use the nvidia runtime by default..."
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default

# Let a volume mount at /var/run/nvidia-container-devices/all request every GPU.
# kind-cluster.yaml uses exactly that mount to pull the GPUs into the node
# container, and the toolkit ignores it unless this is switched on.
log "Enabling volume-mount device requests..."
sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true
sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-envvar-when-unprivileged=false

sudo systemctl restart docker

# The kubelet inside the node container resolves GPUs through /dev/char/<major>:<minor>
# symlinks. A bare-metal host does not create them, so cgroup device setup fails
# without this and pods hang in ContainerCreating.
log "Creating /dev/char symlinks for GPU devices..."
sudo nvidia-ctk system create-dev-char-symlinks --create-all

# --- kind / kubectl / helm --------------------------------------------------
ARCH="$(dpkg --print-architecture)"

if ! command -v kind >/dev/null 2>&1; then
  log "Installing kind..."
  curl -fsSLo /tmp/kind "https://kind.sigs.k8s.io/dl/latest/kind-linux-${ARCH}"
  sudo install -m 0755 /tmp/kind /usr/local/bin/kind && rm -f /tmp/kind
fi

if ! command -v kubectl >/dev/null 2>&1; then
  log "Installing kubectl..."
  KUBECTL_VERSION="$(curl -fsSL https://dl.k8s.io/release/stable.txt)"
  curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${ARCH}/kubectl"
  sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl && rm -f /tmp/kubectl
fi

if ! command -v helm >/dev/null 2>&1; then
  log "Installing helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | sudo bash
fi

log "Verifying the nvidia runtime reaches into containers..."
if sudo docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi -L; then
  log "Host is ready. Next: ./dev/kind/create-cluster.sh"
else
  echo "GPU passthrough check failed -- inspect 'docker info | grep -i runtime'." >&2
  exit 1
fi

# Note the missing "$USER": bare `id -nG` reports the groups this shell actually
# holds, whereas `id -nG "$USER"` reads the group database and would show docker
# the moment usermod ran -- never warning, even though this session still cannot
# reach the daemon.
if ! id -nG | grep -qw docker; then
  log "NOTE: this shell is not in the docker group yet. A fresh login picks it up;"
  log "      in the current session run 'newgrp docker'."
fi
