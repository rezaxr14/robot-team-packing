#!/usr/bin/env bash
# Isaac Sim 5.x + Isaac Lab 2.3 (Python 3.11, torch 2.7 cu128: supports Ada and Blackwell).
# Override with ISAACSIM_VERSION / ISAACLAB_REF if needed (Isaac Lab 3.0 needs Isaac Sim 6, Python 3.12, CUDA 13; not used here).
set -euo pipefail
source "$(dirname "$0")/env.sh"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.3.1}"
sudo apt-get update -y && sudo apt-get install -y cmake build-essential git
uv venv "$VENVS/isaac" --python 3.11 --seed          # --seed: isaaclab.sh needs pip inside the venv
activate isaac
uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
[ -d "$ISAACLAB_DIR" ] || git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
cd "$ISAACLAB_DIR" && git fetch --tags && git checkout "$ISAACLAB_REF"
./isaaclab.sh --install
uv pip install pillow numpy pytest pyflakes
echo "Isaac Sim ${ISAACSIM_VERSION}, Isaac Lab ${ISAACLAB_REF} installed in $VENVS/isaac"
