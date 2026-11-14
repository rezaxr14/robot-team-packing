#!/usr/bin/env bash
# LLaMA-Factory env (VLM LoRA SFT for the Eye), separate venv to avoid pin conflicts.
set -euo pipefail
source "$(dirname "$0")/env.sh"
uv venv "$VENVS/lf" --python 3.12 --seed
activate lf
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install llamafactory pillow || { git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$RTP_HOME/LLaMA-Factory" && uv pip install -e "$RTP_HOME/LLaMA-Factory"; }
llamafactory-cli version || true
