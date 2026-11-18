#!/usr/bin/env bash
# Source this file: `source ops/env.sh`. Everything big lives under $RTP_HOME (default ~/rtp_work).
export RTP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RTP_HOME="${RTP_HOME:-$HOME/rtp_work}"
export VENVS="$RTP_HOME/venvs"
export MODELS="$RTP_HOME/models"
export DATA="$RTP_HOME/data"
export RUNS="$RTP_HOME/runs"
export HF_HOME="${HF_HOME:-$RTP_HOME/hf_cache}"
export ISAACLAB_DIR="$RTP_HOME/IsaacLab"
export ISAACLAB_REF="${ISAACLAB_REF:-v3.0.0}"   # Isaac Lab 3.0 GA (D-25); Python 3.12, Newton backend
export ISAAC_PY="${ISAAC_PY:-$VENVS/isaac_newton/bin/python}"   # the Newton-only venv (ops/install.sh --isaac)
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$VENVS" "$MODELS" "$DATA" "$RUNS" "$HF_HOME"
activate() { # shellcheck disable=SC1090
  source "$VENVS/$1/bin/activate"; }
hf_cli() { if command -v hf >/dev/null 2>&1; then hf "$@"; else huggingface-cli "$@"; fi; }
# optional local overrides (never required): if ops/config.sh exists it is sourced last
if [ -f "$RTP_ROOT/ops/config.sh" ]; then source "$RTP_ROOT/ops/config.sh"; fi
