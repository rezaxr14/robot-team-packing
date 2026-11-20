#!/usr/bin/env bash
# Source this file: `source ops/env.sh`.
# Everything for this project lives under the project root or ~/venvs (the single venv folder).
export RTP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RTP_HOME="${RTP_HOME:-$RTP_ROOT}"
export VENVS="${VENVS:-$HOME/venvs}"  # single venv folder; one env inside: $VENVS/rtp
export MODELS="${RTP_MODELS:-$HOME/models}"   # Qwen3.5-4B lives here already; no repo-root model copy
export DATA="$RTP_HOME/data"
export RUNS="$RTP_HOME/runs"
export HF_HOME="${HF_HOME:-$RTP_ROOT/.hf_cache}"
export ISAACLAB_DIR="$RTP_ROOT/.isaac/IsaacLab"
export ISAACLAB_REF="${ISAACLAB_REF:-v3.0.0}"   # Isaac Lab 3.0 GA (D-25); Python 3.12, Newton backend
export ISAAC_PY="${ISAAC_PY:-$VENVS/rtp/bin/python}"   # the single merged venv: training + Isaac Lab + Newton
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$VENVS" "$MODELS" "$DATA" "$RUNS" "$HF_HOME"
activate() { # shellcheck disable=SC1090
  source "$VENVS/$1/bin/activate"; }
hf_cli() { if command -v hf >/dev/null 2>&1; then hf "$@"; else huggingface-cli "$@"; fi; }
# optional local overrides (never required): if ops/config.sh exists it is sourced last
if [ -f "$RTP_ROOT/ops/config.sh" ]; then source "$RTP_ROOT/ops/config.sh"; fi
