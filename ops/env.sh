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
export VIKI_DIR="$RTP_HOME/VIKI-R"
export VIKI_DATA="$DATA/VIKI-R-data"
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES            # Isaac Sim EULA acceptance for the pip install
mkdir -p "$VENVS" "$MODELS" "$DATA" "$RUNS" "$HF_HOME"
activate() { # shellcheck disable=SC1090
  source "$VENVS/$1/bin/activate"; }
hf_cli() { if command -v hf >/dev/null 2>&1; then hf "$@"; else huggingface-cli "$@"; fi; }
# user-editable paths (Track A files, model choices); copy ops/config.example.sh to ops/config.sh
if [ -f "$RTP_ROOT/ops/config.sh" ]; then source "$RTP_ROOT/ops/config.sh"; fi
