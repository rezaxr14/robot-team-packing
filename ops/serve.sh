#!/usr/bin/env bash
# usage: ops/serve.sh <served-name> <model path or id> <port> [gpu_memory_utilization] [extra vllm args...]
# LoRA adapters: add  --enable-lora --lora-modules name=/path/to/adapter
set -euo pipefail
source "$(dirname "$0")/env.sh"
activate serve
NAME="$1"; MODEL="$2"; PORT="$3"; UTIL="${4:-0.35}"; shift $(( $# < 4 ? $# : 4 ))
exec vllm serve "$MODEL" --served-model-name "$NAME" --port "$PORT" --gpu-memory-utilization "$UTIL" \
  --max-model-len "${MAXLEN:-8192}" --dtype auto "$@"
