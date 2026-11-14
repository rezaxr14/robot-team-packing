#!/usr/bin/env bash
# usage: ops/download_model.sh <hf repo id> <local name>
set -euo pipefail
source "$(dirname "$0")/env.sh"
activate train
hf_cli download "$1" --local-dir "$MODELS/$2"
echo "downloaded to $MODELS/$2"
