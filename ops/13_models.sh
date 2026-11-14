#!/usr/bin/env bash
# Downloads: base models, VIKI-R repo + dataset, and a listing of the released VIKI-R checkpoints.
set -euo pipefail
source "$(dirname "$0")/env.sh"
activate train
hf_cli download Qwen/Qwen3.5-4B --local-dir "$MODELS/Qwen3.5-4B"
hf_cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir "$MODELS/Qwen2.5-VL-3B-Instruct"
hf_cli download henggg/VIKI-R --repo-type dataset --local-dir "$VIKI_DATA"
[ -d "$VIKI_DIR" ] || git clone https://github.com/MARS-EAI/VIKI-R.git "$VIKI_DIR"
python "$RTP_ROOT/ops/list_viki_collection.py"
echo "Pick checkpoints from the list above, then: ops/download_model.sh <repo_id> <name>"
