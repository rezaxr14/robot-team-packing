#!/usr/bin/env bash
# usage: ops/20_smoke.sh <tests|model|vllm|isaac>
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RTP_ROOT"
case "$1" in
  tests)  activate "${RTP_PROFILE:-local_4050}"; python -m pytest tests -q ;;
  model)  activate "${RTP_PROFILE:-local_4050}"; python ops/check_model.py "${BASE_MODEL:-$MODELS/Qwen3.5-4B}" ;;
  vllm)   ops/with_server.sh smoke "${BASE_MODEL:-$MODELS/Qwen3.5-4B}" 8000 0.30 -- \
            bash -c 'curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16}" | tee reports/vllm_smoke.json' ;;
  isaac)  activate isaac_newton; python -c "import newton, warp; print('newton', newton.__version__, 'warp', warp.__version__)" ;;  # G-3 does the real run
  *) echo "unknown smoke: $1"; exit 2 ;;
esac
