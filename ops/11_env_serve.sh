#!/usr/bin/env bash
# Serving env: vLLM in its own venv (it pins its own torch).
set -euo pipefail
source "$(dirname "$0")/env.sh"
uv venv "$VENVS/serve" --python 3.12 --seed
activate serve
uv pip install vllm
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
