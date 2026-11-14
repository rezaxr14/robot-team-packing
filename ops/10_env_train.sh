#!/usr/bin/env bash
# Training / data env: PyTorch cu128 (works on Ada and Blackwell), transformers, TRL, PEFT, rtp test deps.
set -euo pipefail
source "$(dirname "$0")/env.sh"
uv venv "$VENVS/train" --python 3.12 --seed
activate train
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers trl peft datasets accelerate "huggingface_hub[cli]" numpy pandas pyarrow pillow pytest pyflakes mathruler pylatexenc
python - <<'PY'
import torch, transformers, trl, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
x = torch.randn(256, 256, device="cuda"); print("cuda matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
PY
