"""Merge a PEFT LoRA adapter into its base model so vLLM and TRL GRPO can use a plain checkpoint.
  python train/merge_lora.py --base $BASE_MODEL --adapter runs/brain_sft --out runs/brain_sft_merged
STATUS: UNTESTED (needs torch/peft/GPU). Tries the Auto class that can load the base, like ops/check_model.py."""
import argparse

import torch
import transformers
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = None
for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
    cls = getattr(transformers, name, None)
    if cls is None:
        continue
    try:
        model = cls.from_pretrained(a.base, dtype=torch.bfloat16)
        print("loaded base with", name)
        break
    except Exception as e:                                               # noqa: BLE001
        print(f"{name} failed: {type(e).__name__}: {str(e)[:120]}")
if model is None:
    raise SystemExit("could not load the base model")
merged = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
merged.save_pretrained(a.out)
transformers.AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
try:
    transformers.AutoProcessor.from_pretrained(a.base).save_pretrained(a.out)
except Exception as e:                                                   # noqa: BLE001
    print("processor not saved:", type(e).__name__)
print("merged ->", a.out)
