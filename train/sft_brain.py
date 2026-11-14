"""SFT warm-start for the Brain on teacher-validated plans (prompt/completion JSONL from make_datasets.py).

STATUS: UNTESTED (no GPU/TRL in the authoring sandbox). Written for TRL's prompt-completion dataset format;
check the SFTTrainer/SFTConfig argument names against your installed TRL version.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data/brain_sft.jsonl")
ap.add_argument("--out", default="runs/brain_sft")
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--smoke", action="store_true", help="3 steps, CPU-safe settings: proves the wiring, not the model")
args = ap.parse_args()

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import torch
ds = load_dataset("json", data_files=args.data)["train"].remove_columns(["meta"])
extra = dict(max_steps=3, per_device_train_batch_size=2, gradient_accumulation_steps=1, bf16=False, logging_steps=1,
             save_strategy="no", max_length=1024, use_cpu=not torch.cuda.is_available()) if args.smoke else {}
cfg = SFTConfig(**{**dict(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=4,
                          gradient_accumulation_steps=4, learning_rate=1e-4, bf16=True, logging_steps=10,
                          save_strategy="epoch", report_to="none", max_length=4096), **extra})
trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds,
                     peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"))
trainer.train()
trainer.save_model(args.out)
