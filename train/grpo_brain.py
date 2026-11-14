"""GRPO for the Brain with the symbolic plan-level rewards (rtp.rewards).

STATUS: UNTESTED (the authoring sandbox had no GPU and no TRL). Model-ID support for the chosen base
(Qwen3.5-4B) in your TRL/vLLM versions must be verified before relying on this.

Data: python -c "from rtp.datagen import *; from rtp.harness import Harness; \
      write_jsonl(gen_grpo_prompts(2000,0,Harness('harness_files')),'grpo.jsonl')"
Columns used by the reward functions: scene_json, spec_json (extra columns are passed to rewards as kwargs).
Run SFT warm-start first (gen_brain_sft -> your SFT trainer), then point --model at the SFT checkpoint.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="grpo.jsonl")
ap.add_argument("--out", default="runs/brain_grpo")
ap.add_argument("--num_generations", type=int, default=8)
ap.add_argument("--rollout-reward", action="store_true", help="add the closed-loop symbolic rollout reward (slower per step)")
ap.add_argument("--smoke", action="store_true", help="3 steps, tiny completions, CPU-safe: proves the wiring")
args = ap.parse_args()

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from rtp.harness import Harness
from rtp.rewards import trl_reward_funcs, trl_reward_weights

ds = load_dataset("json", data_files=args.data)["train"]
import torch
extra = dict(max_steps=3, max_completion_length=24, bf16=False, logging_steps=1, save_strategy="no",
             use_cpu=not torch.cuda.is_available()) if args.smoke else {}
cfg = GRPOConfig(**{**dict(output_dir=args.out, num_generations=args.num_generations, max_completion_length=1536,
                           learning_rate=1e-5, per_device_train_batch_size=args.num_generations, bf16=True,
                           logging_steps=5, save_steps=200, report_to="none"), **extra})
H = Harness(Path(__file__).resolve().parents[1] / "harness_files") if args.rollout_reward else None
cfg.reward_weights = trl_reward_weights(include_rollout=args.rollout_reward)
trainer = GRPOTrainer(model=args.model, reward_funcs=trl_reward_funcs(harness=H), args=cfg, train_dataset=ds,
                      peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM"))
trainer.train()
trainer.save_model(args.out)
