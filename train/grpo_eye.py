"""GRPO for the Eye with selectable reward signals: --rewards format,accuracy,credit.

Run the three conditions the paper compares: `format,accuracy` (standard perception reward), `format,credit`
(counterfactual team credit), `format,accuracy,credit`. Credit is computed by replaying a symbolic episode with the
completion as the Eye's report (rtp.eye_rewards.r_eye_credit), so it needs seed/view columns and images rendered
from the SAME seeds.

STATUS: UNTESTED. Needs a VLM supported by your TRL version for image-conditioned GRPO (unverified for the chosen
base), rendered images at the paths in the dataset, and a one-time check that `prompt` + image columns match the
trainer's expected multimodal format. The reward functions themselves are tested (tests/test_training_signals.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="data/eye_grpo.jsonl")
ap.add_argument("--rewards", default="format,credit")
ap.add_argument("--out", default="runs/eye_grpo")
ap.add_argument("--num_generations", type=int, default=8)
args = ap.parse_args()

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from rtp.harness import Harness
from rtp.eye_rewards import trl_eye_reward_funcs

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
ds = load_dataset("json", data_files=args.data)["train"]
funcs = trl_eye_reward_funcs(H, which=tuple(args.rewards.split(",")))
cfg = GRPOConfig(output_dir=args.out, num_generations=args.num_generations, max_completion_length=1024,
                 learning_rate=1e-5, per_device_train_batch_size=args.num_generations, bf16=True,
                 logging_steps=5, save_steps=200, report_to="none")
GRPOTrainer(model=args.model, reward_funcs=funcs, args=cfg, train_dataset=ds,
            peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")).train()
