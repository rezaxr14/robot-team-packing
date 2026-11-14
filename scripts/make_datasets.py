"""Generate Brain SFT, GRPO prompt and Eye SFT datasets from the symbolic world.

  python scripts/make_datasets.py --out data --brain-sft 4000 --grpo 4000 --eye 2000 --noisy-eyes
Eye SFT rows reference image paths that must be rendered from the same seeds in Isaac Lab (placeholder paths).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.harness import Harness
import json
from rtp.datagen import gen_brain_sft, gen_grpo_prompts, gen_eye_sft, gen_hindsight_sft, gen_eye_grpo_rows, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--brain-sft", type=int, default=2000)
ap.add_argument("--grpo", type=int, default=2000)
ap.add_argument("--eye", type=int, default=1000)
ap.add_argument("--hindsight", type=int, default=0, help="hindsight-relabeled Brain SFT rows (from failed episodes)")
ap.add_argument("--eye-grpo", type=int, default=0, help="seeds for Eye GRPO rows (2 rows per seed: views A and B)")
ap.add_argument("--noisy-eyes", action="store_true", help="teacher sees noisy Eye reports (recommended for SFT)")
ap.add_argument("--seed0", type=int, default=100000, help="keep disjoint from evaluation seeds (0..)")
a = ap.parse_args()

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
if a.brain_sft:
    write_jsonl(gen_brain_sft(a.brain_sft, a.seed0, H, noisy_eyes=a.noisy_eyes), out / "brain_sft.jsonl")
if a.grpo:
    write_jsonl(gen_grpo_prompts(a.grpo, a.seed0 + 500000, H, noisy_eyes=True), out / "grpo.jsonl")
if a.eye:
    write_jsonl(gen_eye_sft(a.eye, a.seed0 + 900000), out / "eye_sft.jsonl")
if a.hindsight:
    rows, stats = gen_hindsight_sft(a.hindsight, a.seed0 + 300000, H, noisy_eyes=True)
    write_jsonl(rows, out / "hindsight_sft.jsonl")
    json.dump(stats, open(out / "hindsight_funnel.json", "w"), indent=2)     # report this funnel in the paper
    print("hindsight funnel:", stats)
if a.eye_grpo:
    write_jsonl(gen_eye_grpo_rows(a.eye_grpo, a.seed0 + 700000), out / "eye_grpo.jsonl")
print("wrote", sorted(p.name for p in out.iterdir()))
