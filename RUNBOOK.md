# Runbook: what to run, in order, and what to paste back

Paste back only the small JSON/markdown files named below (not model outputs or weights). Nothing here needs
secrets. If a step fails, paste the last ~30 lines of the error and the output of step 0.

## 0. Environment report (any machine, 10 seconds)
    python scripts/s0_env_report.py > reports/env_report.json      # mkdir -p reports first
Paste back: `reports/env_report.json`.

## 1. Tests on your machine
    pip install pytest numpy pandas mathruler pylatexenc
    git clone https://github.com/MARS-EAI/VIKI-R.git ~/VIKI-R
    VIKI_R_REPO=~/VIKI-R python -m pytest tests -q
Expect: all tests pass (115 at the time of writing). Paste back: the last line, plus any failure text.
If torch is installed: `python -m rtp.flowhead` (prints sample MAE for flow vs mlp on a toy task) and
`python train/distill_arm.py --synthetic --steps 500 --out runs/arm_smoke`. Paste back: both outputs.
(`rtp/flowhead.py` has never been executed; this is its first real test.)

## 2. VIKI-R data and checkpoints (needs huggingface.co)
    git clone https://huggingface.co/datasets/henggg/VIKI-R ~/VIKI-R-data      # per the upstream README
Checkpoints are in the collection https://huggingface.co/collections/henggg/viki-r-68ef92c3a67e90535ecc1a9b
(3B and 7B, L1/L2/L3). Pick the L2 7B checkpoint first.

## 3. Dataset inventory and scorer exposure
    python scripts/s0_inventory.py --data ~/VIKI-R-data > reports/inventory.txt
Paste back: `reports/inventory.txt`. It reports splits, columns, row counts, and the share of L2 items with more than
one candidate start position (our exposure to the scorer nondeterminism). If it prints
"no init_pos found", paste the column list; `find_gt()` needs the real schema.

## 4. Reproduce Track A (item 7 of Stage 0)
    vllm serve <path-or-hf-id-of-L2-checkpoint> --served-model-name viki-r-7b --port 8000
    python scripts/s0_trackA_eval.py --repo ~/VIKI-R --level 2 --data <L2 test .parquet> \
        --model viki-r-7b --label L2_split_A --upstream-truncation
Repeat for the other L2 split you consider out-of-domain (you decide which file is which; the upstream eval script
names are inconsistent: it loads `.../id/val.json` and writes to a directory called `ood`), and for L1 with
`--level 1`. Paste back: each `reports/trackA/<label>_summary.json`.
What to look for: "Upstream-style %" should be near the paper's number for that split; "Position-robust %" below it
means the upstream score depends on the random start-position draw.

### 4b. Eye -> Brain decomposition on L2 (see TRACKA_CONTROL.md)
Serve any text-only chat model as the Brain (and optionally a VLM as the Eye), then:
    python scripts/s0_trackA_pipeline_eval.py --repo ~/VIKI-R --data <L2 test .parquet> --label L2_A \
        --brain-endpoint http://localhost:8001/v1 --brain-model brain --conditions oracle,none
Add `--eye-endpoint ... --eye-model ... --conditions oracle,none,eye` when a VLM Eye is available.
Paste back: `reports/trackA_pipeline/<label>_pipeline_summary.json`.

## 5. Symbolic ablation grid (CPU, minutes)
    python scripts/run_experiments.py --seeds 200 --out reports/sym
Paste back: `reports/sym/tables.md`. These are mechanics checks of the symbolic fixture, not paper evidence.
Then hand-label failure attribution: open `reports/sym/attribution_to_label.csv`, fill `human_agent`
(Eye | Brain | Arm | Mixed | Unclear) for as many rows as you can (40 is plenty), and run
    python -c "from rtp.attribution_eval import agreement; print(agreement('reports/sym/attribution_to_label.csv'))"
Paste back: that dict. It tells us whether the automatic Eye/Brain/Arm labels can be reported at all.

## 6. Brain data, SFT, GRPO (needs a GPU; model IDs are yours to choose and verify)
    python scripts/make_datasets.py --out data --brain-sft 4000 --grpo 4000 --noisy-eyes
    python train/sft_brain.py  --model <base-model> --data data/brain_sft.jsonl --out runs/brain_sft
    python train/grpo_brain.py --model runs/brain_sft --data data/grpo.jsonl --out runs/brain_grpo
Evaluate a trained Brain in the symbolic loop (vLLM serving the merged model):
    python scripts/run_experiments.py --seeds 200 --out reports/llm --llm-endpoint http://localhost:8000/v1 --llm-model brain
Add `--tool-brain` to give the Brain tools (check_rule, query_scene, look, dry_run, submit_plan); the records then include tool-call totals.
Paste back: `reports/llm/tables.md` and any trainer errors. TRL/vLLM support for your chosen base is unverified.

## 7. Training-signal data (CPU)
    python scripts/make_datasets.py --out data --hindsight 2000 --eye-grpo 2000 --brain-sft 0 --grpo 0 --eye 0
Paste back: `data/hindsight_funnel.json` (failed episodes -> relabeled -> replay-validated; the yield is a paper number
once the Brain is real). Eye GRPO needs images rendered from the same seeds (`seed`, `view` columns), so it waits for Isaac.

## 8. More CPU analyses (minutes)
    python scripts/run_credit_analysis.py --seeds 100 --out reports/credit        # Eye credit by fault type and field
    python scripts/run_experiments.py --grid verify --seeds 100 --out reports/verify   # plan verification vs a flaky planner
Paste back: `reports/credit/credit.md` and `reports/verify/tables.md`. Both are symbolic-fixture mechanics.
