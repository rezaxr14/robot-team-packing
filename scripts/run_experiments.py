"""Run the ablation grid (symbolic world) and write records + markdown tables.

  python scripts/run_experiments.py --seeds 100 --out reports/sym
  # with a real LLM as Brain (vLLM server running):
  python scripts/run_experiments.py --seeds 100 --out reports/llm --llm-endpoint http://localhost:8000/v1 --llm-model brain-sft

Symbolic-world numbers are a pipeline/ablation-mechanics check unless the Brain is the trained model AND the
claim is scoped to the symbolic task. Physics claims need the Isaac Lab runs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtp.harness import Harness
from rtp.experiments import default_grid, verify_grid, run_config, write_records, summarize
from rtp import tables
from rtp.attribution_eval import sample_for_labeling, write_label_csv

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=50)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--out", default="reports/sym")
ap.add_argument("--clean-eyes", action="store_true", help="oracle Eyes instead of noisy Eyes")
ap.add_argument("--llm-endpoint", default=None)
ap.add_argument("--llm-model", default=None)
ap.add_argument("--grid", choices=("default", "verify"), default="default", help="verify = plan-verification ablation on the flaky-Brain fixture")
ap.add_argument("--tool-brain", action="store_true", help="multi-turn Brain with tools (check_rule, dry_run, look, ...)")
ap.add_argument("--label-sample", type=int, default=40, help="failed episodes to export for hand-labelling")
a = ap.parse_args()

H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
gen = chat = None
if a.llm_endpoint:
    from rtp.llm_client import ChatClient
    client = ChatClient(a.llm_endpoint, a.llm_model)
    gen = client.generate(max_tokens=1500)
    chat = lambda msgs: client.chat(msgs, max_tokens=1500)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
records = []
for cfg in (verify_grid() if a.grid == "verify" else default_grid(noisy=not a.clean_eyes)):
    if gen:
        cfg.brain = "tool" if a.tool_brain else "llm"
    records += run_config(cfg, range(a.seed0, a.seed0 + a.seeds), H, generate=gen, keep_story=True, chat=chat)
    print("done", cfg.name, flush=True)
write_records(records, out / "records.jsonl")
json.dump(summarize(records), open(out / "summary.json", "w"), indent=2, default=str)
md = "## Main\n" + tables.table_main(records) + "\n\n## Ablations vs full\n" + tables.table_ablation(records, reference="full" if any(x["config"] == "full" for x in records) else records[0]["config"]) + \
     "\n\n## Fatal-failure attribution (auto)\n" + tables.table_attribution(records) + "\n"
(out / "tables.md").write_text(md)
write_label_csv(sample_for_labeling(records, a.label_sample), out / "attribution_to_label.csv")
print(md)
