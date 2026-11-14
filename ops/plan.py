"""The stage plan: every command the runbook mentions, as data. Run stages through ops/stage.py (timing, logs, resume).

  python ops/plan.py list                    # all groups and stages
  python ops/plan.py run <group> [--only NAME] [--force]
Stages are skipped if already done (rc 0) unless --force. A failing stage stops the group.
"""
from __future__ import annotations
import argparse
import sys
from collections import OrderedDict

E = "source ops/env.sh; "                      # every command starts from the repo root
T, I, L = E + "activate train; ", E + "activate isaac; ", E + "activate lf; "
VIKI_SERVE = lambda ckpt, label, level, data: (   # serve a released checkpoint, evaluate, shut down
    f"ops/with_server.sh viki \"{ckpt}\" 8000 0.40 -- bash -c '{T}python scripts/s0_trackA_eval.py --repo $VIKI_DIR "
    f"--level {level} --data {data} --model viki --label {label} --upstream-truncation'")

GROUPS: "OrderedDict[str, list]" = OrderedDict([
    ("setup", [
        ("preflight", "python3 ops/preflight.py"),
        ("uv", "ops/01_install_uv.sh"),
        ("env_train", "ops/10_env_train.sh"),
        ("env_serve", "ops/11_env_serve.sh"),
        ("env_isaac", "ops/12_env_isaac.sh"),
        ("env_lf", "ops/14_env_lf.sh"),
        ("models", "ops/13_models.sh"),
    ]),
    ("smoke", [
        ("smoke_tests", "ops/20_smoke.sh tests"),
        ("smoke_model", "ops/20_smoke.sh model"),
        ("smoke_vllm", "ops/20_smoke.sh vllm"),
        ("smoke_isaac", "ops/20_smoke.sh isaac"),
        ("env_report", T + "python scripts/s0_env_report.py > reports/env_report.json"),
        ("smoke_train", T + "python -m pytest tests/test_torch_parts.py tests/test_policy_rollout.py -q"),
    ]),
    ("trackA", [
        ("inventory", T + "python scripts/s0_inventory.py --data $VIKI_DATA > reports/inventory.txt"),
        ("trackA_L2_id", VIKI_SERVE("$VIKI_CKPT_L2", "L2_id", 2, "$L2_ID")),
        ("trackA_L2_ood", VIKI_SERVE("$VIKI_CKPT_L2", "L2_ood", 2, "$L2_OOD")),
        ("trackA_L1_id", VIKI_SERVE("$VIKI_CKPT_L1", "L1_id", 1, "$L1_ID")),
        ("trackA_L1_ood", VIKI_SERVE("$VIKI_CKPT_L1", "L1_ood", 1, "$L1_OOD")),
        ("trackA_pipeline_base", f"ops/with_server.sh brain \"$BASE_MODEL\" 8001 0.30 -- bash -c '{T}python scripts/s0_trackA_pipeline_eval.py "
                                 "--repo $VIKI_DIR --data $L2_ID --label L2_base --brain-endpoint http://localhost:8001/v1 --brain-model brain --conditions oracle,none'"),
    ]),
    ("isaac", [
        ("isaac_01_scene", I + "python isaac/bringup/01_scene.py --headless"),
        ("isaac_02_reach", I + "python isaac/bringup/02_ik_reach.py --headless"),
        ("isaac_03_grasp_place", I + "python isaac/bringup/03_grasp_place.py --headless"),
        ("isaac_04_force_window", I + "python isaac/bringup/04_force_window.py --headless"),
        ("isaac_noisy_eyes_no_probe", I + "python isaac/bringup/05_episode.py --headless --episodes 30 --eyes noisy --tag isaac_noisy_no_probe"),
        ("isaac_noisy_eyes_probe", I + "python isaac/bringup/05_episode.py --headless --episodes 30 --eyes noisy --probe --tag isaac_noisy_probe"),
        ("isaac_08_regression_scripted", I + "python isaac/bringup/08_regression.py --headless --trials 3"),
        ("isaac_05_episodes_scripted", I + "python isaac/bringup/05_episode.py --headless --episodes 10 --tag isaac_scripted_oracle"),
    ]),
    ("data", [
        ("sym_datasets", T + "python scripts/make_datasets.py --out $DATA/sym --brain-sft 4000 --grpo 4000 --hindsight 2000 --noisy-eyes --eye 0"),
        ("arm_demos", I + "python isaac/bringup/07_collect_demos.py --headless --episodes 300 --out $DATA/arm_demos.npz"),
        ("eye_renders", I + "python isaac/bringup/06_render_eye_data.py --headless --enable_cameras --seeds 1500 --out $DATA/eye"),
    ]),
    ("train", [
        ("arm_flow", T + "python train/distill_arm.py --data $DATA/arm_demos.npz --head flow --steps 20000 --out $RUNS/arm_flow"),
        ("arm_mlp", T + "python train/distill_arm.py --data $DATA/arm_demos.npz --head mlp --steps 20000 --out $RUNS/arm_mlp"),
        ("brain_sft", T + "python train/sft_brain.py --model $BASE_MODEL --data $DATA/sym/brain_sft.jsonl --out $RUNS/brain_sft"),
        ("brain_merge", T + "python train/merge_lora.py --base $BASE_MODEL --adapter $RUNS/brain_sft --out $RUNS/brain_sft_merged"),
        ("brain_grpo", T + "python train/grpo_brain.py --model $RUNS/brain_sft_merged --data $DATA/sym/grpo.jsonl --out $RUNS/brain_grpo"),
        ("eye_lf_config", T + "python ops/make_lf_config.py --data $DATA/eye --model $FALLBACK_MODEL --template qwen2_vl --out $RUNS/eye_sft"),
        ("eye_sft", L + "llamafactory-cli train $RUNS/eye_sft/eye_sft.yaml"),
        ("eye_merge", T + "python train/merge_lora.py --base $FALLBACK_MODEL --adapter $RUNS/eye_sft --out $RUNS/eye_sft_merged"),
        ("eye_rft_sample_accuracy", "ops/with_server.sh eye \"$RUNS/eye_sft_merged\" 8002 0.30 -- bash -c '" + T +
                                  "python scripts/eye_rft.py --rows $DATA/eye/eye_grpo.jsonl --endpoint http://localhost:8002/v1 --model eye --mode accuracy --k 8 --out $DATA/eye/rft_accuracy.jsonl'"),
        ("eye_rft_train_accuracy", T + "python ops/make_lf_config.py --data $DATA/eye --file rft_accuracy.jsonl --name eye_rft_accuracy --model $FALLBACK_MODEL --template qwen2_vl --out $RUNS/eye_rft_accuracy && " + L.replace("source ops/env.sh; ", "") + "llamafactory-cli train $RUNS/eye_rft_accuracy/eye_sft.yaml"),
        ("eye_rft_merge_accuracy", T + "python train/merge_lora.py --base $FALLBACK_MODEL --adapter $RUNS/eye_rft_accuracy --out $RUNS/eye_rft_accuracy_merged"),
        ("eye_rft_sample_credit", "ops/with_server.sh eye \"$RUNS/eye_sft_merged\" 8002 0.30 -- bash -c '" + T +
                                  "python scripts/eye_rft.py --rows $DATA/eye/eye_grpo.jsonl --endpoint http://localhost:8002/v1 --model eye --mode credit --k 8 --out $DATA/eye/rft_credit.jsonl'"),
        ("eye_rft_train_credit", T + "python ops/make_lf_config.py --data $DATA/eye --file rft_credit.jsonl --name eye_rft_credit --model $FALLBACK_MODEL --template qwen2_vl --out $RUNS/eye_rft_credit && " + L.replace("source ops/env.sh; ", "") + "llamafactory-cli train $RUNS/eye_rft_credit/eye_sft.yaml"),
        ("eye_rft_merge_credit", T + "python train/merge_lora.py --base $FALLBACK_MODEL --adapter $RUNS/eye_rft_credit --out $RUNS/eye_rft_credit_merged"),
        ("eye_grpo_accuracy", T + "python train/grpo_eye.py --model $RUNS/eye_sft_merged --data $DATA/eye/eye_grpo.jsonl --rewards format,accuracy --out $RUNS/eye_grpo_accuracy"),
        ("eye_grpo_credit", T + "python train/grpo_eye.py --model $RUNS/eye_sft_merged --data $DATA/eye/eye_grpo.jsonl --rewards format,credit --out $RUNS/eye_grpo_credit"),
    ]),
    ("eval", [
        ("sym_grid", T + "python scripts/run_experiments.py --seeds 200 --out reports/sym"),
        ("sym_verify", T + "python scripts/run_experiments.py --grid verify --seeds 200 --out reports/verify"),
        ("sym_credit", T + "python scripts/run_credit_analysis.py --seeds 100 --out reports/credit"),
        ("isaac_regression_learned", I + "python isaac/bringup/08_regression.py --headless --trials 3 --policy $RUNS/arm_flow --demos $DATA/arm_demos.npz"),
        ("isaac_learned_flow", I + "python isaac/bringup/09_eval_learned_arm.py --headless --episodes 50 --policy $RUNS/arm_flow --demos $DATA/arm_demos.npz --tag isaac_learned_flow"),
        ("isaac_learned_mlp", I + "python isaac/bringup/09_eval_learned_arm.py --headless --episodes 50 --policy $RUNS/arm_mlp --demos $DATA/arm_demos.npz --tag isaac_learned_mlp"),
        ("isaac_scripted_arm_same_seeds", I + "python isaac/bringup/05_episode.py --headless --episodes 50 --tag isaac_scripted_arm"),
        ("isaac_vlm_eye_episodes", "ops/with_server.sh eye \"$RUNS/eye_sft_merged\" 8002 0.30 -- bash -c '" + I +
                                   "python isaac/bringup/05_episode.py --headless --enable_cameras --episodes 50 --eye-endpoint http://localhost:8002/v1 --eye-model eye --tag isaac_vlm_eye'"),
        ("isaac_vlm_eye_rft_accuracy", "ops/with_server.sh eye \"$RUNS/eye_rft_accuracy_merged\" 8002 0.30 -- bash -c '" + I +
                                     "python isaac/bringup/05_episode.py --headless --enable_cameras --episodes 50 --eye-endpoint http://localhost:8002/v1 --eye-model eye --tag isaac_vlm_eye_rft_accuracy'"),
        ("isaac_vlm_eye_rft_credit", "ops/with_server.sh eye \"$RUNS/eye_rft_credit_merged\" 8002 0.30 -- bash -c '" + I +
                                     "python isaac/bringup/05_episode.py --headless --enable_cameras --episodes 50 --eye-endpoint http://localhost:8002/v1 --eye-model eye --tag isaac_vlm_eye_rft_credit'"),
        ("isaac_full_system", "ops/with_server.sh eye \"$RUNS/eye_sft_merged\" 8002 0.25 -- ops/with_server.sh brain \"$RUNS/brain_grpo\" 8001 0.25 -- bash -c '" + I +
                              "python isaac/bringup/05_episode.py --headless --enable_cameras --episodes 50 --brain-endpoint http://localhost:8001/v1 --brain-model brain --tool-brain --eye-endpoint http://localhost:8002/v1 --eye-model eye --tag isaac_full'"),
        ("brain_symbolic", "ops/with_server.sh brain \"$RUNS/brain_grpo\" 8001 0.30 -- bash -c '" + T +
                           "python scripts/run_experiments.py --seeds 200 --out reports/llm --llm-endpoint http://localhost:8001/v1 --llm-model brain'"),
        ("brain_symbolic_tools", "ops/with_server.sh brain \"$RUNS/brain_grpo\" 8001 0.30 -- bash -c '" + T +
                                 "python scripts/run_experiments.py --seeds 200 --out reports/llm_tools --llm-endpoint http://localhost:8001/v1 --llm-model brain --tool-brain'"),
        ("isaac_brain_episodes", "ops/with_server.sh brain \"$RUNS/brain_grpo\" 8001 0.30 -- bash -c '" + I +
                                 "python isaac/bringup/05_episode.py --headless --episodes 100 --seed 1000 --brain-endpoint http://localhost:8001/v1 --brain-model brain --tool-brain --tag isaac_brain'"),
    ]),
    ("bundle", [
        ("report", T + "python scripts/build_report.py --reports reports --extra $DATA --figures"),
        ("bundle", T + "python ops/bundle.py"),
    ]),
])


# variables a stage cannot run without (set them in ops/config.sh); checked BEFORE the stage starts
REQUIRES = {
    "trackA_L2_id": ["VIKI_CKPT_L2", "L2_ID"], "trackA_L2_ood": ["VIKI_CKPT_L2", "L2_OOD"],
    "trackA_L1_id": ["VIKI_CKPT_L1", "L1_ID"], "trackA_L1_ood": ["VIKI_CKPT_L1", "L1_OOD"],
    "trackA_pipeline_base": ["L2_ID", "BASE_MODEL"],
    "smoke_model": ["BASE_MODEL"], "smoke_vllm": ["BASE_MODEL"], "brain_sft": ["BASE_MODEL"], "brain_merge": ["BASE_MODEL"],
    "eye_lf_config": ["FALLBACK_MODEL"],
}

# paths a stage reads that earlier stages must have written ($RUNS/x or $DATA/x), by stage name
PRODUCES = {
    "models": ["$DATA/VIKI-R-data"], "sym_datasets": ["$DATA/sym"], "arm_demos": ["$DATA/arm_demos.npz"],
    "eye_renders": ["$DATA/eye"], "arm_flow": ["$RUNS/arm_flow"], "arm_mlp": ["$RUNS/arm_mlp"], "brain_sft": ["$RUNS/brain_sft"],
    "brain_merge": ["$RUNS/brain_sft_merged"], "brain_grpo": ["$RUNS/brain_grpo"], "eye_lf_config": ["$RUNS/eye_sft"],
    "eye_sft": ["$RUNS/eye_sft"], "eye_merge": ["$RUNS/eye_sft_merged"],
    "eye_rft_sample_accuracy": ["$DATA/eye/rft_accuracy.jsonl"], "eye_rft_sample_credit": ["$DATA/eye/rft_credit.jsonl"],
    "eye_rft_train_accuracy": ["$RUNS/eye_rft_accuracy"], "eye_rft_train_credit": ["$RUNS/eye_rft_credit"],
    "eye_rft_merge_accuracy": ["$RUNS/eye_rft_accuracy_merged"], "eye_rft_merge_credit": ["$RUNS/eye_rft_credit_merged"],
    "eye_grpo_accuracy": ["$RUNS/eye_grpo_accuracy"], "eye_grpo_credit": ["$RUNS/eye_grpo_credit"],
}
SHELL_VARS = {"HOME", "PATH"}


def missing_vars(stage: str, environ) -> list:
    return [v for v in REQUIRES.get(stage, []) if not environ.get(v)]


def lint() -> list:
    """Static checks of the plan: undefined $VARS, and inputs ($RUNS/..., $DATA/...) that no earlier stage produces."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    defined = set(SHELL_VARS)
    for f in (root / "ops" / "env.sh", root / "ops" / "config.example.sh"):
        defined |= set(re.findall(r"export (\w+)=", f.read_text()))
        defined |= set(re.findall(r"^(\w+)=", f.read_text(), re.M))
    problems, produced = [], set()
    for g, name, cmd in all_stages():
        for v in set(re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", cmd)):
            if v not in defined:
                problems.append(f"{name}: undefined variable ${v}")
        for v in REQUIRES.get(name, []):
            if v not in defined:
                problems.append(f"{name}: REQUIRES unknown variable {v}")
        outs = set(re.findall(r"--out (\$[A-Z]+/[\w.]+)", cmd)) | set(PRODUCES.get(name, []))
        for tok in set(re.findall(r"(\$(?:RUNS|DATA)/[\w.]+)", cmd)):
            if tok in outs or any(tok == p or tok.startswith(p + "/") for p in produced):
                continue
            problems.append(f"{name}: reads {tok} but no earlier stage produces it")
        produced |= outs
    return problems


def all_stages():
    return [(g, n, c) for g, items in GROUPS.items() for n, c in items]


def _config_env() -> dict:
    """Environment as the stages will see it: the process env plus whatever ops/env.sh (and config.sh) export."""
    import subprocess
    import os
    try:
        out = subprocess.run(["bash", "-c", "source ops/env.sh >/dev/null 2>&1; env -0"], capture_output=True, timeout=30).stdout
        env = dict(kv.split("=", 1) for kv in out.decode(errors="replace").split("\0") if "=" in kv)
    except Exception:                                                   # noqa: BLE001
        env = {}
    return {**env, **os.environ}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("lint")
    r = sub.add_parser("run")
    r.add_argument("group", choices=list(GROUPS))
    r.add_argument("--only", default=None)
    r.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "lint":
        probs = lint()
        print("\n".join(probs) if probs else "plan lint: ok")
        return 1 if probs else 0
    if a.cmd == "list":
        for g, items in GROUPS.items():
            print(f"[{g}]")
            for n, _ in items:
                print(f"   {n}")
        return 0
    from ops.stage import run_stage
    import os
    for n, cmd in GROUPS[a.group]:
        if a.only and n != a.only:
            continue
        env = _config_env()
        miss = missing_vars(n, env)
        if miss:
            print(f"stage {n} needs {', '.join(miss)}: set it in ops/config.sh (copy ops/config.example.sh), then rerun")
            return 2
        rc = run_stage(n, ["bash", "-lc", cmd], skip_if_done=not a.force)
        if rc != 0:
            print(f"stage {n} failed (rc={rc}); fix it and rerun: python ops/plan.py run {a.group} --only {n}")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
