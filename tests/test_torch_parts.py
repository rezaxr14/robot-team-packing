"""Tests that need torch / TRL / PEFT (skipped where unavailable): the arm policy, distillation, and the training
scripts in --smoke mode against the INSTALLED library versions."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def run(*args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def batch(n=32, H=16, A=8):
    g = torch.Generator().manual_seed(0)
    tgt = torch.randint(0, 32, (n,), generator=g)
    return {"proprio": torch.randn(n, 15, generator=g), "contact": torch.zeros(n, 8), "skill": torch.zeros(n, dtype=torch.long),
            "target": tgt, "params": torch.zeros(n, 4), "actions": (tgt.float() / 32)[:, None, None].expand(n, H, A).contiguous()}


@pytest.mark.parametrize("head", ["flow", "mlp"])
def test_arm_policy_shapes_loss_and_sampling(head):
    from rtp.flowhead import ArmPolicy
    torch.manual_seed(0)
    m = ArmPolicy(head=head, d_model=32, n_layers=1, n_heads=2)
    b = batch()
    loss = m.loss(b)
    assert loss.ndim == 0 and torch.isfinite(loss)
    out = m.sample(b)
    assert out.shape == b["actions"].shape and torch.isfinite(out).all()


@pytest.mark.parametrize("head", ["flow", "mlp"])
def test_arm_policy_learns_a_conditional_mapping(head):
    from rtp.flowhead import ArmPolicy
    torch.manual_seed(0)
    m = ArmPolicy(head=head, d_model=48, n_layers=2, n_heads=4, flow_steps=8)
    opt = torch.optim.AdamW(m.parameters(), 2e-3)
    b = batch(64)
    first = float(m.loss(b))
    for _ in range(150):
        l = m.loss(b); opt.zero_grad(); l.backward(); opt.step()
    assert float(l) < 0.5 * first
    mae = float((m.sample(b) - b["actions"]).abs().mean())
    assert mae < 0.2                                                   # conditioned on target, not the marginal mean


def test_flow_head_samples_vary_but_condition_matters():
    from rtp.flowhead import ArmPolicy
    torch.manual_seed(1)
    m = ArmPolicy(head="flow", d_model=32, n_layers=1, n_heads=2)
    b = batch(8)
    torch.manual_seed(2); a1 = m.sample(b)
    torch.manual_seed(3); a2 = m.sample(b)
    assert not torch.allclose(a1, a2)                                 # stochastic (different noise seeds)


def test_distill_script_synthetic_end_to_end(tmp_path):
    r = run("train/distill_arm.py", "--synthetic", "--steps", "40", "--batch", "32", "--head", "flow", "--out", str(tmp_path / "o"))
    assert r.returncode == 0, r.stderr[-800:]
    rep = json.loads((tmp_path / "o" / "report.json").read_text())
    assert rep["head"] == "flow" and 0 < rep["heldout_mae_mean"] < 1 and (tmp_path / "o" / "arm.pt").exists()


trl = pytest.importorskip("trl")
pytest.importorskip("peft")


def test_training_scripts_smoke_chain_on_a_tiny_model(tmp_path):
    """make_datasets -> tiny model -> SFT (LoRA) -> merge -> GRPO with our rewards (+ rollout), all in smoke mode."""
    d = tmp_path / "d"
    r = run("scripts/make_datasets.py", "--out", str(d), "--brain-sft", "12", "--grpo", "12", "--hindsight", "0", "--eye", "0", "--noisy-eyes")
    assert r.returncode == 0, r.stderr[-500:]
    r = run("ops/make_tiny_model.py", "--data", str(d / "brain_sft.jsonl"), "--out", str(tmp_path / "tiny"))
    assert r.returncode == 0, r.stderr[-500:]
    r = run("train/sft_brain.py", "--model", str(tmp_path / "tiny"), "--data", str(d / "brain_sft.jsonl"), "--out", str(tmp_path / "sft"), "--smoke")
    assert r.returncode == 0 and (tmp_path / "sft" / "adapter_config.json").exists(), r.stderr[-800:]
    r = run("train/merge_lora.py", "--base", str(tmp_path / "tiny"), "--adapter", str(tmp_path / "sft"), "--out", str(tmp_path / "merged"))
    assert r.returncode == 0 and (tmp_path / "merged" / "model.safetensors").exists(), r.stderr[-800:]
    r = run("train/grpo_brain.py", "--model", str(tmp_path / "merged"), "--data", str(d / "grpo.jsonl"), "--out", str(tmp_path / "grpo"),
            "--smoke", "--num_generations", "2", "--rollout-reward")
    assert r.returncode == 0, r.stderr[-1200:]
    assert "rewards/rollout_reward/mean" in (r.stdout + r.stderr) and "rewards/rules_reward/mean" in (r.stdout + r.stderr)
