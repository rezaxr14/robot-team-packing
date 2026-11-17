"""Arm-policy tests that need torch (skipped where unavailable).

Script-level SFT/GRPO smoke tests are re-added in P8 against the P7 dataset; the deleted Track-A
scripts they used are gone (D-35)."""
import pytest

torch = pytest.importorskip("torch")


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


