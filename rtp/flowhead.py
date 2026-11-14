"""Blind arm policy: small transformer encoder over (proprio, contact/force, skill, target, params,
messages) tokens + flow-matching action head (or MLP regression head for the ablation).

UNTESTED in the authoring sandbox (no torch / GPU there). The math is tested in tests/test_flowmath.py;
run `python -m rtp.flowhead` on a machine with torch for the toy sanity check.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .flowmath import interpolate, target_velocity, euler_integrate


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ArmPolicy(nn.Module):
    def __init__(self, proprio_dim=15, contact_dim=8, n_skills=8, n_targets=32, param_dim=4,
                 horizon=16, act_dim=8, d_model=128, n_layers=3, n_heads=4, head="flow", flow_steps=8):
        super().__init__()
        self.head, self.horizon, self.act_dim, self.flow_steps, self.d = head, horizon, act_dim, flow_steps, d_model
        self.proprio = nn.Linear(proprio_dim, d_model)
        self.contact = nn.Linear(contact_dim, d_model)
        self.skill = nn.Embedding(n_skills, d_model)
        self.target = nn.Embedding(n_targets, d_model)
        self.params = nn.Linear(param_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        if head == "flow":
            self.act_in = nn.Linear(act_dim, d_model)
            self.pos = nn.Parameter(torch.zeros(1, horizon, d_model))
            dec = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(dec, 2)
            self.vel_out = nn.Linear(d_model, act_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, horizon * act_dim))

    def encode(self, b: dict) -> torch.Tensor:
        toks = torch.stack([self.proprio(b["proprio"]), self.contact(b["contact"]), self.skill(b["skill"]),
                            self.target(b["target"]), self.params(b["params"])], dim=1)
        return self.encoder(toks)

    def velocity(self, x_t, t, mem):
        h = self.act_in(x_t) + self.pos + timestep_embedding(t, self.d)[:, None]
        return self.vel_out(self.decoder(h, mem))

    def loss(self, b: dict) -> torch.Tensor:
        mem, x1 = self.encode(b), b["actions"]
        if self.head == "mlp":
            pred = self.mlp(mem.mean(1)).view(-1, self.horizon, self.act_dim)
            return ((pred - x1) ** 2).mean()
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t = interpolate(x0, x1, t[:, None, None])
        return ((self.velocity(x_t, t, mem) - target_velocity(x0, x1)) ** 2).mean()

    @torch.no_grad()
    def sample(self, b: dict) -> torch.Tensor:
        mem = self.encode(b)
        if self.head == "mlp":
            return self.mlp(mem.mean(1)).view(-1, self.horizon, self.act_dim)
        x0 = torch.randn(mem.shape[0], self.horizon, self.act_dim, device=mem.device)
        B = mem.shape[0]
        return euler_integrate(lambda x, t: self.velocity(x, torch.full((B,), t, device=x.device), mem),
                               x0, self.flow_steps)


if __name__ == "__main__":      # toy check: learn a target-conditioned constant action chunk
    torch.manual_seed(0)
    for head in ("flow", "mlp"):
        m = ArmPolicy(head=head)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        for step in range(300):
            tgt = torch.randint(0, 32, (64,))
            b = {"proprio": torch.randn(64, 15), "contact": torch.zeros(64, 8), "skill": torch.zeros(64, dtype=torch.long),
                 "target": tgt, "params": torch.zeros(64, 4),
                 "actions": (tgt.float() / 32.0)[:, None, None].expand(64, 16, 8).contiguous()}
            l = m.loss(b); opt.zero_grad(); l.backward(); opt.step()
        b["actions"] = b["actions"]
        err = (m.sample(b) - b["actions"]).abs().mean().item()
        print(head, "final loss", round(l.item(), 4), "sample MAE", round(err, 4))
