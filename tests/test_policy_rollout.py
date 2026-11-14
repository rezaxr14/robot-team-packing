import pytest

torch = pytest.importorskip("torch")

from rtp.flowhead import ArmPolicy
from rtp.policy_rollout import rollout_skill, median_lengths
from rtp.vlm_eye import VLMEye, eye_messages, EYE_PROMPT

GOALS = torch.tensor([[0.5] * 8, [-0.4] * 8, [0.2, -0.3, 0.6, 0.0, 0.1, -0.5, 0.3, 0.04], [0.0] * 8])
H = 16


class ToyArm:
    """First-order arm: q <- q + 0.5 (a - q). Reaching target k means driving q to GOALS[k]."""
    def __init__(self, q0):
        self.q = torch.tensor(q0, dtype=torch.float32)
        self.applied = 0

    def obs(self):
        return {"proprio": self.q.tolist() + [0.0] * 7, "contact": [0.0] * 8}

    def apply(self, a):
        self.q = self.q + 0.5 * (torch.tensor(a) - self.q)
        self.applied += 1


def make_demos(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.rand(n, 8, generator=g) * 1.0 - 0.5
    tgt = torch.randint(0, 4, (n,), generator=g)
    alpha = torch.linspace(0.25, 1.0, H)[None, :, None]
    chunk = q[:, None, :] + alpha * (GOALS[tgt][:, None, :] - q[:, None, :])     # straight-line expert chunk
    return {"proprio": torch.cat([q, torch.zeros(n, 7)], 1), "contact": torch.zeros(n, 8), "skill": torch.zeros(n, dtype=torch.long),
            "target": tgt, "params": torch.zeros(n, 4), "actions": chunk}


@pytest.mark.parametrize("head", ["mlp", "flow"])
def test_receding_horizon_rollout_reaches_the_commanded_target(head):
    torch.manual_seed(0)
    d = make_demos()
    m = ArmPolicy(head=head, d_model=64, n_layers=2, n_heads=4, flow_steps=8)
    opt = torch.optim.AdamW(m.parameters(), 2e-3)
    for step in range(500 if head == "flow" else 250):
        ix = torch.randint(0, len(d["target"]), (128,))
        l = m.loss({k: v[ix] for k, v in d.items()})
        opt.zero_grad(); l.backward(); opt.step()
    errs = []
    for k in range(4):
        env = ToyArm([0.3, -0.2, 0.1, 0.4, -0.3, 0.2, 0.0, 0.04])
        log = rollout_skill(m, env.obs, env.apply, n_steps=40, skill_id=0, target_id=k, params=[0, 0, 0, 0], exec_steps=4)
        errs.append(float((env.q - GOALS[k]).abs().mean()))
        assert env.applied == 40 and log["replans"] == 10
    assert max(errs) < 0.12, errs                                   # each command drives the arm to ITS target


def test_rollout_respects_step_budget_and_exec_steps():
    m = ArmPolicy(head="mlp", d_model=16, n_layers=1, n_heads=2)
    env = ToyArm([0.0] * 8)
    log = rollout_skill(m, env.obs, env.apply, n_steps=7, skill_id=0, target_id=0, params=[0, 0, 0, 0], exec_steps=3)
    assert env.applied == 7 and log["replans"] == 3 and len(log["actions"]) == 7


def test_median_lengths():
    assert median_lengths([0, 0, 0, 1, 1], [100, 120, 140, 50, 60]) == {0: 120, 1: 60}
    assert median_lengths([], []) == {}


# ---------- VLM Eye ----------
GOOD = '[{"name":"book1","cls":"book","region":"table_left","size":2.0,"mass_guess":1.0,"fragility_guess":0.1}]'


def test_vlm_eye_parses_replies_and_counts_failures():
    from rtp.datagen import sample_world
    replies = iter([GOOD, "I see a table", GOOD])
    eye = VLMEye("A", lambda m: next(replies), lambda view: b"\x89PNGfake")
    w = sample_world(0)
    r1, r2, r3 = eye.report(w, 0), eye.report(w, 1), eye.report(w, 2)
    assert [o["name"] for o in r1.content["objects"]] == ["book1"] and r1.content["view"] == "A"
    assert r2.content["objects"] == [] and eye.stats == {"calls": 3, "parse_failures": 1, "errors": 0}
    boom = VLMEye("B", lambda m: (_ for _ in ()).throw(RuntimeError("server down")), lambda v: b"x")
    assert boom.report(w, 0).content["objects"] == [] and boom.stats["errors"] == 1 and boom.stats["parse_failures"] == 1


def test_vlm_eye_message_has_image_and_the_shared_prompt():
    m = eye_messages(b"abc")
    assert m[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,") and m[0]["content"][1]["text"] == EYE_PROMPT


def test_vlm_eyes_drive_a_full_episode_and_failures_are_attributed():
    from pathlib import Path
    from rtp.harness import Harness
    from rtp.datagen import sample_world, sample_spec
    from rtp.brain import ScriptedBrain
    from rtp.loop import LoopConfig, run_episode
    from rtp.perception import oracle_report
    import json
    H = Harness(Path(__file__).resolve().parents[1] / "harness_files")
    w = sample_world(2)
    chat = lambda msgs: json.dumps(oracle_report(w, "A").content["objects"])       # an Eye that reads the world perfectly
    eyes = [VLMEye("A", chat, lambda v: b"x")]
    res = run_episode(w, eyes, ScriptedBrain(), sample_spec(w, 2), H, LoopConfig())
    assert res.success and eyes[0].stats["parse_failures"] == 0
    w2 = sample_world(2)
    dead = [VLMEye("A", lambda m: "no json", lambda v: b"x")]
    res2 = run_episode(w2, dead, ScriptedBrain(), sample_spec(w2, 2), H, LoopConfig())
    assert not res2.success and {f["agent"] for f in res2.failures if f.get("fatal")} == {"Eye"}
    assert dead[0].stats["parse_failures"] >= 1
