"""Checkpoint policy: push cadence, pruning, and fetching the newest remote step to resume from.

The HF client is a double, so the test pins the ordering and the retention arithmetic without a
network. ``tests/test_remotes.py`` covers the real repo id and slug shapes; the live round-trip is
recorded in DECISIONS.md (step 7 pushed and read back).
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rtp import remotes as rtp_remotes                         # noqa: E402
from rtp.checkpoints import CheckpointPolicy, fetch_latest     # noqa: E402
from rtp.trainer import TrainState, load_checkpoint, save_checkpoint  # noqa: E402


class FakeHF:
    """Just enough HfApi: list/upload/delete/snapshot_download over an in-memory tree."""

    def __init__(self, files=()):
        self.files = set(files)
        self.downloads = []
        self.deleted = []

    def create_repo(self, **kw):
        return None

    def upload_folder(self, *, folder_path, path_in_repo, **kw):
        root = Path(folder_path)
        for p in root.rglob("*"):
            if p.is_file():
                self.files.add(f"{path_in_repo}/{p.relative_to(root).as_posix()}")
        self.files.add(f"{path_in_repo}/{Path(folder_path).name}")   # folder marker, harmless
        for p in root.iterdir():
            if p.is_dir():
                self.files.add(f"{path_in_repo}/{p.name}/x")

    def list_repo_files(self, **kw):
        return sorted(self.files)

    def delete_folder(self, *, path_in_repo, **kw):
        self.files = {f for f in self.files if not f.startswith(path_in_repo + "/")}
        self.deleted.append(path_in_repo)

    def snapshot_download(self, *, repo_id, allow_patterns, local_dir, **kw):
        prefix = allow_patterns[0].split("**")[0].rstrip("/")
        self.downloads.append(prefix)
        dest = Path(local_dir) / prefix
        (dest / "model").mkdir(parents=True, exist_ok=True)
        (dest / "model" / "model.pt").write_bytes(b"weights")
        (dest / "trainer_state.json").write_text('{"trainer": {"step": 7}}')
        return str(dest)


class Store(rtp_remotes.RemoteStore):
    """The real prefix arithmetic, with the HF client swapped for the in-memory double."""

    def __init__(self, hf):
        super().__init__(kaggle=None, hf=hf, hf_repo="Reza-Nadimi/robot-team-packing",
                         say=lambda _m: None)
        self.pushes = []                # (profile, stage, step)

    def push_checkpoint(self, root, profile, stage, step, message=""):
        self.pushes.append((profile, stage, step))
        return super().push_checkpoint(root, profile, stage, step, message)


class Weights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4))
        self.peft_config = {}                      # look like a PeftModel to save_checkpoint

    def save_pretrained(self, path):
        from safetensors.torch import save_file
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")
        save_file({"w": self.w.detach()}, Path(path) / "adapter_model.safetensors")


def policy(tmp_path, store, **kw):
    model, opt = Weights(), torch.optim.AdamW([torch.nn.Parameter(torch.zeros(3))], lr=1e-3)
    return CheckpointPolicy(root=tmp_path, profile="local_4050", stage="sft", store=store,
                            log=lambda _m: None, **kw).bind(model, opt)


def test_saves_only_on_the_cadence(tmp_path):
    p = policy(tmp_path, Store(FakeHF()), every=2)
    assert p(TrainState(step=1)) is None
    path = p(TrainState(step=2))
    assert path and (path / "trainer_state.json").exists()
    assert (path / "model" / "adapter_model.safetensors").exists()
    assert (path / "optimizer.pt").exists()


def test_pushing_the_cadence_prunes_to_the_newest_keep(tmp_path):
    hf = FakeHF()
    st = Store(hf)
    p = policy(tmp_path, st, every=1, keep=2)
    for step in (1, 2, 3):
        p(TrainState(step=step))
    assert [s for _, _, s in st.pushes] == [1, 2, 3]
    assert st.list_checkpoints("local_4050", "sft") == [2, 3]
    assert hf.deleted == ["checkpoints/local_4050/sft/step-000000001"]


def test_a_failed_push_keeps_the_local_checkpoint_and_training(tmp_path):
    class Broken(Store):
        def push_checkpoint(self, *a, **kw):
            raise OSError("network down")

    p = policy(tmp_path, Broken(FakeHF()), every=1)
    path = p(TrainState(step=4))                 # must not raise
    assert path and (path / "trainer_state.json").exists()


def test_push_can_be_disabled_for_a_local_only_run(tmp_path):
    p = policy(tmp_path, None, every=1, push=True)
    p(TrainState(step=1))
    assert p.pushed == []


def test_fetch_latest_returns_the_newest_step_and_its_state(tmp_path):
    hf = FakeHF(["checkpoints/local_4050/sft/step-000000003/trainer_state.json",
                 "checkpoints/local_4050/sft/step-000000007/trainer_state.json"])
    step, path = fetch_latest(Store(hf), "local_4050", "sft", tmp_path / "resume")
    assert step == 7 and path is not None
    assert load_checkpoint(path)["state"].step == 7


def test_fetch_latest_with_nothing_remote_starts_fresh(tmp_path):
    step, path = fetch_latest(Store(FakeHF()), "local_4050", "grpo", tmp_path / "resume")
    assert (step, path) == (None, None)


def test_profiles_do_not_see_each_others_checkpoints(tmp_path):
    hf = FakeHF(["checkpoints/local_4090/sft/step-000000500/x"])
    st = Store(hf)
    assert st.latest_checkpoint("local_4050", "sft") is None
    assert st.latest_checkpoint("local_4090", "sft") == 500


def test_adapter_weights_are_restored_from_the_pushed_checkpoint(tmp_path):
    model = Weights()
    with torch.no_grad():
        model.w.fill_(3.0)
    save_checkpoint(tmp_path, model=model, state=TrainState(step=9))
    fresh = Weights()
    assert float(fresh.w.sum()) == 0.0
    load_checkpoint(tmp_path, model=fresh)
    assert torch.allclose(fresh.w, torch.full_like(fresh.w, 3.0))


def test_real_peft_adapter_round_trip_uses_the_official_loader(tmp_path):
    """PEFT writes keys without '.default.'; a raw load_state_dict silently misses every adapter."""
    peft = pytest.importorskip("peft")
    nn = torch.nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.head = nn.Linear(8, 4)

        def forward(self, x):
            return self.head(self.q_proj(x))

    def make():
        return peft.get_peft_model(Tiny(), peft.LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj"], task_type="FEATURE_EXTRACTION"))

    first = make()
    with torch.no_grad():
        for name, p in first.named_parameters():
            if "lora_A" in name:
                p.fill_(0.25)
    before = {n: p.detach().clone() for n, p in first.named_parameters() if "lora_" in n}
    save_checkpoint(tmp_path, model=first, state=TrainState(step=3))

    second = make()
    load_checkpoint(tmp_path, model=second)
    after = {n: p for n, p in second.named_parameters() if "lora_" in n}
    assert set(after) == set(before)
    assert all(torch.allclose(after[n], before[n]) for n in before)
