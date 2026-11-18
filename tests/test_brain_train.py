"""Orchestration tests for ``rtp.brain_train``: pairs, LoRA-target filtering, and a full run.

The real 4.66 B checkpoint is not loaded here (that is the notebook's job, on a GPU). A toy
causal LM stands in so the *logic* is pinned on CPU: which target names survive filtering, which
DeltaNet state tensors stay frozen, that a short run stops on request with its checkpoint pushed,
and that the next run resumes from what was pushed.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from rtp import brain_train                                        # noqa: E402
from rtp.profiles import get_profile, with_overrides               # noqa: E402


class Tok:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 20) for c in text]}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


class Backbone(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = nn.Linear(16, 32, bias=False)
        # names the profile asks for, plus DeltaNet state that must stay frozen in full tuning
        self.q_proj = nn.Linear(16, 16)
        self.in_proj_qkv = nn.Linear(16, 16)
        self.delta = nn.Linear(16, 16)
        self.delta.A_log = nn.Parameter(torch.zeros(4))

    def get_output_embeddings(self):
        return self.lm_head

    def get_base_model(self):
        return self

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")


def test_pairs_must_have_prompt_and_completion(tmp_path):
    good = tmp_path / "ok.jsonl"
    good.write_text('{"prompt": "p", "completion": "c", "meta": {"x": 1}}\n\n')
    assert brain_train.load_pairs(good) == [{"prompt": "p", "completion": "c"}]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "p"}\n')
    with pytest.raises(ValueError, match="completion"):
        brain_train.load_pairs(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no rows"):
        brain_train.load_pairs(empty)


def test_lora_targets_are_filtered_to_modules_that_exist():
    have, missing = brain_train.existing_targets(
        Toy(), ["q_proj", "in_proj_qkv", "linear_fc1", "qkv"])
    assert have == ["q_proj", "in_proj_qkv"]
    assert missing == ["linear_fc1", "qkv"]


def test_deltanet_state_is_frozen_in_full_tuning():
    names = brain_train.frozen_parameters(Toy())
    assert names == ["delta.A_log"]


def test_a_run_stops_on_request_and_the_next_one_resumes(tmp_path):
    prof = with_overrides(get_profile("local_4050", root=tmp_path), micro_batch=1, grad_accum=2)
    rows = [{"prompt": f"look {i}", "completion": f'{{"skill": {i % 2}}}'} for i in range(8)]

    class Store:
        def __init__(self, hf):
            from rtp.remotes import RemoteStore
            self._s = RemoteStore(kaggle=None, hf=hf, hf_repo="Reza-Nadimi/robot-team-packing")

        def __getattr__(self, name):
            return getattr(self._s, name)

    class HF:
        def __init__(self):
            self.files = set()
            self.blob = {}
            self.uploads = []

        def create_repo(self, **kw):
            return None

        def upload_folder(self, *, folder_path, path_in_repo, **kw):
            self.uploads.append(path_in_repo)
            root = Path(folder_path)
            for p in root.rglob("*"):
                if p.is_file():
                    key = f"{path_in_repo}/{p.relative_to(root).as_posix()}"
                    self.files.add(key)
                    self.blob[key] = p.read_bytes()

        def list_repo_files(self, **kw):
            return sorted(self.files)

        def delete_folder(self, *, folder_path, **kw):
            self.files = {f for f in self.files if not f.startswith(folder_path + "/")}

        def snapshot_download(self, *, repo_id, allow_patterns, local_dir, **kw):
            prefix = allow_patterns[0].split("**")[0].rstrip("/")
            for f in list(self.files):
                if f.startswith(prefix + "/"):
                    dest = Path(local_dir) / f
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(self.blob[f])
            return str(Path(local_dir) / prefix)

    def loader(profile, model_dir, log=print):
        torch.manual_seed(0)
        return Toy(), Tok(), {"tuning": "test"}

    hf = HF()
    first = brain_train.run_sft(prof, model_dir=tmp_path, data=rows, store=Store(hf), epochs=2.0,
                                every=1, seed=0, loader=loader, stop_after=2,
                                log=lambda _m: None)
    assert first.state.step == 2 and first.resumed_from is None
    pushed = [u for u in hf.uploads if "step-000000002" in u]
    assert pushed, hf.uploads

    second = brain_train.run_sft(prof, model_dir=tmp_path, data=rows, store=Store(hf), epochs=2.0,
                                 every=1, seed=0, loader=loader, log=lambda _m: None)
    assert second.resumed_from == 2
    assert second.state.step == 8                  # ceil(8/2) steps per epoch x 2 epochs
    assert (prof.paths.adapters / "sft" / "model.pt").exists()
