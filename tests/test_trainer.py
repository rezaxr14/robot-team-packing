"""Trainer tests: prompt masking, exact mid-epoch resume, and the checkpoint format.

A tiny hand-built causal LM stands in for Qwen3.5 so this runs in a second on CPU. The properties
tested are the ones the paper depends on: the loss never sees prompt tokens, and interrupting a run
and resuming it from the pushed checkpoint produces the same trajectory.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from rtp.trainer import (StopTraining, TrainState, Trainer, collate, epoch_order,  # noqa: E402
                         load_checkpoint, save_checkpoint, tokenize_pairs)


class _Tok:
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 20) for c in text]}


class _Backbone(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.proj(self.emb(input_ids)))


class TinyLM(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.model = _Backbone(vocab, dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_base_model(self):
        return self

    def forward(self, **kw):
        raise AssertionError("the trainer must use the backbone+head path, not model.forward")


def rows(n=16):
    return [{"prompt": f"look at the table {i}", "completion": f'{{"skill": "{i % 3}"}}'} for i in range(n)]


def build(seed=0):
    torch.manual_seed(seed)
    model = TinyLM()
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def test_prompt_tokens_are_masked_and_completion_is_kept():
    tok = _Tok()
    out = tokenize_pairs([{"prompt": "abc", "completion": "de"}], tok, max_len=64)
    r = out[0]
    assert len(r["input_ids"]) == len(r["labels"])
    assert r["labels"][:3] == [-100, -100, -100]          # "abc" is the prompt
    assert all(t != -100 for t in r["labels"][3:])         # completion + eos are targets
    assert r["input_ids"][-1] == tok.eos_token_id


def test_over_long_rows_keep_the_completion():
    tok = _Tok()
    out = tokenize_pairs([{"prompt": "p" * 40, "completion": "hello"}], tok, max_len=8)[0]
    assert len(out["input_ids"]) == 8
    assert sum(t != -100 for t in out["labels"]) >= 5       # "hello" + eos still learnable


def test_a_prompt_only_row_is_dropped_when_no_eos_is_added():
    tok = _Tok()
    assert tokenize_pairs([{"prompt": "abc", "completion": ""}], tok, max_len=8,
                          add_eos=False) == []
    # with eos the row still teaches "stop here", which is a real (if tiny) signal
    assert tokenize_pairs([{"prompt": "abc", "completion": ""}], tok, max_len=8)[0]["labels"][-1] == tok.eos_token_id


def test_epoch_order_is_a_pure_function_of_seed_and_epoch():
    a = epoch_order(20, seed=3, epoch=0)
    assert a == epoch_order(20, seed=3, epoch=0)
    assert a != epoch_order(20, seed=3, epoch=1)
    assert sorted(a) == list(range(20))


def test_collate_pads_and_masks():
    b = collate([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
                 {"input_ids": [8], "labels": [8]}], pad_id=0)
    assert b["input_ids"].shape == (2, 3)
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["labels"][1].tolist()[1:] == [-100, -100]


def test_optimizer_state_and_step_survive_a_round_trip(tmp_path):
    model, opt = build()
    state = TrainState(step=5, epoch=1, micro_in_epoch=12, samples_seen=40)
    save_checkpoint(tmp_path, model=model, optimizer=opt, state=state)
    back = load_checkpoint(tmp_path, optimizer=opt)
    assert back["state"].step == 5 and back["state"].micro_in_epoch == 12
    assert (tmp_path / "model" / "model.pt").exists()


def test_load_checkpoint_without_state_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path)


def test_resume_mid_epoch_reproduces_the_uninterrupted_trajectory(tmp_path):
    ex = tokenize_pairs(rows(16), _Tok(), max_len=32)
    kw = dict(pad_id=0, micro_batch=2, grad_accum=2, epochs=2.0, lr=1e-3, log=lambda _m: None)
    # steps_per_epoch = ceil(16 / 4) = 4; total = 8. Interrupt after step 1: mid-epoch.
    model, opt = build()
    trainer = Trainer(model, optimizer=opt, examples=ex, **kw)
    trainer.fit()
    reference = [h["loss"] for h in trainer.state.history]
    assert trainer.state.step == 8

    model2, opt2 = build()

    def on_ckpt(state):
        save_checkpoint(tmp_path / "ckpt", model=model2, optimizer=opt2, state=state)
        if state.step >= 3:
            raise StopTraining("session over")

    t2 = Trainer(model2, optimizer=opt2, examples=ex, on_checkpoint=on_ckpt, **kw)
    t2.fit()
    assert t2.state.step == 3 and t2.state.epoch == 0 and t2.state.micro_in_epoch > 0

    # Rebuild the model from the checkpoint exactly as a resumed run would.
    model3, opt3 = build()
    model3.load_state_dict(torch.load(tmp_path / "ckpt" / "model" / "model.pt"))
    t3 = Trainer(model3, optimizer=opt3, examples=ex, **kw)
    t3.fit(resume_from=tmp_path / "ckpt")
    assert t3.state.step == 8
    assert t3.state.epoch == 2 and t3.state.micro_in_epoch == 0
    # the resumed run continues the same trajectory: losses line up within float noise
    got = [h["loss"] for h in t3.state.history]
    assert got[:3] == pytest.approx(reference[:3], abs=2e-4)
    assert got[3:] == pytest.approx(reference[3:], abs=2e-3)


def test_a_short_epoch_still_takes_its_update():
    """grad_accum larger than the epoch must not loop forever without stepping (the 4050 default)."""
    ex = tokenize_pairs(rows(4), _Tok(), max_len=32)
    model, opt = build()
    t = Trainer(model, optimizer=opt, examples=ex, pad_id=0, micro_batch=1, grad_accum=8,
                epochs=2.0, lr=1e-3, log=lambda _m: None)
    state = t.fit()
    assert state.step == 2 and state.epoch == 2       # one update per short epoch
    assert len(state.history) == 2
