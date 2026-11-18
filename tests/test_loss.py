"""Chunked LM loss: identical to the full-vocab loss, and never materialises the full logits."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from rtp.loss import (DEFAULT_CHUNK_TOKENS, chunked_cross_entropy, chunked_lm_loss, find_backbone,
                      find_head, shifted_labels)


class Backbone(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kw):
        h = torch.tanh(self.proj(self.embed(input_ids)))
        return SimpleNamespace(last_hidden_state=h)


class TinyLM(nn.Module):
    def __init__(self, vocab=53, hidden=8):
        super().__init__()
        self.model = Backbone(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, labels=None, **kw):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids).last_hidden_state))


class PeftLike(nn.Module):
    """Mimics PeftModel: the real model sits behind get_base_model()."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def get_base_model(self):
        return self.inner


def ids(b=2, t=37, vocab=53, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=g)


def reference(model, x):
    logits = model(x).logits[:, :-1]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))


def test_chunked_loss_equals_the_full_loss_and_is_chunk_size_independent():
    torch.manual_seed(0)
    model = TinyLM()
    x = ids()
    ref = reference(model, x)
    for chunk in (1, 5, 7, 36, 1000):
        got = chunked_lm_loss(model, x, chunk_tokens=chunk)
        assert torch.allclose(got, ref, atol=1e-6), (chunk, float(got), float(ref))
    assert DEFAULT_CHUNK_TOKENS == 128


def test_the_head_never_sees_more_than_one_chunk_of_positions():
    torch.manual_seed(0)
    model = TinyLM(vocab=53, hidden=8)
    x = ids(t=37)
    seen = []
    head = model.lm_head

    def counting_head(h):
        seen.append(h.shape[1])
        return head(h)

    hidden = model.model(x).last_hidden_state[:, :-1]
    chunked_cross_entropy(counting_head, hidden, x[:, 1:], chunk_tokens=7)
    assert max(seen) <= 7 and sum(seen) == x.shape[1] - 1          # T-1 predictions, in chunks
    assert len(seen) == 6                                          # ceil(36 / 7)


def test_shifted_labels_and_ignored_positions_match_torch():
    torch.manual_seed(0)
    model = TinyLM()
    x = ids()
    labels = shifted_labels(x)
    assert labels[0, 0] == x[0, 1] and labels[0, -1] == -100
    hidden = model.model(x).last_hidden_state[:, :-1]
    got = chunked_cross_entropy(model.lm_head, hidden, labels[:, 1:], chunk_tokens=4)
    ref = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, 53), labels[:, 1:].reshape(-1),
                          ignore_index=-100)
    assert torch.allclose(got, ref, atol=1e-6)


def test_peft_wrapper_is_unwrapped_and_gradients_reach_the_head():
    torch.manual_seed(0)
    inner = TinyLM()
    wrapped = PeftLike(inner)
    assert find_backbone(wrapped) is inner.model and find_head(wrapped) is inner.lm_head
    loss = chunked_lm_loss(wrapped, ids(), chunk_tokens=5)
    loss.backward()
    assert inner.lm_head.weight.grad is not None and torch.isfinite(inner.lm_head.weight.grad).all()


def test_a_model_without_a_head_or_backbone_says_so():
    with pytest.raises(AttributeError, match="LM head"):
        find_head(nn.Linear(4, 4))
    with pytest.raises(AttributeError, match="backbone"):
        find_backbone(nn.Linear(4, 4))
    with pytest.raises(ValueError):
        chunked_cross_entropy(nn.Linear(4, 4), torch.zeros(1, 3, 4), torch.full((1, 3), -100))
