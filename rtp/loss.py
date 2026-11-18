"""Chunked causal-LM loss: never materialise ``[batch, tokens, vocab]`` logits.

The vocab is 248,320 for Qwen3.5, so a full logits tensor costs

    tokens * 248320 * 4 bytes      # float32, what cross-entropy needs

which is 8.1 GB at 8192 tokens and 16.3 GB at 16384, before gradients and the log-softmax
intermediate. That is larger than the whole 4-bit model on a 6 GB laptop, so the loss is computed
in token chunks: the backbone runs once, then the LM head is applied to ``chunk_tokens`` positions
at a time and a summed cross-entropy is accumulated in float32.

    loss = chunked_lm_loss(model, input_ids)               # model may be a PeftModel
    loss.backward()                                        # gradients flow through every chunk

``chunk_tokens`` trades speed for memory: 128 tokens cost ~127 MB of float32 logits (plus the
log-softmax of the same size), 1024 costs ~1 GB. The default is sized for a 6 GB card at 248 k vocab
and can be raised on a 24/96 GB card.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: 128 * 248_320 * 4 B ≈ 127 MB per float32 logits chunk.
DEFAULT_CHUNK_TOKENS = 128


def find_head(model):
    """The LM head (``lm_head`` / ``get_output_embeddings``), through a PEFT wrapper if present."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    if head is None:
        head = getattr(base, "lm_head", None)
    if head is None:
        raise AttributeError("cannot find the LM head: no get_output_embeddings() or lm_head on "
                             f"{type(base).__name__}")
    return head


def find_backbone(model):
    """The transformer that turns token ids into hidden states (not the head)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(base, "model", None)                  # Llama/Qwen-style *ForCausalLM
    if backbone is None and hasattr(base, "get_decoder"):
        backbone = base.get_decoder()
    if backbone is None:
        raise AttributeError("cannot find the backbone: no .model or .get_decoder() on "
                             f"{type(base).__name__}")
    return backbone


def shifted_labels(input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Token ``t`` of the labels is token ``t+1`` of the input; the last position has no target."""
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index
    return labels


def chunked_cross_entropy(head, hidden, targets, *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                          ignore_index: int = -100) -> torch.Tensor:
    """Mean NLL of ``targets`` given ``hidden``, applying ``head`` in token chunks.

    ``hidden`` is ``[B, T, H]`` and ``targets`` is ``[B, T]`` already aligned to those positions.
    The sum is accumulated in float32 and divided by the number of counted targets, so the result
    matches a single ``F.cross_entropy`` over the full tensor.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    total = None
    counted = 0
    for start in range(0, hidden.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, hidden.shape[1])
        logits = head(hidden[:, start:stop])
        tgt = targets[:, start:stop].reshape(-1)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt,
                              ignore_index=ignore_index, reduction="sum")
        total = nll if total is None else total + nll
        counted += int((tgt != ignore_index).sum())
    if total is None or counted == 0:
        raise ValueError("no target tokens: every position is ignored")
    return total / counted


def chunked_lm_loss(model, input_ids: torch.Tensor, *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                    attention_mask: torch.Tensor | None = None, ignore_index: int = -100,
                    backbone_kwargs: dict | None = None) -> torch.Tensor:
    """Causal-LM loss with logits produced chunk by chunk (one full-size tensor is never built)."""
    backbone = find_backbone(model)
    head = find_head(model)
    out = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                   **(backbone_kwargs or {}))
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return chunked_cross_entropy(head, hidden[:, :-1], input_ids[:, 1:],
                                 chunk_tokens=chunk_tokens, ignore_index=ignore_index)
