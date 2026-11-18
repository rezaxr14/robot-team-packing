"""A small, real SFT trainer with **mid-epoch** resume and pluggable checkpoint shipping.

Why not just TRL's ``SFTTrainer``: two of the plan's requirements are not satisfied by its defaults
here. (1) The loss must never materialise ``tokens x vocab`` logits (6 GB card, 248 k vocab), which
is what :func:`rtp.loss.chunked_lm_loss` is for. (2) Training must resume **inside** an epoch and
continue on another machine from a checkpoint pushed to our HF repo, so the resume point has to be
exact, not "start the epoch again".

Resume is exact because batch order is a pure function of ``(seed, epoch)``: the same permutation
is rebuilt on resume and the first ``state.micro_in_epoch`` micro-batches are skipped. A checkpoint
also stores the optimizer state, the torch RNG state, learning-rate progress and the history, so a
resumed run is bit-comparable in its trajectory to an uninterrupted one (same seed, same data).

The trainer knows nothing about the network: it calls ``on_checkpoint(state)`` and the caller (see
``rtp.checkpoints.CheckpointPolicy``) saves locally and pushes to HF.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch

from rtp.loss import DEFAULT_CHUNK_TOKENS, chunked_lm_loss

class StopTraining(Exception):
    """Raise from a callback to end ``fit`` now (the checkpoint is already written).

    This is how a notebook stops at the end of a session and resumes on another machine: the state
    stays mid-epoch, and ``fit(resume_from=...)`` continues at the next micro-batch.
    """


STATE_FILE = "trainer_state.json"
OPTIMIZER_FILE = "optimizer.pt"
RNG_FILE = "rng.pt"
MODEL_DIR = "model"
TRAINABLE_FILE = "trainable.pt"


# --------------------------------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------------------------------
@dataclass
class TrainState:
    """Everything needed to continue a run exactly where it stopped."""

    step: int = 0                 # optimizer steps completed
    epoch: int = 0                # 0-based epoch currently in progress
    micro_in_epoch: int = 0       # micro-batches consumed in that epoch
    samples_seen: int = 0
    tokens_seen: int = 0
    best_loss: float = float("inf")
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch, "micro_in_epoch": self.micro_in_epoch,
                "samples_seen": self.samples_seen, "tokens_seen": self.tokens_seen,
                "best_loss": self.best_loss, "history": self.history[-200:]}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainState":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# --------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------
def tokenize_pairs(examples: Sequence[dict], tokenizer, *, max_len: int,
                   mask_prompt: bool = True, add_eos: bool = True) -> list[dict]:
    """Turn ``{"prompt": str, "completion": str}`` rows into ``input_ids``/``labels``.

    Prompt tokens are masked out of the loss (``-100``) when ``mask_prompt`` is true: SFT should
    learn to produce the plan, not to reproduce the question. Over-long rows keep the completion
    and drop prompt tokens from the *left*, because the tail of the prompt is what the answer
    conditions on.
    """
    eos = tokenizer.eos_token_id
    out = []
    for row in examples:
        prompt = row.get("prompt") or ""
        completion = row.get("completion") or ""
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if add_eos and eos is not None:
            c_ids = c_ids + [eos]
        if len(c_ids) >= max_len:
            c_ids = c_ids[:max_len]
        room = max_len - len(c_ids)
        if len(p_ids) > room:
            p_ids = p_ids[len(p_ids) - room:] if room > 0 else []
        ids = p_ids + c_ids
        labels = ([-100] * len(p_ids) if mask_prompt else list(p_ids)) + list(c_ids)
        if not any(t != -100 for t in labels):
            continue                                   # nothing to learn from
        out.append({"input_ids": ids, "labels": labels})
    return out


def epoch_order(n: int, *, seed: int, epoch: int) -> list[int]:
    """Deterministic permutation of ``range(n)`` for one epoch (the basis of exact resume)."""
    idx = list(range(n))
    random.Random(f"{seed}:{epoch}").shuffle(idx)
    return idx


def collate(rows: Sequence[dict], pad_id: int) -> dict:
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        k = len(r["input_ids"])
        ids[i, :k] = torch.tensor(r["input_ids"], dtype=torch.long)
        labels[i, :k] = torch.tensor(r["labels"], dtype=torch.long)
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# --------------------------------------------------------------------------------------------------
# checkpointing (local format; shipping is the caller's business)
# --------------------------------------------------------------------------------------------------
def save_checkpoint(root: Path | str, *, model, optimizer=None, state: TrainState,
                    tokenizer=None, extra: dict | None = None) -> Path:
    """Write a resume-able checkpoint. ``state`` is written last, so a half-written dir is detectable."""
    root = Path(root)
    (root / MODEL_DIR).mkdir(parents=True, exist_ok=True)
    # A PeftModel's save_pretrained writes only adapter weights (~0.1 GB). Never call it on a full
    # HF model here: that would write the entire 9 GB base into every checkpoint pushed to HF.
    is_peft = hasattr(model, "peft_config")
    if is_peft and hasattr(model, "save_pretrained"):
        model.save_pretrained(root / MODEL_DIR)
        model_format = "peft"
    else:
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        torch.save({n: v.detach() for n, v in model.state_dict().items() if n in trainable},
                   root / MODEL_DIR / TRAINABLE_FILE)
        model_format = "trainable"
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(root / MODEL_DIR)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), root / OPTIMIZER_FILE)
    torch.save(torch.get_rng_state(), root / RNG_FILE)
    payload = {"trainer": state.as_dict(), "extra": extra or {}, "model_format": model_format}
    tmp = root / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(root / STATE_FILE)                        # atomic: last writer wins
    return root


def _restore_model(model, root: Path, model_format: str | None) -> None:
    """Load the weights the checkpoint actually changed — adapters, or trainable tensors."""
    directory = root / MODEL_DIR
    if model_format == "peft" and (directory / "adapter_config.json").exists():
        try:
            from safetensors.torch import load_file
            weights = load_file(directory / "adapter_model.safetensors")
        except (ImportError, FileNotFoundError):
            weights = torch.load(directory / "adapter_model.bin", map_location="cpu")
    elif (directory / TRAINABLE_FILE).exists():
        weights = torch.load(directory / TRAINABLE_FILE, map_location="cpu")
    else:
        raise FileNotFoundError(f"no trainable weights under {directory}")
    try:
        from peft import PeftModel
        is_peft = isinstance(model, PeftModel)
    except Exception:                                        # noqa: BLE001
        is_peft = False
    if is_peft:
        # PEFT's on-disk format writes adapter keys WITHOUT the adapter name
        # ("...lora_A.weight"), while a live PeftModel expects "...lora_A.default.weight".
        # A raw load_state_dict therefore reports every adapter key as missing. Use the official
        # loader, which performs that rename and respects the target-module mapping.
        from peft import set_peft_model_state_dict
        result = set_peft_model_state_dict(model, weights, adapter_name="default")
        # The frozen base layers are legitimately absent from an adapter-only checkpoint.
        missing = [k for k in (getattr(result, "missing_keys", None) or []) if "lora_" in k]
    else:
        info = model.load_state_dict(weights, strict=False)
        missing = list(info.missing_keys)
    if missing:
        raise RuntimeError(f"checkpoint is missing weights for {missing[:5]}")


def load_checkpoint(root: Path | str, *, model=None, optimizer=None) -> dict:
    """Restore what exists. Returns ``{"state": TrainState, "extra": dict}``; raises if no state."""
    root = Path(root)
    state_path = root / STATE_FILE
    if not state_path.exists():
        raise FileNotFoundError(f"no {STATE_FILE} under {root}")
    payload = json.loads(state_path.read_text())
    if model is not None:
        _restore_model(model, root, payload.get("model_format"))
    if optimizer is not None and (root / OPTIMIZER_FILE).exists():
        try:
            optimizer.load_state_dict(torch.load(root / OPTIMIZER_FILE, map_location="cpu"))
        except (ValueError, RuntimeError):
            # Different parameter set (e.g. fresh adapters): keep training, lose momentum only.
            pass
    if (root / RNG_FILE).exists():
        torch.set_rng_state(torch.load(root / RNG_FILE, map_location="cpu"))
    return {"state": TrainState.from_dict(payload.get("trainer", {})),
            "extra": payload.get("extra", {})}


# --------------------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------------------
class Trainer:
    """Micro-batch loop with gradient accumulation and exact mid-epoch resume."""

    def __init__(self, model, *, optimizer, examples: Sequence[dict], pad_id: int, device="cpu",
                 micro_batch: int = 1, grad_accum: int = 1, epochs: float = 1.0,
                 lr: float = 1e-4, warmup_steps: int = 0, max_grad_norm: float = 1.0,
                 chunk_tokens: int = DEFAULT_CHUNK_TOKENS, seed: int = 0,
                 log_every: int = 1, on_step: Callable | None = None,
                 on_checkpoint: Callable | None = None, log: Callable[[str], None] = print):
        if not examples:
            raise ValueError("no training rows")
        self.model = model
        self.opt = optimizer
        self.examples = examples
        self.pad_id = pad_id
        self.device = device
        self.micro_batch = micro_batch
        self.grad_accum = grad_accum
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.chunk_tokens = chunk_tokens
        self.seed = seed
        self.log_every = log_every
        self.on_step = on_step
        self.on_checkpoint = on_checkpoint
        self.say = log or (lambda _m: None)
        self.state = TrainState()

    # ---- schedule ----------------------------------------------------------------------------
    def steps_per_epoch(self) -> int:
        return max(1, math.ceil(len(self.examples) / (self.micro_batch * self.grad_accum)))

    def total_steps(self) -> int:
        return max(1, int(math.ceil(self.steps_per_epoch() * self.epochs)))

    def lr_at(self, step: int) -> float:
        total = self.total_steps()
        if self.warmup and step < self.warmup:
            return self.lr * (step + 1) / self.warmup
        prog = min(1.0, step / max(1, total - self.warmup))
        return self.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    # ---- run ----------------------------------------------------------------------------------
    def fit(self, resume_from: Path | str | None = None) -> TrainState:
        if resume_from is not None:
            loaded = load_checkpoint(resume_from, model=self.model, optimizer=self.opt)
            self.state = loaded["state"]
            self.say(f"resumed at step {self.state.step}, epoch {self.state.epoch}, "
                     f"{self.state.micro_in_epoch} micro-batches into it")
        self.model.train()
        done_steps = self.total_steps()
        try:
            self._run(done_steps)
        except StopTraining:
            self.say(f"stopped on request at step {self.state.step} "
                     f"(epoch {self.state.epoch}, {self.state.micro_in_epoch} micro-batches in)")
        return self.state

    def _step(self, loss, epoch: int) -> None:
        """One optimizer step: clip, update, record, fire callbacks."""
        lr = self.lr_at(self.state.step)
        for group in self.opt.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        value = float(loss.detach())
        self.state.step += 1
        self.state.best_loss = min(self.state.best_loss, value)
        self.state.history.append({"step": self.state.step, "epoch": epoch,
                                   "loss": round(value, 6), "lr": round(lr, 9)})
        if self.on_step:
            self.on_step(self.state, value)
        if self.on_checkpoint:
            self.on_checkpoint(self.state)
        if self.log_every and self.state.step % self.log_every == 0:
            self.say(f"step {self.state.step}/{self.total_steps()} epoch {epoch} loss {value:.4f}")

    def _run(self, done_steps: int) -> None:
        while self.state.step < done_steps:
            epoch = self.state.epoch
            order = epoch_order(len(self.examples), seed=self.seed, epoch=epoch)
            consumed = self.state.micro_in_epoch
            if consumed % self.micro_batch:
                raise ValueError(f"resume point {consumed} is not a multiple of micro_batch "
                                 f"{self.micro_batch}; the checkpoint was not written at a step boundary")
            stop = False
            self.opt.zero_grad(set_to_none=True)
            accumulated = 0
            loss = None
            for start in range(consumed, len(order), self.micro_batch):
                if self.state.step >= done_steps:
                    stop = True
                    break
                rows = [self.examples[i] for i in order[start:start + self.micro_batch]]
                batch = collate(rows, self.pad_id)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # An all-ones attention mask carries no information but makes flash-attn take a
                # different, measurably more expensive path on the 6 GB card. With micro_batch=1
                # every batch is unpadded; pass None instead and keep the same semantics.
                mask = batch["attention_mask"]
                attention_mask = None if bool(mask.all()) else mask
                loss = chunked_lm_loss(self.model, batch["input_ids"],
                                       attention_mask=attention_mask,
                                       labels=batch["labels"], chunk_tokens=self.chunk_tokens)
                (loss / self.grad_accum).backward()
                accumulated += 1
                self.state.micro_in_epoch = start + len(rows)
                self.state.samples_seen += len(rows)
                self.state.tokens_seen += int(batch["attention_mask"].sum())
                if accumulated == self.grad_accum:
                    self._step(loss, epoch)
                    accumulated = 0
            # A short final window (fewer micro-batches than grad_accum) still gets its update;
            # without this a tiny epoch would loop forever without ever stepping.
            if not stop and accumulated and self.state.step < done_steps:
                self._step(loss, epoch)
                accumulated = 0
            if not stop:
                self.state.epoch = epoch + 1
                self.state.micro_in_epoch = 0
