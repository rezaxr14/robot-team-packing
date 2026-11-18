"""Load the Brain for training, run SFT, and keep the run resumable.

This is the module the notebooks call, so every knob comes from the profile (``rtp/profiles.py``)
instead of being re-typed per notebook:

* ``tuning: qlora`` -> 4-bit nf4 base + LoRA; ``lora`` -> bf16 base + LoRA; ``full`` -> all params
  except the DeltaNet state tensors (``FROZEN_NAME_PATTERNS``).
* dtype and attention backend are *resolved* against the real box (``resolve_runtime``): a 4050
  asking for FlashAttention-2 on Turing gets fp16 + sdpa with a printed reason, not a crash.
* LoRA target names come from the profile and are filtered to modules the checkpoint really has,
  because the vision-merger names only exist if the vision tower is in this checkpoint (G-1 report).

``run_sft`` is written against an injectable ``loader`` so the orchestration (fetch latest ->
train -> push every N -> stop -> resume) is testable on CPU with a toy model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from rtp.checkpoints import CheckpointPolicy, fetch_latest
from rtp.profiles import FROZEN_NAME_PATTERNS, Profile, resolve_runtime
from rtp.trainer import TrainState, Trainer, StopTraining, tokenize_pairs


# --------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------
def load_pairs(path: Path | str) -> list[dict]:
    """Read a JSONL of ``{"prompt", "completion"}`` rows, ignoring extra columns like ``meta``."""
    rows = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "prompt" not in row or "completion" not in row:
            raise ValueError(f"{path}:{line_no}: needs both 'prompt' and 'completion' "
                             f"(got {sorted(row)})")
        rows.append({"prompt": row["prompt"], "completion": row["completion"]})
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


# --------------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------------
def existing_targets(model, wanted: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``wanted`` into module names the model has and names it does not.

    PEFT raises when a target module is absent, and the profile lists both text and vision names;
    a text-only checkpoint legitimately lacks ``linear_fc1``. Reporting the missing names keeps
    "which blocks did we actually adapt" answerable instead of silently narrower than the plan.
    """
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    have = [t for t in wanted if t in present]
    missing = [t for t in wanted if t not in present]
    return have, missing


def frozen_parameters(model) -> list[str]:
    """Names matching ``FROZEN_NAME_PATTERNS`` (DeltaNet recurrence state, conv, norms, A_log/dt_bias)."""
    out = []
    for name, _ in model.named_parameters():
        if any(pat in name for pat in FROZEN_NAME_PATTERNS):
            out.append(name)
    return out


def _torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.float32)


def load_for_training(profile: Profile, model_dir: Path | str, *, log: Callable[[str], None] = print):
    """Return ``(model, tokenizer, info)`` ready for SFT on this profile.

    ``model_dir`` is a local, manifest-verified snapshot of the base model (resolve it with
    ``rtp.remotes.RemoteStore.resolve`` — the notebook does exactly that).
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    say = log or (lambda _m: None)
    attn, dtype_name, notes = resolve_runtime(profile)
    for note in notes:
        say(f"runtime: {note}")
    dtype = _torch_dtype(dtype_name)
    model_dir = Path(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": dtype, "attn_implementation": attn}
    if attn == "flash_attention_2":
        kwargs["attn_implementation"] = "flash_attention_2"
    quantized = False
    if profile.uses_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": 0}
            quantized = True
        except Exception as exc:                          # noqa: BLE001
            say(f"4-bit unavailable ({type(exc).__name__}); falling back to {dtype_name} LoRA")

    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    device = "cuda" if (torch.cuda.is_available() and not quantized) else "cpu"
    if not quantized:
        model = model.to(device)
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=profile.grad_ckpt)
    if profile.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()            # 4-bit + checkpointing needs this

    info = {"attn": attn, "dtype": dtype_name, "quantized": quantized, "tuning": profile.tuning,
            "device": "cuda" if quantized else device}
    if profile.tuning == "full":
        frozen = set(frozen_parameters(model))
        for name, param in model.named_parameters():
            param.requires_grad = name not in frozen
        info["frozen"] = sorted(frozen)[:20]
        say(f"full tuning: {len(frozen)} state tensors frozen (DeltaNet/conv/norm)")
        return model, tokenizer, info

    have, missing = existing_targets(model, profile.lora_targets)
    if not have:
        raise RuntimeError(f"none of the profile's LoRA targets exist in {model_dir}: "
                           f"{list(profile.lora_targets)}")
    if missing:
        say(f"LoRA targets absent from this checkpoint (not adapted): {missing}")
    model = get_peft_model(model, LoraConfig(
        r=profile.lora_r, lora_alpha=profile.lora_alpha, lora_dropout=profile.lora_dropout,
        target_modules=have, task_type="CAUSAL_LM"))
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    info["lora_targets"] = have
    info["lora_missing"] = missing
    return model, tokenizer, info


def build_optimizer(model, lr: float, weight_decay: float = 0.0):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters: check the tuning mode and LoRA targets")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# --------------------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------------------
@dataclass
class SFTResult:
    state: TrainState
    out_dir: Path
    resumed_from: int | None


def run_sft(profile: Profile, *, model_dir: Path | str, data: Path | str | Sequence[dict],
            store=None, stage: str = "sft", epochs: float = 1.0, lr: float = 1e-4,
            every: int = 50, keep: int = 2, warmup_steps: int = 0, seed: int = 0,
            chunk_tokens: int | None = None, max_rows: int | None = None,
            loader: Callable | None = None, log: Callable[[str], None] = print,
            stop_after: int | None = None) -> SFTResult:
    """Fetch the newest checkpoint, train, and push a checkpoint every ``every`` steps.

    ``stop_after`` exists for notebooks and tests: the run stops cleanly at that step with the
    checkpoint already pushed, which is exactly the "session ended, resume tomorrow" path.
    """
    say = log or (lambda _m: None)
    profile.ensure_dirs()
    out_dir = profile.paths.ckpt / stage
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pairs(data) if not isinstance(data, (list, tuple)) else [{"prompt": r["prompt"],
                                                                         "completion": r["completion"]}
                                                                        for r in data]
    if max_rows:
        rows = rows[:max_rows]

    resumed_from = None
    resume_path = None
    if store is not None:
        resumed_from, resume_path = fetch_latest(store, profile.name, stage,
                                                 out_dir / "_resume", log=say)

    loader = loader or load_for_training
    model, tokenizer, info = loader(profile, model_dir, log=say)
    say(f"loaded: {info}")

    examples = tokenize_pairs(rows, tokenizer, max_len=profile.max_len)
    if not examples:
        raise ValueError("no usable rows after tokenisation (all completions empty?)")
    say(f"sft: {len(examples)} rows, max_len {profile.max_len}, "
        f"micro {profile.micro_batch} x accum {profile.grad_accum} on {profile.n_gpu} device(s)")

    optimizer = build_optimizer(model, lr)
    policy = CheckpointPolicy(root=out_dir, profile=profile.name, stage=stage, store=store,
                              every=every, keep=keep, tokenizer=tokenizer,
                              extra={"stage": stage, "profile": profile.name}, log=say)
    policy.bind(model, optimizer)

    def on_checkpoint(state: TrainState) -> None:
        policy(state)
        if stop_after is not None and state.step >= stop_after:
            raise StopTraining("stop_after reached")

    trainer = Trainer(model, optimizer=optimizer, examples=examples,
                      pad_id=tokenizer.pad_token_id, micro_batch=profile.micro_batch,
                      grad_accum=profile.grad_accum, epochs=epochs, lr=lr,
                      warmup_steps=warmup_steps, seed=seed,
                      chunk_tokens=chunk_tokens or (1024 if profile.vram_gb >= 20 else 128),
                      on_checkpoint=on_checkpoint, device=info.get("device", "cpu"), log=say)
    state = trainer.fit(resume_from=resume_path)

    adapters = profile.paths.adapters / stage
    adapters.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapters)
    tokenizer.save_pretrained(adapters)
    say(f"final adapters -> {adapters} (step {state.step})")
    return SFTResult(state=state, out_dir=out_dir, resumed_from=resumed_from)
