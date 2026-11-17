"""Hardware/run profiles: the single source of truth for every run.

A notebook cell sets only

    ACTIVE = "kaggle_t4x2"

and every other knob (dtype, attention backend, tuning mode, batch geometry, output folders,
resume file, serving flags) is derived here. The CLI equivalent is the ``RTP_PROFILE`` env var;
an explicit ``get_profile(name)`` argument wins over both.

Isolation is the point: a profile writes only inside its own ``runs/<profile>/``,
``data/<profile>/`` and ``models/<profile>/`` trees, so two profiles can train, resume and
evaluate without ever touching each other's adapters, optimizer state, logs or caches.

Capability claims (bf16, FlashAttention-2) are *resolved*, never assumed:

    impl, dtype, notes = resolve_runtime(profile, caps)

``caps`` comes from ``detect_caps()`` on a GPU box, or is passed by hand in tests. The rules are
the documented ones -- bf16 and mainstream FlashAttention-2 need Ampere (cc >= 8.0); Turing gets
fp16 + SDPA; the ``flash_attention_2`` request on an unsupported box degrades to ``sdpa`` with a
reason instead of crashing.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# adapter coverage: every block type the user asked to train, by real module name
# (verified in transformers/models/qwen3_5/modeling_qwen3_5.py, transformers 5.23.0)
# --------------------------------------------------------------------------------------------------
TEXT_LORA_TARGETS = (
    # full attention (8 layers)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # gated DeltaNet projections (24 layers)
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    # dense MLP up/down/gate
    "gate_proj", "up_proj", "down_proj",
)
# VisionAttention.qkv / .proj; VisionMLP and VisionPatchMerger both expose linear_fc1/linear_fc2,
# so these four names also cover the vision merger the user wants trained.
VISION_LORA_TARGETS = ("qkv", "proj", "linear_fc1", "linear_fc2")

# Parameters that must stay frozen even under `tuning: full`: the DeltaNet recurrence is fp32
# state math (config `mamba_ssm_dtype: float32`) and the conv/norm/A_log/dt_bias are its state.
FROZEN_NAME_PATTERNS = ("A_log", "dt_bias", "conv1d", "norm")

PROFILE_NAMES = (
    "local_4050",
    "local_4090",
    "kaggle_t4x2",
    "pro6000",
    "tpu_v5e8",
    "tpu_v6e1",
    "tpu_v5e1",
)

DEFAULT_PROFILE = "local_4050"


@dataclass(frozen=True)
class ProfilePaths:
    """Every directory a profile may write to. Two profiles never share one."""

    runs: Path
    ckpt: Path
    adapters: Path
    logs: Path
    reports: Path
    data: Path
    models: Path
    hf_home: Path

    @property
    def state(self) -> Path:
        return self.runs / "state.json"

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Profile:
    name: str
    accel: str                  # cuda | xla | cpu
    gpu: str                    # human label
    n_gpu: int
    vram_gb: float              # per device; 0 for CPU/TPU stubs
    tuning: str                 # qlora | lora | full
    dtype: str                  # bfloat16 | float16 | float32
    attn: str                   # flash_attention_2 | sdpa | eager
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    micro_batch: int            # per device
    grad_accum: int
    grad_ckpt: bool
    max_len: int
    eval_seeds: tuple[int, ...]
    serve: dict = field(default_factory=dict)
    physics: str = "isaac_newton"   # D-25: Isaac Lab 3.0 + Newton is the only stack
    paper_class: str = "dev"    # dev | reportable | stub
    supported: bool = True
    notes: str = ""
    root: Path = field(default=Path("."), compare=False, repr=False)

    # ---- derived geometry -------------------------------------------------------------------
    @property
    def paths(self) -> ProfilePaths:
        r = self.root
        runs = r / "runs" / self.name
        return ProfilePaths(
            runs=runs,
            ckpt=runs / "ckpt",
            adapters=runs / "adapters",
            logs=runs / "logs",
            reports=runs / "reports",
            data=r / "data" / self.name,
            models=r / "models" / self.name,
            hf_home=r / "models" / self.name / "hf",
        )

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * max(self.n_gpu, 1) * self.grad_accum

    @property
    def uses_4bit(self) -> bool:
        return self.tuning == "qlora"

    def env_vars(self) -> dict:
        """Environment a subprocess needs to stay inside this profile's tree."""
        p = self.paths
        return {
            "RTP_PROFILE": self.name,
            "RTP_ROOT": str(self.root),
            "HF_HOME": str(p.hf_home),
            "HF_HUB_CACHE": str(p.hf_home / "hub"),
            "RTP_RUNS": str(p.runs),
            "RTP_DATA": str(p.data),
        }

    def ensure_dirs(self) -> ProfilePaths:
        p = self.paths
        for d in (p.ckpt, p.adapters, p.logs, p.reports, p.data, p.hf_home):
            d.mkdir(parents=True, exist_ok=True)
        return p

    # ---- reproducibility --------------------------------------------------------------------
    def config_dict(self) -> dict:
        """Config that affects a run's results. Paths are excluded so hashes are portable."""
        d = asdict(self)
        d.pop("root", None)
        d["lora_targets"] = list(self.lora_targets)
        d["eval_seeds"] = list(self.eval_seeds)
        d["effective_batch"] = self.effective_batch
        return d

    def config_hash(self) -> str:
        blob = json.dumps(self.config_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _p(root: Path, **kw) -> Profile:
    return Profile(root=root, **kw)


def _all_targets() -> tuple[str, ...]:
    return TEXT_LORA_TARGETS + VISION_LORA_TARGETS


def _profile_table(root: Path) -> dict[str, Profile]:
    text_vision = _all_targets()
    return {
        # 6 GB laptop: 4-bit base + bf16 compute, the only way 4.66 B params fit with adapters.
        "local_4050": _p(
            root,
            name="local_4050", accel="cuda", gpu="RTX 4050 Laptop", n_gpu=1, vram_gb=6.0,
            tuning="qlora", dtype="bfloat16", attn="flash_attention_2",
            lora_r=32, lora_alpha=64, lora_dropout=0.05, lora_targets=text_vision,
            micro_batch=1, grad_accum=16, grad_ckpt=True, max_len=8192,
            eval_seeds=(0, 1, 2), serve={"mode": "4bit", "backend": "hf"},
            physics="isaac_newton", paper_class="dev",
            notes="dev/SMOKE: QLoRA is the only way the 4.66 B model fits; tiny Newton env counts only [G->G-3].",
        ),
        # 24 GB: bf16 LoRA + FlashAttention-2, the local reference run.
        "local_4090": _p(
            root,
            name="local_4090", accel="cuda", gpu="RTX 4090", n_gpu=1, vram_gb=24.0,
            tuning="lora", dtype="bfloat16", attn="flash_attention_2",
            lora_r=32, lora_alpha=64, lora_dropout=0.05, lora_targets=text_vision,
            micro_batch=4, grad_accum=4, grad_ckpt=False, max_len=16384,
            eval_seeds=(0, 1, 2, 3, 4), serve={"mode": "bf16", "backend": "hf"},
            physics="isaac_newton", paper_class="reportable",
            notes="Full local reference: bf16 LoRA + FA2, Isaac Lab 3.0 + Newton.",
        ),
        # 2x16 GB, sm75: no bf16, no mainstream FA2 -> fp16 + SDPA, DDP.
        "kaggle_t4x2": _p(
            root,
            name="kaggle_t4x2", accel="cuda", gpu="2x Tesla T4", n_gpu=2, vram_gb=16.0,
            tuning="lora", dtype="float16", attn="sdpa",
            lora_r=32, lora_alpha=64, lora_dropout=0.05, lora_targets=text_vision,
            micro_batch=2, grad_accum=8, grad_ckpt=True, max_len=8192,
            eval_seeds=(0, 1, 2, 3, 4), serve={"mode": "fp16", "backend": "vllm",
                                               "enforce_eager": True, "tensor_parallel": 2},
            physics="isaac_newton", paper_class="reportable",
            notes="Headline training profile: fp16 + SDPA (no bf16/FA2). cc 7.5 meets Newtons floor; G-3 records sim pass/fail.",
        ),
        # 96 GB workstation card (future): headroom for r64 or full tuning.
        "pro6000": _p(
            root,
            name="pro6000", accel="cuda", gpu="RTX Pro 6000", n_gpu=1, vram_gb=96.0,
            tuning="lora", dtype="bfloat16", attn="flash_attention_2",
            lora_r=64, lora_alpha=128, lora_dropout=0.05, lora_targets=text_vision,
            micro_batch=8, grad_accum=4, grad_ckpt=False, max_len=32768,
            eval_seeds=(0, 1, 2, 3, 4, 5, 6, 7), serve={"mode": "bf16", "backend": "vllm"},
            physics="isaac_newton", paper_class="reportable",
            notes="Reserved: added when the card exists; no run yet.",
        ),
        # TPU targets: declared so the profile selector is complete, not claimable.
        "tpu_v5e8": _p(
            root,
            name="tpu_v5e8", accel="xla", gpu="TPU v5e-8", n_gpu=8, vram_gb=0.0,
            tuning="lora", dtype="bfloat16", attn="sdpa",
            lora_r=32, lora_alpha=64, lora_dropout=0.05, lora_targets=text_vision,
            micro_batch=1, grad_accum=16, grad_ckpt=True, max_len=8192, eval_seeds=(0,),
            serve={"mode": "xla", "backend": "none"},
            physics="isaac_newton", paper_class="stub", supported=False,
            notes="STUB: no access, no XLA training path implemented; supported=False.",
        ),
        "tpu_v6e1": _p(
            root,
            name="tpu_v6e1", accel="xla", gpu="TPU v6e-1", n_gpu=1, vram_gb=0.0,
            tuning="lora", dtype="bfloat16", attn="sdpa",
            lora_r=32, lora_alpha=64, lora_dropout=0.05, lora_targets=text_vision,
            micro_batch=1, grad_accum=16, grad_ckpt=True, max_len=8192, eval_seeds=(0,),
            serve={"mode": "xla", "backend": "none"},
            physics="isaac_newton", paper_class="stub", supported=False,
            notes="STUB: see tpu_v5e8.",
        ),
        "tpu_v5e1": _p(
            root,
            name="tpu_v5e1", accel="xla", gpu="TPU v5e-1", n_gpu=1, vram_gb=0.0,
            tuning="lora", dtype="bfloat16", attn="sdpa",
            lora_r=32, lora_alpha=64, lora_dropout=0.05, lora_targets=text_vision,
            micro_batch=1, grad_accum=16, grad_ckpt=True, max_len=8192, eval_seeds=(0,),
            serve={"mode": "xla", "backend": "none"},
            physics="isaac_newton", paper_class="stub", supported=False,
            notes="STUB: see tpu_v5e8.",
        ),
    }


def repo_root() -> Path:
    return Path(os.environ.get("RTP_ROOT") or Path(__file__).resolve().parents[1])


def profiles(root: Path | str | None = None) -> dict[str, Profile]:
    """The profile dict. ``root`` defaults to ``$RTP_ROOT`` or the repo checkout."""
    return _profile_table(Path(root) if root is not None else repo_root())


#: Static view for introspection (paths anchored at the repo root at import time).
PROFILES: dict[str, Profile] = profiles()


def get_profile(name: str | None = None, root: Path | str | None = None) -> Profile:
    """Resolve the active profile: explicit name > ``$RTP_PROFILE`` > ``local_4050``."""
    chosen = name or os.environ.get("RTP_PROFILE") or DEFAULT_PROFILE
    table = profiles(root)
    if chosen not in table:
        raise KeyError(f"unknown profile {chosen!r}; choose from {sorted(table)}")
    return table[chosen]


def with_overrides(profile: Profile, **kw) -> Profile:
    """Derive a variant (ablation knobs) without mutating the shared table."""
    unknown = set(kw) - set(profile.__dataclass_fields__)
    if unknown:
        raise KeyError(f"unknown profile fields: {sorted(unknown)}")
    return replace(profile, **kw)


# --------------------------------------------------------------------------------------------------
# runtime capability resolution: never claim bf16/FA2 that the box cannot do
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Caps:
    """What the machine actually offers. ``detect_caps()`` fills this; tests pass it directly."""

    cuda: bool = False
    cc: tuple[int, int] = (0, 0)          # compute capability, e.g. (8, 9)
    bf16: bool = False
    flash_attn: bool = False             # importable flash_attn
    bitsandbytes: bool = False
    torchao: bool = False


def detect_caps() -> Caps:
    """Best-effort probe. Import failures are facts, not errors."""
    try:
        import torch
    except Exception:                                    # noqa: BLE001
        return Caps()
    cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    cc = torch.cuda.get_device_capability(0) if cuda else (0, 0)
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    try:
        import flash_attn  # noqa: F401
        fa = True
    except Exception:                                    # noqa: BLE001
        fa = False
    try:
        import bitsandbytes  # noqa: F401
        bnb = True
    except Exception:                                    # noqa: BLE001
        bnb = False
    return Caps(cuda=cuda, cc=cc, bf16=bf16, flash_attn=fa, bitsandbytes=bnb)


def resolve_runtime(profile: Profile, caps: Caps | None = None) -> tuple[str, str, list[str]]:
    """Return ``(attn_impl, dtype, notes)`` that is actually usable on ``caps``.

    Rules, from the FlashAttention-2 README and torch's own guards:
      * bf16 needs Ampere or newer; on Turing use fp16.
      * mainstream FlashAttention-2 needs Ampere/Ada/Hopper (cc >= 8.0) and fp16/bf16.
      * QLoRA needs bitsandbytes; without it we fall back to fp16/bf16 LoRA and say so.
    """
    caps = caps or detect_caps()
    notes: list[str] = []
    attn, dtype = profile.attn, profile.dtype

    if profile.accel != "cuda" or not caps.cuda:
        notes.append(f"no CUDA on this box: running CPU/XLA path, attn={attn} kept for record")
        if dtype == "bfloat16" and not caps.bf16:
            notes.append("bf16 unverified off-CUDA; training path is not exercised here")
        return attn, dtype, notes

    if dtype == "bfloat16" and not (caps.bf16 and caps.cc >= (8, 0)):
        dtype = "float16"
        notes.append(f"bf16 unsupported on cc {caps.cc[0]}.{caps.cc[1]}; using float16")

    if attn == "flash_attention_2":
        if caps.cc < (8, 0):
            attn = "sdpa"
            notes.append(f"FlashAttention-2 needs cc >= 8.0 (got {caps.cc[0]}.{caps.cc[1]}); using sdpa")
        elif not caps.flash_attn:
            attn = "sdpa"
            notes.append("flash_attn not importable; using sdpa")

    if profile.uses_4bit and not caps.bitsandbytes:
        notes.append("bitsandbytes missing: QLoRA unavailable, run install.sh before training")

    return attn, dtype, notes


# --------------------------------------------------------------------------------------------------
# per-profile resume state: {stage: {status, last_step, config_hash}}
# --------------------------------------------------------------------------------------------------
def load_state(profile: Profile) -> dict:
    path = profile.paths.state
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def save_state(profile: Profile, state: dict) -> Path:
    path = profile.paths.state
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)
    return path


def stage_done(profile: Profile, stage: str, config_hash: str | None = None) -> bool:
    """A stage is done iff it succeeded under the same config hash (so edits force a rerun)."""
    rec = load_state(profile).get(stage) or {}
    if rec.get("status") != "done":
        return False
    if config_hash is None:
        return True
    return rec.get("config_hash") == config_hash


def record_stage(profile: Profile, stage: str, *, status: str, last_step: int = 0,
                 config_hash: str | None = None) -> dict:
    state = load_state(profile)
    state[stage] = {
        "status": status,
        "last_step": int(last_step),
        "config_hash": config_hash if config_hash is not None else profile.config_hash(),
    }
    save_state(profile, state)
    return state


if __name__ == "__main__":                 # tiny CLI: `python -m rtp.profiles [name]`
    import sys

    p = get_profile(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({"profile": p.name, "paths": p.paths.as_dict(),
                      "config_hash": p.config_hash(),
                      "effective_batch": p.effective_batch}, indent=2))
