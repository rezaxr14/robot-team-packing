"""Profile selector, isolation, resume state and capability resolution."""
from __future__ import annotations

import json

import pytest

from rtp.profiles import (
    DEFAULT_PROFILE,
    FROZEN_NAME_PATTERNS,
    PROFILE_NAMES,
    PROFILES,
    VISION_LORA_TARGETS,
    Caps,
    detect_caps,
    get_profile,
    load_state,
    profiles,
    record_stage,
    resolve_runtime,
    save_state,
    stage_done,
    with_overrides,
)


def test_all_expected_profiles_exist():
    assert tuple(PROFILES) == PROFILE_NAMES
    assert DEFAULT_PROFILE in PROFILES


def test_profile_selection_precedence(monkeypatch):
    monkeypatch.delenv("RTP_PROFILE", raising=False)
    assert get_profile().name == DEFAULT_PROFILE
    monkeypatch.setenv("RTP_PROFILE", "kaggle_t4x2")
    assert get_profile().name == "kaggle_t4x2"
    assert get_profile("local_4090").name == "local_4090"   # explicit wins


def test_unknown_profile_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("RTP_PROFILE", raising=False)
    with pytest.raises(KeyError) as e:
        get_profile("a100")
    assert "local_4050" in str(e.value)


def test_isolated_output_trees(tmp_path):
    table = profiles(tmp_path)
    seen = {}
    for name, p in table.items():
        for kind, path in p.paths.as_dict().items():
            assert str(tmp_path) in path, (name, kind)
            assert path not in seen, f"{path} shared by {name} and {seen[path]}"
            seen[path] = name
        # runs tree is per profile, never the profile root itself
        assert p.paths.runs.parent == tmp_path / "runs"


def test_ensure_dirs_creates_only_own_tree(tmp_path):
    table = profiles(tmp_path)
    a, b = table["local_4050"], table["local_4090"]
    a.ensure_dirs()
    for d in (a.paths.ckpt, a.paths.adapters, a.paths.logs, a.paths.reports, a.paths.data):
        assert d.is_dir()
    assert not b.paths.runs.exists()
    assert not (tmp_path / "runs" / "local_4090").exists()


def test_env_vars_keep_subprocesses_in_profile(tmp_path):
    p = profiles(tmp_path)["kaggle_t4x2"]
    env = p.env_vars()
    assert env["RTP_PROFILE"] == "kaggle_t4x2"
    assert env["HF_HOME"].startswith(str(p.paths.models))
    assert env["RTP_RUNS"] == str(p.paths.runs)


def test_hardware_choices_match_the_decisions():
    p = PROFILES
    assert (p["local_4050"].tuning, p["local_4050"].vram_gb) == ("qlora", 6.0)
    assert p["local_4050"].dtype == "bfloat16"
    assert (p["local_4090"].tuning, p["local_4090"].dtype, p["local_4090"].attn) == (
        "lora", "bfloat16", "flash_attention_2")
    t4 = p["kaggle_t4x2"]
    assert (t4.tuning, t4.dtype, t4.attn, t4.n_gpu) == ("lora", "float16", "sdpa", 2)
    assert t4.serve["tensor_parallel"] == 2 and t4.serve["enforce_eager"] is True
    for stub in ("tpu_v5e8", "tpu_v6e1", "tpu_v5e1"):
        assert p[stub].supported is False and p[stub].paper_class == "stub"


def test_every_block_type_is_adapted():
    targets = set(PROFILES["local_4090"].lora_targets)
    # full attention + gated deltanet projections + mlp
    assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= targets
    assert {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"} <= targets
    assert {"gate_proj", "up_proj", "down_proj"} <= targets
    # vision attention, and the merger shares linear_fc1/linear_fc2 with the vision MLP
    assert set(VISION_LORA_TARGETS) <= targets
    # state params must never be adapter targets
    for pattern in FROZEN_NAME_PATTERNS:
        assert not any(pattern in t for t in targets)


def test_effective_batch_and_config_hash(tmp_path):
    p = profiles(tmp_path)
    assert p["local_4050"].effective_batch == 1 * 1 * 16
    assert p["kaggle_t4x2"].effective_batch == 2 * 2 * 8

    h = p["local_4090"].config_hash()
    assert h == profiles("/some/other/root")["local_4090"].config_hash()   # portable
    assert h != with_overrides(p["local_4090"], micro_batch=2).config_hash()
    assert h != p["local_4050"].config_hash()


def test_with_overrides_does_not_mutate_the_table(tmp_path):
    p = profiles(tmp_path)["local_4090"]
    v = with_overrides(p, max_len=1024)
    assert v.max_len == 1024 and p.max_len == 16384
    with pytest.raises(KeyError):
        with_overrides(p, nonsense=1)


def test_state_roundtrip_and_resume_semantics(tmp_path):
    p = profiles(tmp_path)["kaggle_t4x2"]
    assert load_state(p) == {}
    assert not stage_done(p, "sft")

    record_stage(p, "sft", status="done", last_step=120, config_hash="abc")
    assert json.loads(p.paths.state.read_text())["sft"]["last_step"] == 120
    assert stage_done(p, "sft") and stage_done(p, "sft", "abc")
    assert not stage_done(p, "sft", "def")          # config changed -> rerun
    assert not stage_done(p, "grpo")

    record_stage(p, "sft", status="failed", last_step=130, config_hash="abc")
    assert not stage_done(p, "sft", "abc")          # a failure is never "done"

    record_stage(p, "grpo", status="done")
    assert load_state(p)["grpo"]["config_hash"] == p.config_hash()

    save_state(p, {})                                # explicit reset
    assert load_state(p) == {}


def test_states_do_not_leak_between_profiles(tmp_path):
    table = profiles(tmp_path)
    record_stage(table["local_4050"], "sft", status="done", last_step=7)
    assert load_state(table["local_4090"]) == {}
    assert not table["local_4090"].paths.state.exists()


CUDA_ADVANCED = Caps(cuda=True, cc=(8, 9), bf16=True, flash_attn=True, bitsandbytes=True)


def test_resolve_runtime_keeps_supported_choices(tmp_path):
    p = profiles(tmp_path)["local_4090"]
    attn, dtype, notes = resolve_runtime(p, CUDA_ADVANCED)
    assert (attn, dtype, notes) == ("flash_attention_2", "bfloat16", [])


def test_resolve_runtime_falls_back_without_flash_attn(tmp_path):
    p = profiles(tmp_path)["local_4090"]
    caps = Caps(cuda=True, cc=(8, 9), bf16=True, flash_attn=False)
    attn, dtype, notes = resolve_runtime(p, caps)
    assert (attn, dtype) == ("sdpa", "bfloat16")
    assert any("flash_attn not importable" in n for n in notes)


def test_resolve_runtime_degrades_on_turing(tmp_path):
    p = profiles(tmp_path)["local_4090"]          # asks for bf16 + FA2
    turing = Caps(cuda=True, cc=(7, 5), bf16=False, flash_attn=True)
    attn, dtype, notes = resolve_runtime(p, turing)
    assert (attn, dtype) == ("sdpa", "float16")
    assert any("bf16 unsupported" in n for n in notes)
    assert any("cc >= 8.0" in n for n in notes)


def test_resolve_runtime_flags_missing_bitsandbytes(tmp_path):
    p = profiles(tmp_path)["local_4050"]
    caps = Caps(cuda=True, cc=(8, 9), bf16=True, flash_attn=True, bitsandbytes=False)
    _, _, notes = resolve_runtime(p, caps)
    assert any("bitsandbytes missing" in n for n in notes)


def test_resolve_runtime_on_cpu_is_honest(tmp_path):
    p = profiles(tmp_path)["local_4050"]
    attn, dtype, notes = resolve_runtime(p, Caps())
    assert any("no CUDA" in n for n in notes)


def test_detect_caps_never_raises():
    caps = detect_caps()
    assert isinstance(caps, Caps)
    assert isinstance(caps.cc, tuple) and len(caps.cc) == 2
