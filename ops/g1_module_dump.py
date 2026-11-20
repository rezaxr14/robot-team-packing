"""G-1: dump the module tree that the downloaded Qwen3.5-4B checkpoint really defines.

  python ops/g1_module_dump.py --model $HOME/models/Qwen3.5-4B --out reports/qwen35_modules.json

No weights are loaded and no device is touched: the model is built on the meta device from the
checkpoint's own config, so a 6 GB laptop can run this gate. The JSON is the authority for adapter
targets (plan §8): every name it lists exists in this checkpoint, not in a guess.

Exit code 0 = gate PASS, 1 = FAIL (details in `checks`).
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Adapter block map (plan §8 / D-07..D-09). Keys are class names; values are the own-parameter
# suffixes a LoRA adapter should cover. `checks["adapter_targets"]` verifies each one really exists.
ADAPTER_TARGETS = {
    "Qwen3_5Attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "Qwen3_5GatedDeltaNet": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
    "Qwen3_5MLP": ["gate_proj", "up_proj", "down_proj"],
    "Qwen3_5VisionAttention": ["qkv", "proj"],
    "Qwen3_5VisionMLP": ["linear_fc1", "linear_fc2"],
    "Qwen3_5VisionPatchMerger": ["linear_fc1", "linear_fc2"],
}
# Must stay frozen (plan §8): exposed here so the training config can be diffed against reality.
FROZEN_SUFFIXES = ["A_log", "dt_bias", "conv1d", "input_layernorm", "post_attention_layernorm", "norm"]


def build_meta_model(model_dir: str):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(model_dir)
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(cfg)
    return cfg, model


def dump(model) -> dict:
    classes: Counter = Counter()
    own_params: dict = {}
    n_modules = 0
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        classes[cls] += 1
        n_modules += 1
        if cls in ADAPTER_TARGETS:
            # LoRA names the *child* modules (q_proj is a Linear child), plus direct parameters
            # such as A_log/dt_bias that are frozen rather than adapted.
            own = own_params.setdefault(cls, set())
            own.update(name for name, _ in mod.named_children())
            own.update(pname.rsplit(".", 1)[-1] for pname, _ in mod.named_parameters(recurse=False))
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "n_modules": n_modules,
        "n_parameters": n_params,
        "classes": dict(classes.most_common()),
        "own_parameters": {k: sorted(v) for k, v in own_params.items()},
    }


def checks_for(classes: dict, own: dict) -> dict:
    def has(cls, *suffixes):
        got = set(own.get(cls, []))
        return {f"{cls}.{s}": (s in got) for s in suffixes}

    adapter = {}
    for cls, suffixes in ADAPTER_TARGETS.items():
        adapter.update(has(cls, *suffixes))
    return {
        "adapter_targets": adapter,
        "adapter_targets_all_present": all(adapter.values()),
        "dense_no_experts": not any("expert" in c.lower() or "moe" in c.lower() for c in classes),
        "gated_deltanet_present": classes.get("Qwen3_5GatedDeltaNet", 0) > 0,
        "full_attention_present": classes.get("Qwen3_5Attention", 0) > 0,
        "vision_blocks_present": classes.get("Qwen3_5VisionAttention", 0) > 0,
        "merger_present": classes.get("Qwen3_5VisionPatchMerger", 0) > 0,
        # verified in the checkpoint: the DeltaNet recurrence has no `D` parameter
        "gated_deltanet_no_D_param": "D" not in set(own.get("Qwen3_5GatedDeltaNet", [])),
        "frozen_suffixes_seen": {s: any(s in p for p in {q for ps in own.values() for q in ps})
                                 for s in FROZEN_SUFFIXES},
    }


def config_summary(cfg) -> dict:
    text = getattr(cfg, "text_config", None)
    vision = getattr(cfg, "vision_config", None)
    layers = getattr(text, "layer_types", None)
    out = {
        "model_type": cfg.model_type,
        "architectures": cfg.architectures,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "vocab_size": getattr(text or cfg, "vocab_size", None),
        "hidden_size": getattr(text or cfg, "hidden_size", None),
        "num_hidden_layers": getattr(text or cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(text or cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(text or cfg, "num_key_value_heads", None),
        "head_dim": getattr(text or cfg, "head_dim", None),
        "full_attention_interval": getattr(text or cfg, "full_attention_interval", None),
        "mtp_num_hidden_layers": getattr(text or cfg, "mtp_num_hidden_layers", None),
        "linear_num_key_heads": getattr(text or cfg, "linear_num_key_heads", None),
        "linear_num_value_heads": getattr(text or cfg, "linear_num_value_heads", None),
        "linear_key_head_dim": getattr(text or cfg, "linear_key_head_dim", None),
        "linear_value_head_dim": getattr(text or cfg, "linear_value_head_dim", None),
        "linear_conv_kernel_dim": getattr(text or cfg, "linear_conv_kernel_dim", None),
        "mamba_ssm_dtype": getattr(text or cfg, "mamba_ssm_dtype", None),
        "vision_hidden_size": getattr(vision, "hidden_size", None),
        "vision_depth": getattr(vision, "depth", None),
        "vision_patch_size": getattr(vision, "patch_size", None),
        "vision_spatial_merge_size": getattr(vision, "spatial_merge_size", None),
    }
    if layers:
        out["layer_types"] = layers
        out["layer_type_counts"] = dict(Counter(layers))
    return out


def header_facts(model_dir: str) -> dict:
    """Tensor shapes read from the safetensors headers (no weights): the authoritative parameter count."""
    import glob
    from safetensors import safe_open
    total, tensors, branches = 0, 0, {}
    for f in sorted(glob.glob(str(Path(model_dir) / "*.safetensors"))):
        with safe_open(f, framework="np") as g:
            for k in g.keys():
                n = 1
                for d in g.get_slice(k).get_shape():
                    n *= d
                total += n
                tensors += 1
                parts = k.split(".")
                branch = ".".join(parts[1:2]) if len(parts) > 2 else parts[0]
                branches[branch] = branches.get(branch, 0) + n
    keys = set()
    idx = Path(model_dir) / "model.safetensors.index.json"
    if idx.exists():
        keys = set(json.loads(idx.read_text()).get("weight_map", {}))
    return {
        "tensor_count": tensors,
        "parameter_count": total,
        "parameters_by_branch": {k: v for k, v in sorted(branches.items(), key=lambda kv: -kv[1])},
        "has_lm_head": any(k.startswith("lm_head.") for k in keys),
        "has_mtp": any(k.startswith("mtp.") for k in keys),
    }


def index_facts(model_dir: str) -> dict:
    """Parameter index facts read from the checkpoint itself (no weights)."""
    p = Path(model_dir) / "model.safetensors.index.json"
    if not p.exists():
        return {"index": "missing"}
    idx = json.loads(p.read_text())
    return {
        "index": str(p),
        "index_parameter_count": len(idx.get("weight_map", {})),
        "index_total_size_bytes": idx.get("metadata", {}).get("total_size"),
        "index_shards": sorted(set(idx.get("weight_map", {}).values())),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="reports/qwen35_modules.json")
    a = ap.parse_args(argv)

    cfg, model = build_meta_model(a.model)
    facts = dump(model)
    checks = checks_for(facts["classes"], facts["own_parameters"])
    gate_keys = [k for k in checks if k != "frozen_suffixes_seen"]
    passed = all(checks[k] for k in gate_keys)
    report = {
        "gate": "G-1",
        "model_dir": str(Path(a.model).resolve()),
        "dtype_on_meta": str(next(iter(model.parameters())).dtype),
        "config": config_summary(cfg),
        "modules": facts["classes"],
        "module_count": facts["n_modules"],
        "parameter_count_meta_graph": facts["n_parameters"],
        "parameter_count_from_headers": header_facts(a.model)["parameter_count"],
        "own_parameters": facts["own_parameters"],
        "checks": checks,
        "adapter_targets_expected": ADAPTER_TARGETS,
        "frozen_suffixes_expected": FROZEN_SUFFIXES,
        "checkpoint": {**index_facts(a.model), **header_facts(a.model)},
        "result": "PASS" if passed else "FAIL",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"G-1 module dump: {a.model}")
    print(f"  parameters {facts['n_parameters']:,}  modules {facts['n_modules']}")
    print(f"  classes   {', '.join(f'{k}x{v}' for k, v in list(facts['classes'].items())[:8])}")
    for k, v in checks.items():
        if k in ("adapter_targets", "frozen_suffixes_seen"):
            continue
        print(f"  {k:34s} {v}")
    print(f"  -> {report['result']}  ({out})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
