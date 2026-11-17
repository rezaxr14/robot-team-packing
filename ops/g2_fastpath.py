"""G-2: prove the Qwen3.5 Gated-DeltaNet fast path is really bound on this machine, then (optionally)
run a finite-loss training smoke and record tokens/s + peak VRAM.

  python ops/g2_fastpath.py --model ~/models/Qwen3.5-4B --out reports/g2_fastpath.json
  python ops/g2_fastpath.py --model ~/models/Qwen3.5-4B --steps 100 --out reports/g2_fastpath.json

What "fast path" means here (verified in `transformers/models/qwen3_5/modeling_qwen3_5.py`): the
DeltaNet calls `causal_conv1d_fn/update` and fla's `chunk_gated_delta_rule` through
`use_kernel_func_from_hub_with_fallback`, so the *resolved function's module* is the evidence -- a
missing package silently leaves the pure-torch fallback in place, which is slow but not wrong.

Exit 0 = fast path bound and (with --steps) loss finite; 1 = fallback in use or smoke failed.
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
import time
from pathlib import Path


def kernel_versions() -> dict:
    out = {}
    for mod, attr in (("torch", "__version__"), ("causal_conv1d", "__version__"), ("fla", "__version__"),
                      ("flash_attn", "__version__"), ("flash_linear_attention", "__version__")):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, attr, "installed")
        except Exception as e:                                          # noqa: BLE001
            out[mod] = f"MISSING ({type(e).__name__})"
    return out


def binding() -> dict:
    """Which implementation Qwen3.5 will call: mirror the decorator's own resolution order.

    `use_kernel_func_from_hub_with_fallback` imports the original package, resolves the function via
    `_KERNELS_INTERNAL_PATH_MAPPINGS`, and keeps the torch reference only if that fails. The wrapper
    uses functools.wraps, so `__module__` cannot show the choice -- resolution is the evidence.
    """
    from transformers.integrations import hub_kernels as H

    def resolve(name: str, package: str):
        try:
            pkg = importlib.import_module(package)
        except Exception:                                               # noqa: BLE001
            return None
        internal = H._KERNELS_INTERNAL_PATH_MAPPINGS.get(name)
        path = name if internal is None else f"{internal}.{name}"
        impl = H.resolve_internal_import(pkg, path)
        if impl is None and internal:
            try:
                impl = getattr(importlib.import_module(f"{package}.{internal}"), name, None)
            except Exception:                                           # noqa: BLE001
                impl = None
        return impl

    specs = {
        "causal_conv1d_fn": "causal_conv1d",
        "causal_conv1d_update": "causal_conv1d",
        "chunk_gated_delta_rule": "fla",
        "fused_recurrent_gated_delta_rule": "fla",
    }
    res = {}
    for name, package in specs.items():
        impl = resolve(name, package)
        mod = getattr(impl, "__module__", "") or ""
        res[name] = {"resolved": f"{mod}.{getattr(impl, '__name__', '?')}" if impl else None,
                     "kernel": bool(impl) and not mod.startswith("transformers.")}
    return res


def functional_check() -> dict:
    import torch
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    out = {}
    x = torch.randn(2, 64, 32, device=dev, dtype=dt)          # [B, D, L] for the conv kernel
    w = torch.randn(64, 4, device=dev, dtype=dt)
    y = M.causal_conv1d_fn(x, w, bias=None, activation="silu")
    out["causal_conv1d"] = {"shape": list(y.shape), "finite": bool(torch.isfinite(y).all())}
    b, l, hk, hd = 1, 64, 2, 64
    q = torch.randn(b, l, hk, hd, device=dev, dtype=dt)
    k = torch.randn(b, l, hk, hd, device=dev, dtype=dt)
    v = torch.randn(b, l, 4, 64, device=dev, dtype=dt)
    g = -torch.rand(b, l, 4, device=dev, dtype=dt)
    beta = torch.rand(b, l, 4, device=dev, dtype=dt)
    o, _ = M.torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    out["chunk_gated_delta_rule"] = {"shape": list(o.shape), "finite": bool(torch.isfinite(o).all())}
    out["device"] = dev
    out["dtype"] = str(dt)
    return out


def training_smoke(model_dir: str, steps: int, seq_len: int) -> dict:
    """4-bit QLoRA on the adapter targets, a synthetic batch, `steps` optimizer steps."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_dir)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, quantization_config=bnb,
                                                        device_map={"": 0}, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                                      "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.train()
    text = "The robot looks at the table, moves its left arm to the cup and places it in the tray."
    ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(0)
    ids = ids.repeat(2, 1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    losses, t0 = [], time.time()
    for _ in range(steps):
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    dt = time.time() - t0
    return {
        "steps": steps,
        "seq_len": int(ids.shape[1]),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "all_finite": all(l == l and abs(l) != float("inf") for l in losses),
        "tokens_per_s": round(steps * ids.numel() / dt, 1),
        "seconds": round(dt, 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="reports/g2_fastpath.json")
    ap.add_argument("--steps", type=int, default=0, help="0 = binding check only; 100 = the gate's smoke")
    ap.add_argument("--seq-len", type=int, default=256)
    a = ap.parse_args(argv)

    import torch
    vers = kernel_versions()
    bind = binding()
    fast = all(v.get("kernel") for v in bind.values())
    func = functional_check() if fast else {}
    report = {
        "gate": "G-2",
        "model_dir": str(Path(a.model).resolve()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "kernels": vers,
        "binding": bind,
        "fast_path_bound": fast,
        "functional": func,
        "result": "PASS" if fast and all(v["finite"] for k, v in func.items() if isinstance(v, dict)) else "FAIL",
    }
    if a.steps:
        smoke = training_smoke(a.model, a.steps, a.seq_len)
        report["training_smoke"] = smoke
        if not smoke["all_finite"]:
            report["result"] = "FAIL"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"G-2 fast path: {report['device']} cc={report['capability']}")
    for k, v in vers.items():
        print(f"  {k:22s} {v}")
    for k, v in bind.items():
        print(f"  {k:26s} -> {v.get('resolved')}  kernel={v.get('kernel')}")
    for k, v in func.items():
        if isinstance(v, dict):
            print(f"  {k:26s} shape={v.get('shape')} finite={v.get('finite')}")
    if a.steps:
        s = report["training_smoke"]
        print(f"  smoke: steps={s['steps']} loss {s['loss_first']:.4f} -> {s['loss_last']:.4f} "
              f"tokens/s={s['tokens_per_s']} peak={s['peak_vram_gb']} GB")
    print(f"  -> {report['result']}  ({out})")
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
