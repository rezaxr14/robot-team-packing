"""Find the longest sequence a profile can actually train at, instead of guessing `max_len`.

`rtp/profiles.py` sets `max_len` per hardware profile (8 k on the 6 GB 4050, 16 k on the 4090).
Those numbers were chosen before anything ran, so this script measures them: one model load, then a
forward + backward at each requested length with the chunked LM loss (D-37), recording peak VRAM and
stopping at the first out-of-memory instead of crashing.

    python ops/seq_ceiling.py --model ~/models/Qwen3.5-4B --lengths 512,1024,2048,4096,8192
    python ops/seq_ceiling.py --model ... --batch 2 --out reports/seq_ceiling_local_4050.json

The answer is a property of the profile (VRAM, dtype, tuning), so it is recorded per profile and
`max_len` is corrected from it rather than from a table someone typed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 6 GB cards fragment badly at these sizes; must be set before torch touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # `rtp` imports when run as a script

SAMPLE = ("The left arm reaches for the red cup on the table and places it in the tray. "
          "The right arm picks up the blue bottle and sets it upright in the front compartment. "
          "Look at the counter before moving: the bowl is behind the kettle. ")


def build(model_dir: str, attn: str = "auto", lora_r: int = 32):
    """The same 4-bit QLoRA stack the gate and SFT use; returns (model, tokenizer)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    from rtp.profiles import TEXT_LORA_TARGETS, VISION_LORA_TARGETS

    tok = AutoTokenizer.from_pretrained(model_dir)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    kwargs = {} if attn == "auto" else {"attn_implementation": attn}
    model = AutoModelForImageTextToText.from_pretrained(model_dir, quantization_config=bnb,
                                                        device_map={"": 0}, dtype=torch.bfloat16, **kwargs)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=list(TEXT_LORA_TARGETS) + list(VISION_LORA_TARGETS)))
    model.train()
    return model, tok


def token_ids(tok, seq_len: int, batch: int):
    """A real-text row repeated to exactly `seq_len`; synthetic ids would change nothing but risk."""
    row = tok(SAMPLE, return_tensors="pt", add_special_tokens=False).input_ids[0]
    row = row.repeat(seq_len // row.numel() + 1)[:seq_len]
    if tok.bos_token_id is not None:
        row[0] = tok.bos_token_id
    return row.unsqueeze(0).repeat(batch, 1)


def measure(model, ids, loss_chunk: int) -> dict:
    """One optimisation step; returns peak allocated VRAM for this length."""
    import torch

    from rtp.loss import chunked_lm_loss

    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss = chunked_lm_loss(model, ids, chunk_tokens=loss_chunk)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return {"loss": float(loss.detach()), "peak_vram_gb": round(peak, 2),
            "seconds": round(time.time() - t0, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local checkpoint directory")
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--loss-chunk", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=32, help="adapter rank: it competes with activations for VRAM")
    ap.add_argument("--attn", default="auto", choices=["auto", "sdpa", "flash_attention_2"])
    ap.add_argument("--gpu-limit", type=float, default=0.0,
                    help="stop before an OOM if peak would exceed this many GB (0 = try anyway)")
    ap.add_argument("--out", default="")
    ap.add_argument("--profile", default="", help="label the report with a profile name")
    a = ap.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("no CUDA: nothing to measure", file=sys.stderr)
        return 2
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"gpu: {torch.cuda.get_device_name(0)} | {total:.1f} GB total | batch={a.batch}")

    model, tok = build(a.model, a.attn, a.lora_r)
    lengths = [int(s) for s in a.lengths.split(",") if s.strip()]
    report = {"profile": a.profile, "gpu": torch.cuda.get_device_name(0),
              "vram_total_gb": round(total, 2), "batch": a.batch, "lora_r": a.lora_r,
              "loss_chunk_tokens": a.loss_chunk, "attn": a.attn,
              "torch": torch.__version__, "lengths": {}}
    longest = 0
    for seq_len in lengths:
        ids = token_ids(tok, seq_len, a.batch).to(0)
        try:
            result = measure(model, ids, a.loss_chunk)
        except torch.OutOfMemoryError as exc:
            # An OOM here is the answer, not a failure: record where it happens and stop growing.
            report["lengths"][str(seq_len)] = {"fits": False, "error": str(exc).splitlines()[0][:200]}
            print(f"  seq={seq_len:<6d} OOM")
            torch.cuda.empty_cache()
            break
        result["fits"] = True
        report["lengths"][str(seq_len)] = result
        longest = seq_len
        print(f"  seq={seq_len:<6d} loss={result['loss']:.4f} peak={result['peak_vram_gb']:.2f} GB "
              f"{result['seconds']:.1f}s")
        del ids
        if a.gpu_limit and result["peak_vram_gb"] >= a.gpu_limit:
            print(f"  stopping: peak reached the {a.gpu_limit} GB limit")
            break
        torch.cuda.empty_cache()

    report["longest_fitting"] = longest
    report["headroom_gb"] = round(total - max(
        (v["peak_vram_gb"] for v in report["lengths"].values() if v.get("fits")), default=0.0), 2)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  -> {out}")
    print(f"  longest fitting length at batch {a.batch}: {longest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
