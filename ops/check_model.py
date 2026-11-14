"""Model-support gate: does this transformers install load the model, run a forward pass and generate?
  python ops/check_model.py /path/or/hf-id          (run in the train env)
Writes reports/model_check_<name>.json. If this fails for Qwen3.5-4B, switch BASE_MODEL to the Qwen2.5-VL fallback."""
import json
import sys
import time
from pathlib import Path

import torch
import transformers

model_id = sys.argv[1]
name = Path(model_id).name or model_id.replace("/", "_")
rep = {"model": model_id, "transformers": transformers.__version__, "torch": torch.__version__, "ok": False}
t0 = time.time()
try:
    cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    rep["architecture"] = (getattr(cfg, "architectures", None) or [None])[0]
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
            rep["loaded_with"] = cls_name
            break
        except Exception as e:                                           # noqa: BLE001
            rep.setdefault("load_errors", {})[cls_name] = f"{type(e).__name__}: {str(e)[:200]}"
    if model is None:
        raise RuntimeError("no Auto class could load the model")
    msgs = [{"role": "user", "content": "Reply with exactly: {\"ok\": true}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=24, do_sample=False)
    rep["sample"] = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    rep["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    rep["ok"] = True
except Exception as e:                                                   # noqa: BLE001
    rep["error"] = f"{type(e).__name__}: {str(e)[:500]}"
rep["seconds"] = round(time.time() - t0, 1)
Path("reports").mkdir(exist_ok=True)
Path(f"reports/model_check_{name}.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
sys.exit(0 if rep["ok"] else 1)
