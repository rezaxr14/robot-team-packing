"""Write a LLaMA-Factory dataset_info.json and LoRA SFT yaml for the Eye data rendered by 06_render_eye_data.py.
  python ops/make_lf_config.py --data data/eye --model $MODELS/Qwen2.5-VL-3B-Instruct --template qwen2_vl --out runs/eye_sft
The default template matches the upstream VIKI-R configs' family (Qwen2.5-VL). For Qwen3.5 find the right template name
in your LLaMA-Factory version first; if there is none, use the Qwen2.5-VL fallback for the Eye."""
import argparse
import json
from pathlib import Path

def dataset_entry(file_name: str) -> dict:
    return {"file_name": file_name, "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}

YAML = """### model
model_name_or_path: {model}
image_max_pixels: 262144
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {rank}
lora_target: all

### dataset
dataset: {name}
dataset_dir: {data}
template: {template}
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: {out}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: {bs}
gradient_accumulation_steps: {ga}
learning_rate: 1.0e-4
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
"""


def build(data: str, model: str, template: str, out: str, rank: int = 32, bs: int = 1, ga: int = 8, epochs: float = 2.0,
          file_name: str = "eye_sft.jsonl", name: str = "eye_sft"):
    d = Path(data)
    info_path = d / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info[name] = dataset_entry(file_name)                       # merge: several Eye datasets can share one dataset_dir
    info_path.write_text(json.dumps(info, indent=2))
    y = YAML.format(model=model, data=str(d), template=template, out=out, rank=rank, bs=bs, ga=ga, epochs=epochs, name=name)
    Path(out).mkdir(parents=True, exist_ok=True)
    cfg = Path(out) / "eye_sft.yaml"
    cfg.write_text(y)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default="qwen2_vl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--file", default="eye_sft.jsonl", help="jsonl inside --data (e.g. rft_credit.jsonl)")
    ap.add_argument("--name", default="eye_sft", help="dataset name registered in dataset_info.json")
    a = ap.parse_args()
    print(build(a.data, a.model, a.template, a.out, epochs=a.epochs, file_name=a.file, name=a.name))
