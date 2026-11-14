"""Build a tiny random causal LM + tokenizer for CPU smoke tests of the training scripts (no downloads).
  python ops/make_tiny_model.py --data data/sym/brain_sft.jsonl --out runs/tiny
The tokenizer is a BPE trained on the dataset text; the model is a 2-layer Qwen2 config. Output is garbage text by design:
the point is to prove the script, the dataset format and the reward functions run against the installed TRL/PEFT versions."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, AutoModelForCausalLM


def build(data: str, out: str, vocab: int = 2000, hidden: int = 64):
    texts = []
    for line in open(data):
        r = json.loads(line)
        texts += [r.get("prompt", ""), r.get("completion", "")]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<pad>", "<unk>", "<eos>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")
    fast.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      pad_token_id=fast.pad_token_id, eos_token_id=fast.eos_token_id, bos_token_id=fast.eos_token_id)
    model = AutoModelForCausalLM.from_config(cfg)
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    fast.save_pretrained(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(build(a.data, a.out))
