"""Collect a small, non-sensitive environment report to paste back (no env vars, no tokens, no paths beyond cwd)."""
import importlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.request

PKGS = ["torch", "transformers", "trl", "peft", "vllm", "verl", "ray", "flash_attn", "datasets", "pandas",
        "pyarrow", "numpy", "openai", "mathruler", "pylatexenc", "mani_skill", "isaaclab", "pytest"]


def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    except Exception as e:                                           # noqa: BLE001
        return f"MISSING ({type(e).__name__})"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:                                           # noqa: BLE001
        return f"error: {e}"


def reach(url):
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=8)
        return True
    except Exception as e:                                           # noqa: BLE001
        return f"no ({type(e).__name__})"


rep = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(),
       "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1),
       "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
       "nvcc": sh("nvcc --version | tail -1"), "packages": {p: ver(p) for p in PKGS},
       "reach": {"huggingface.co": reach("https://huggingface.co"), "github.com": reach("https://github.com"),
                 "pypi.org": reach("https://pypi.org")}}
try:
    import torch
    rep["torch_cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda,
                         "devices": [(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1))
                                     for i in range(torch.cuda.device_count())]}
except Exception:                                                    # noqa: BLE001
    pass
print(json.dumps(rep, indent=2))
