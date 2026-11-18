#!/usr/bin/env python3
"""Generate the notebook form of the model publisher.

Kaggle script kernels cannot read account secrets here (UserSecretsClient returns HTTP 400 for every
label), so the same publisher is also shipped as a **notebook** kernel, where secrets are available.
Only the wrapper changes: `publish_model.py` itself is unmodified.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ops.nbbuild import Notebook                                      # noqa: E402

DEST = ROOT / "ops" / "kaggle" / "publish_model"
src = (DEST / "publish_model.py").read_text()
# strip the shebang and the module docstring; keep the executable body
def strip_docstring(text: str) -> str:
    """Remove the module docstring only (the body may contain triple-quoted strings)."""
    lines = text.splitlines()
    assert lines[0].startswith("#!"), "expected a shebang"
    i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    assert lines[i].startswith('"""'), "expected a module docstring"
    j = i
    if lines[i].strip() != '"""':
        j += 1
        while j < len(lines) and not lines[j].strip().endswith('"""'):
            j += 1
    return "\n".join(lines[j + 1:])


body = strip_docstring(src)

nb = Notebook("publish_model", path=DEST / "publish_model.ipynb")
nb.md("""# Publish Qwen3.5-4B from Kaggle into our private dataset

This notebook runs **on Kaggle**, never on the laptop: Kaggle downloads the 9.4 GB snapshot to its
own disk and publishes the private dataset `rtp-model-qwen35-4b-851bf6e8`, which our notebooks then
attach read-only at `/kaggle/input`. The laptop never uploads the model.

The only secret needed is `KAGGLE_API_TOKEN`, attached to this notebook in Kaggle's Secrets panel.
""")
nb.code(body)
path = nb.save()
print("wrote", path)
