#!/usr/bin/env python3
"""Build ``.ipynb`` files from Python, so a notebook is reviewable and diffs cleanly.

Hand-editing notebook JSON is how cells end up with duplicate ids, broken syntax or missing
``execution_count`` keys. Each notebook here is written by a small script that calls this builder:

    from ops.nbbuild import Notebook
    nb = Notebook("10_sft_brain")
    nb.md("# SFT ...")
    nb.code("ACTIVE = 'local_4050'")
    nb.save()                      # -> notebooks/10_sft_brain.ipynb

Cell ids are derived from the notebook name, the position and a hash of the source, so they are
stable under edits and unique within the file (what ``nbformat`` requires).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def _clean(text: str) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


class Notebook:
    def __init__(self, name: str, *, path: Path | None = None):
        self.name = name
        self.path = Path(path) if path else NB_DIR / f"{name}.ipynb"
        self.cells: list[dict] = []

    def _add(self, kind: str, text: str, **extra) -> "Notebook":
        src = _clean(text)
        digest = hashlib.sha1("".join(src).encode()).hexdigest()[:8]
        cell = {"cell_type": kind, "id": f"{self.name.split('_')[0]}-{len(self.cells):02d}-{digest}",
                "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cell.update(extra)
        self.cells.append(cell)
        return self

    def md(self, text: str) -> "Notebook":
        return self._add("markdown", text)

    def code(self, text: str) -> "Notebook":
        return self._add("code", text)

    def save(self) -> Path:
        nb = {"cells": self.cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(nb, indent=1) + "\n")
        return self.path


INSTALL_CELL = '''# Install anything missing (Kaggle/local). Nothing here is imported unconditionally: a notebook
# must fail with a readable message, not at `import torch`.
import importlib.util, subprocess, sys

REQUIRED = %r
missing = [p for p in REQUIRED if importlib.util.find_spec(p.replace("-", "_").split("[")[0]) is None]
if missing:
    print("installing:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("dependencies present")
'''


def install_cell(packages: list[str]) -> str:
    return INSTALL_CELL % (packages,)
