#!/usr/bin/env python3
"""Execute a notebook headlessly and print what it produced (a smoke test for a Kaggle session).

    python ops/nbrun.py notebooks/00_setup_cache.ipynb [--timeout 900] [--out /tmp/x.ipynb]

``jupyter nbconvert`` is not installed in this venv (only library stubs), and a readable transcript
is more useful anyway: every stream line is echoed, and the first failing cell is named and turned
into exit code 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def dump(nb) -> tuple | None:
    """Echo stream/result output; return ``(cell_id, error)`` for the first failing cell."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                sys.stdout.write("".join(out.get("text", "")))
            elif kind == "execute_result":
                text = out.get("data", {}).get("text/plain")
                if text:
                    sys.stdout.write("".join(text) + "\n")
            elif kind == "error":
                print(f"\n!! error in {cell.get('id')}: {out.get('ename')}: {out.get('evalue')}")
                return (cell.get("id"), RuntimeError(f"{out.get('ename')}: {out.get('evalue')}"))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    ap.add_argument("--out", default="", help="optional path to write the executed notebook")
    a = ap.parse_args(argv)

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    path = Path(a.notebook).resolve()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=a.timeout, kernel_name="python3",
                            allow_errors=True,
                            resources={"metadata": {"path": str(path.parent)}})
    failed = None
    try:
        executed = client.execute()
    except CellExecutionError as exc:                          # pragma: no cover - hard crash
        print(f"\n!! kernel failed: {exc}")
        return 1
    failed = dump(executed)
    if a.out:
        nbformat.write(executed, a.out)
        print(f"\nwrote {a.out}")
    print("\nOK" if failed is None else f"\nFAILED at {failed[0]}")
    return 0 if failed is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
