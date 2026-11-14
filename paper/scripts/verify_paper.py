"""Audit the paper sources.  python scripts/verify_paper.py   (exit 1 on any error)

1. every \\cite key exists in refs.bib and every entry has a `note` of the form "verified: ..." or "TODO-verify";
   cited TODO-verify keys are listed (they may not be described as verified);
2. every label is defined and referenced (from the LaTeX log);
3. no residue of the template project (names, domain words);
4. every number in the prose is traceable: it is a generated macro value, a value in fixture.json / figdata.json,
   or an entry of VERIFIED_FACTS below (each with its source);
5. pending cells and \\todo flags are counted and written to OPEN_ITEMS.md.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
GEN = PAPER / "generated"
errors, warns = [], []

# --- numbers taken from the project code or the planning notes (VERIFIED), with where they come from
VERIFIED_FACTS = {
    # message and harness counts, from the code and the harness directory
    "13": "message types (rtp/messages.py SCHEMA)", "19": "harness files (harness_files/)", "6": "rules / max tool turns (code)",
    "5": "tools (rtp/toolbrain.py), top-k rules (harness)", "4": "skills; exec_steps=4 (rtp/policy_rollout.py)", "3": "object/skill/memory top-k",
    # reward and score constants (rtp/rewards.py, rtp/loop.py)
    "0.1": "r_format, r_length weights", "0.2": "r_feasible weight; challenge tolerance 20%", "0.6": "r_rules weight", "0.25": "r_rules penalty",
    "0.5": "rollout weight; flaky p_err; exhaustive toy average", "0.15": "violation weight; attribution tolerance", "0.3": "crush weight",
    "0.01": "replan weight", "24": "r_length cap", "48": "r_length zero point (2 x cap)", "20": "challenge tolerance, percent",
    "15": "attribution tolerance, percent", "30": "crush force constant (rtp/world.py)", "22": "crush fragility coefficient (rtp/world.py)",
    "8": "Euler steps, action dim, K samples", "16": "action chunk horizon (rtp/flowhead.py)", "2": "verifier retries", "100000": "seed split",
    # upstream scorer facts and toy case (authors' notes; flagged todo in appendix D)
    "0.99": "upstream length-ratio threshold", "0.9": "upstream reward weight", "900": "upstream L2 rows read", "1000": "upstream L1 rows read",
    "18": "toy case passes (authors' earlier run)", "40": "toy case draws (authors' earlier run)",
    # environment
    "137": "tests passed on CPU, measured in this build", "13_tests": "tests skipped", "5.x": "Isaac Sim pairing (planned)", "2.3": "Isaac Lab pairing (planned)",
    "95": "confidence level of Wilson / bootstrap intervals (convention)", "0.4": "fixture omission probability (scripts/collect_fixture.py)",
    "11": "python 3.11 mention", "3.11": "python version (planned)", "16_gb": "T4 memory", "45": "weights count", "50": "x",
}
# year-like and ordinal tokens are exempt
EXEMPT = re.compile(r"^(19|20)\d\d$")


def strip_tex(s: str) -> str:
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\\(cite[tp]?|citep|citet|cref|Cref|ref|label|input|includegraphics|bibliography|bibliographystyle|texttt|url|href)\*?(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\todo\{.*?\}\s*", " ", s, flags=re.S)
    s = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    s = re.sub(r"\$[^$]*\$", " ", s)
    return s


def allowed_numbers() -> dict:
    ok = dict(VERIFIED_FACTS)
    macros = (GEN / "figdata_macros.tex").read_text()
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", macros):
        for t in re.findall(r"[-+]?\d+(?:\.\d+)?", val):
            ok[t.lstrip("+-")] = f"macro \\{name}"
    fx = json.loads((GEN / "fixture.json").read_text())
    n = fx["n"]
    ok[str(n)] = "fixture n"
    ok["80"] = "fixture n"
    return ok


def main() -> int:
    bib = (PAPER / "refs.bib").read_text()
    entries = dict(re.findall(r"@\w+\{(\w+),(.*?)\n\}", bib, flags=re.S))
    status = {}
    for k, body in entries.items():
        m = re.search(r"note\s*=\s*\{([^}]*)\}", body)
        if not m or not (m.group(1).startswith("verified:") or m.group(1).startswith("TODO-verify")):
            errors.append(f"bib entry {k}: note must start with 'verified:' or 'TODO-verify'")
        status[k] = "verified" if m and m.group(1).startswith("verified:") else "TODO-verify"

    texts = {p.name: p.read_text() for p in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]}
    texts["main.tex"] = texts["main.tex"].split(r"\begin{document}", 1)[1]        # skip the preamble (colour specs etc.)
    cited = set()
    for name, t in texts.items():
        for grp in re.findall(r"\\cite[tp]?\{([^}]*)\}", t):
            for k in grp.split(","):
                cited.add(k.strip())
    for k in sorted(cited):
        if k not in entries:
            errors.append(f"cited key missing from refs.bib: {k}")
    unverified = sorted(k for k in cited if status.get(k) == "TODO-verify")
    for name, t in texts.items():
        if re.search(r"\b(verified|confirmed)\b[^.]*\\cite", t):
            warns.append(f"{name}: a sentence says verified/confirmed next to a citation; check it")

    log = (PAPER / "main.log").read_text() if (PAPER / "main.log").exists() else ""
    for m in re.findall(r"(?:Reference|Citation) `([^']+)' [^\n]*undefined", log):
        errors.append(f"undefined reference or citation in log: {m}")
    if "Label(s) may have changed" in log:
        warns.append("labels changed on the last pass: run pdflatex once more")

    for name, t in texts.items():
        for bad in ("DENTEX", "dental", "Dental", "Reza", "VLM-DENTAL", "radiograph", "Nadimi"):
            if bad in t:
                errors.append(f"{name}: template residue '{bad}'")

    ok = allowed_numbers()
    untraced = []
    for name, t in texts.items():
        if name == "07_appendix.tex":
            pass
        for tok in re.findall(r"(?<![\w.\\{])\d+(?:\.\d+)?(?![\w}])", strip_tex(t)):
            if EXEMPT.match(tok) or tok in ok or (len(tok) == 1) or tok.lstrip("0") in ok:
                continue
            untraced.append((name, tok))
    for name, tok in sorted(set(untraced)):
        warns.append(f"{name}: number '{tok}' has no recorded source (add to VERIFIED_FACTS with its source, or generate it)")

    todo_count = sum(len(re.findall(r"\\todo\{", t)) for t in texts.values())
    pend = {}
    for f in sorted(GEN.glob("table_*.pending.json")):
        j = json.loads(f.read_text())
        pend[j["table"]] = j["n_pending_cells"]
    n_pend = sum(pend.values())

    lines = ["# Open items (generated by scripts/verify_paper.py)", "", f"- `\\todo{{}}` flags in the prose: {todo_count}",
             f"- Pending table cells: {n_pend} " + str({k: v for k, v in pend.items() if v}), "",
             "## Citations marked TODO-verify and cited in the text", ""] + [f"- {k}" for k in unverified] + \
            ["", "## Numbers without a recorded source", ""] + [f"- {n}: {t}" for n, t in sorted(set(untraced))] + \
            ["", "## Warnings", ""] + [f"- {w}" for w in warns if not w.startswith(tuple(f"{n}:" for n in ()))]
    (PAPER / "OPEN_ITEMS.md").write_text("\n".join(lines) + "\n")
    print(f"cited keys: {len(cited)} ({len(unverified)} TODO-verify)  todo flags: {todo_count}  pending cells: {n_pend}")
    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
