"""Harness: skills / rules / object notes / memory as .md files with retrieval and write-back.

File format:
    ---
    id: heavy_bottom
    type: rule            # rule | skill | object | agent | task | memory
    keywords: heavy bottom stack order
    ---
    body text ...
    <!-- stats -->  (monitor-managed block, optional)
    <!-- /stats -->
    ## Notes        (monitor appends failure notes here)
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_TOK = re.compile(r"[a-z]+|[0-9]+")


def tokenize(s: str) -> list:
    return _TOK.findall(s.lower())


@dataclass
class Doc:
    id: str
    type: str
    keywords: str
    body: str
    path: Path | None = None

    @property
    def text(self) -> str:
        return f"{self.id} {self.type} {self.keywords} {self.body}"


def parse_md(path: Path) -> Doc:
    raw = Path(path).read_text()
    m = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.DOTALL)
    if not m:
        raise ValueError(f"{path}: missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.split("#")[0].strip()
    for k in ("id", "type"):
        if k not in meta:
            raise ValueError(f"{path}: front matter missing '{k}'")
    return Doc(meta["id"], meta["type"], meta.get("keywords", ""), m.group(2).strip(), Path(path))


class Harness:
    def __init__(self, root):
        self.root = Path(root)
        self.docs: dict = {}
        self.reload()

    def reload(self):
        self.docs = {}
        for p in sorted(self.root.rglob("*.md")):
            d = parse_md(p)
            if d.id in self.docs:
                raise ValueError(f"duplicate id {d.id}")
            self.docs[d.id] = d
        self.bm25 = BM25([d.text for d in self.docs.values()], list(self.docs))

    def get(self, id_: str) -> Doc:
        return self.docs[id_]

    def retrieve(self, query: str, k: int = 6, types=None, embed_fn=None) -> list:
        ranked = self.bm25.rank(query)
        if embed_fn is not None:
            ranked = _rrf(ranked, _embed_rank(query, self.docs, embed_fn))
        out = []
        for id_, _s in ranked:
            d = self.docs[id_]
            if types and d.type not in types:
                continue
            out.append(d)
            if len(out) == k:
                break
        return out

    # ---- write-back (monitor) ----
    def add_note(self, id_: str, note: str):
        d = self.docs[id_]
        txt = d.path.read_text()
        if "## Notes" not in txt:
            txt = txt.rstrip() + "\n\n## Notes\n"
        if note not in txt:
            txt = txt.rstrip() + f"\n- {note}\n"
        d.path.write_text(txt)

    def set_stats(self, id_: str, stats: dict):
        d = self.docs[id_]
        txt = d.path.read_text()
        block = "<!-- stats -->\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(stats.items())) + "\n<!-- /stats -->"
        if "<!-- stats -->" in txt:
            txt = re.sub(r"<!-- stats -->.*?<!-- /stats -->", block, txt, flags=re.DOTALL)
        else:
            txt = txt.rstrip() + "\n\n" + block + "\n"
        d.path.write_text(txt)


class BM25:
    def __init__(self, texts, ids, k1: float = 1.5, b: float = 0.75):
        self.ids, self.k1, self.b = ids, k1, b
        self.toks = [tokenize(t) for t in texts]
        self.N = len(self.toks)
        self.avg = sum(len(t) for t in self.toks) / max(1, self.N)
        df = {}
        for t in self.toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def rank(self, query: str) -> list:
        q = tokenize(query)
        scores = []
        for i, t in enumerate(self.toks):
            tf = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q:
                if w in tf:
                    s += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / self.avg))
            scores.append((self.ids[i], s))
        return sorted(scores, key=lambda x: (-x[1], x[0]))


def _embed_rank(query, docs, embed_fn):
    qv = embed_fn(query)
    sims = []
    for id_, d in docs.items():
        v = embed_fn(d.text)
        dot = sum(a * b for a, b in zip(qv, v))
        na = math.sqrt(sum(a * a for a in qv)) or 1.0
        nb = math.sqrt(sum(b * b for b in v)) or 1.0
        sims.append((id_, dot / (na * nb)))
    return sorted(sims, key=lambda x: (-x[1], x[0]))


def _rrf(r1, r2, k: int = 60):
    sc = {}
    for ranking in (r1, r2):
        for rank, (id_, _s) in enumerate(ranking):
            sc[id_] = sc.get(id_, 0.0) + 1.0 / (k + rank + 1)
    return sorted(sc.items(), key=lambda x: (-x[1], x[0]))


class Monitor:
    """Aggregates skill outcomes and writes measured stats and failure notes back into the harness."""

    def __init__(self, harness: Harness):
        self.h = harness
        self.stats: dict = {}

    def record(self, skill: str, obj_cls: str, ok: bool, cause: str = ""):
        s = self.stats.setdefault(skill, {"attempts": 0, "success": 0, "causes": {}})
        s["attempts"] += 1
        s["success"] += int(ok)
        if not ok:
            s["causes"][cause] = s["causes"].get(cause, 0) + 1

    def flush(self):
        for skill, s in self.stats.items():
            if skill in self.h.docs:
                rate = s["success"] / max(1, s["attempts"])
                self.h.set_stats(skill, {"attempts": s["attempts"], "success_rate": round(rate, 3),
                                         "top_failure": max(s["causes"], key=s["causes"].get) if s["causes"] else "none"})
