"""Hybrid retrieval: BM25 (already in ``rtp.harness``) fused with a Qdrant vector index.

``Harness.retrieve(query, k, embed_fn=...)`` already fuses a BM25 ranking with an embedding ranking
by reciprocal-rank fusion; this module makes the second ranking a **real persistent vector index**
instead of a per-query brute-force pass, which is what the plan's "vector DB alongside BM25" needs.

Qdrant runs in embedded/local mode by default (``path=...``), so there is no server to install on a
laptop. A real server URL can be supplied instead. The client is injected in tests, so no Qdrant
import happens until someone actually uses the class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashedEmbedder:
    """A deterministic stand-in for a real embedding model (tests, smoke, offline notebooks).

    It is *not* semantic: it is only here to prove the plumbing and the rank fusion. Use a real
    embedding model for any reported retrieval number.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import hashlib
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = [((digest[i % len(digest)] / 255.0) * 2 - 1) for i in range(self.dim)]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


def rrf(*rankings: Iterable[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion of several ordered ``(id, score)`` lists; stable and deterministic."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    """BM25 + Qdrant over a fixed document set.

    ``docs`` are ``(id, text)`` pairs — ``Harness.docs`` values have ``.id`` and ``.text``. The
    Qdrant point payload stores the id so a search returns the same ids BM25 does.
    """

    def __init__(self, docs: Sequence[tuple[str, str]], embedder: Embedder, *,
                 path: Path | str | None = None, url: str | None = None,
                 collection: str = "rtp_harness", client=None, log: Callable[[str], None] = print):
        self.docs = list(docs)
        self.embedder = embedder
        self.collection = collection
        self.say = log or (lambda _m: None)
        self.client = client or self._local_client(path, url)
        self._bm25 = self._make_bm25()

    @staticmethod
    def _local_client(path, url):
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:                                  # noqa: BLE001
            raise RuntimeError("qdrant-client is not installed; run ops/install.sh --profile ...") from exc
        if url:
            return QdrantClient(url=url)
        return QdrantClient(path=str(path or Path.cwd() / "qdrant_data"))

    def _make_bm25(self):
        from rtp.harness import BM25
        return BM25([text for _id, text in self.docs], [doc_id for doc_id, _text in self.docs])

    # ---- index -------------------------------------------------------------------------------
    def upsert(self) -> int:
        """(Re)create the collection and index every document. Returns the point count."""
        vectors = self.embedder.embed([text for _id, text in self.docs])
        if len(vectors) != len(self.docs):
            raise ValueError("embedder returned the wrong number of vectors")
        dim = len(vectors[0])
        if any(len(v) != dim for v in vectors):
            raise ValueError("all embeddings must have the same dimension")
        from qdrant_client.models import Distance, PointStruct, VectorParams

        config = VectorParams(size=dim, distance=Distance.COSINE)
        if hasattr(self.client, "recreate_collection"):        # qdrant-client < 1.10
            self.client.recreate_collection(collection_name=self.collection, vectors_config=config)
        else:
            if self.indexed:
                self.client.delete_collection(collection_name=self.collection)
            self.client.create_collection(collection_name=self.collection, vectors_config=config)
        points = [PointStruct(id=i, vector=vec, payload={"doc_id": doc_id})
                  for i, (vec, (doc_id, _text)) in enumerate(zip(vectors, self.docs))]
        self.client.upsert(collection_name=self.collection, points=points)
        self.say(f"indexed {len(points)} documents in Qdrant collection {self.collection!r} (dim={dim})")
        return len(points)

    @property
    def indexed(self) -> bool:
        names = getattr(self.client, "get_collections", None)
        if names is None:
            return False
        try:
            return any(c.name == self.collection for c in names().collections)
        except Exception:                                        # noqa: BLE001
            return False

    # ---- query -------------------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 6) -> list[tuple[str, float]]:
        """Top ``k`` document ids with their fused RRF score (BM25 rank + vector rank)."""
        bm25 = [(doc_id, score) for doc_id, score in self._bm25.rank(query)[:k * 3]]
        if not self.indexed:
            self.upsert()
        vector = self.embedder.embed([query])[0]
        if hasattr(self.client, "query_points"):              # qdrant-client >= 1.10
            hits = self.client.query_points(collection_name=self.collection, query=vector,
                                            limit=k * 3, with_payload=True).points
        else:
            hits = self.client.search(collection_name=self.collection, query_vector=vector,
                                      limit=k * 3, with_payload=True)
        vect = []
        for hit in hits:
            doc_id = (hit.payload or {}).get("doc_id")
            if doc_id is not None:
                vect.append((doc_id, float(hit.score)))
        fused = rrf(bm25, vect)
        return fused[:k]
