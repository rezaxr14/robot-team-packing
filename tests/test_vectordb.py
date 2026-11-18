"""Hybrid retrieval tests with an injected Qdrant double (no server, no embeddings download).

The point here is the *plumbing*: BM25 and the vector index are both ranked, fused by RRF, and the
fused result is a permutation of the same document ids. The hashed embedder is explicitly a stand-in
and is never used to claim retrieval quality.
"""
from types import SimpleNamespace

import pytest

from rtp.vectordb import HashedEmbedder, HybridRetriever, rrf


class FakeClient:
    def __init__(self):
        self.collections = []
        self.points = []
        self.config = None

    def recreate_collection(self, collection_name, vectors_config):
        self.collections = [SimpleNamespace(name=collection_name)]
        self.config = vectors_config
        self.points = []

    def upsert(self, collection_name, points):
        self.points.extend(points)

    def get_collections(self):
        return SimpleNamespace(collections=self.collections)

    def search(self, collection_name, query_vector, limit, with_payload):
        # reverse the id order so the vector ranking is not identical to BM25's
        return [SimpleNamespace(payload=p.payload, score=1.0 / (i + 1))
                for i, p in enumerate(reversed(self.points))][:limit]


class ConstantEmbedder:
    def __init__(self, dim=4, value=0.5):
        self.dim, self.value = dim, value

    def embed(self, texts):
        return [[self.value] * self.dim for _ in texts]


DOCS = [("rules", "heavy items go at the bottom of the bag"),
        ("skills", "weigh a fragile item in the gripper before placing"),
        ("notes", "the notebook is on the right side of the table")]


def test_rrf_rewards_documents_ranked_high_by_either_list():
    out = rrf([("a", 3), ("b", 2)], [("b", 9), ("c", 1)])
    assert out[0][0] == "b"
    assert {d for d, _ in out} == {"a", "b", "c"}


def test_upsert_indexes_every_document_and_records_the_dimension():
    c = FakeClient()
    r = HybridRetriever(DOCS, ConstantEmbedder(dim=7), client=c, log=lambda _m: None)
    assert r.upsert() == 3
    assert c.config.size == 7
    assert {p.payload["doc_id"] for p in c.points} == {"rules", "skills", "notes"}


def test_retrieve_fuses_bm25_and_vector_rankings():
    c = FakeClient()
    r = HybridRetriever(DOCS, ConstantEmbedder(dim=4), client=c, log=lambda _m: None)
    out = r.retrieve("weigh the fragile notebook", k=3)
    assert len(out) == 3 and len({d for d, _ in out}) == 3
    assert all(d in {x for x, _ in DOCS} for d, _ in out)
    assert all(score > 0 for _d, score in out)


def test_a_broken_embedder_is_rejected():
    class Wrong:
        def embed(self, texts):
            return [[1.0, 0.0]]                     # one vector for three documents

    r = HybridRetriever(DOCS, Wrong(), client=FakeClient(), log=lambda _m: None)
    with pytest.raises(ValueError, match="number of vectors"):
        r.upsert()


def test_hashed_embedder_is_deterministic_and_normalised():
    a = HashedEmbedder(16).embed(["one", "two"])
    b = HashedEmbedder(16).embed(["one", "two"])
    assert a == b and all(abs(sum(v * v for v in vec) - 1.0) < 1e-9 for vec in a)
