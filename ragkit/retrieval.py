"""Retrievers: vector (pgvector cosine), BM25, hybrid (RRF) and hybrid + cross-encoder rerank."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .store import ChunkStore


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int  # 1-based
    detail: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    ``score(d) = sum_i 1 / (k + rank_i(d))`` with 1-based ranks; documents missing
    from a ranking contribute nothing for it. Ties are broken by the best rank
    the document reached in any list, then by id, so results are deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r)
            best_rank[doc] = min(best_rank.get(doc, r), r)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


# --------------------------------------------------------------------------- #
# BM25 (Okapi BM25, Lucene variant, via bm25s)
# --------------------------------------------------------------------------- #


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75, Lucene IDF) over the same chunk texts the vector index uses.

    Tokenisation: lowercase, ``\\w\\w+`` tokens (so ``response_model`` stays one
    token), English stopwords removed, Snowball English stemming.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self._stemmer = Stemmer.Stemmer("english")
        self.ids = [c.chunk_id for c in chunks]
        tokens = bm25s.tokenize(
            [c.search_text for c in chunks], stopwords="en", stemmer=self._stemmer, show_progress=False
        )
        self._vocab = tokens.vocab
        self._model = bm25s.BM25(k1=k1, b=b, method="lucene")
        self._model.index(tokens, show_progress=False)
        self._lock = threading.Lock()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self._bm25s.tokenize([query], stopwords="en", stemmer=self._stemmer, return_ids=False, show_progress=False)[
            0
        ]
        q = [t for t in q if t in self._vocab]
        if not q:
            return []
        k = min(k, len(self.ids))
        with self._lock:
            docs, scores = self._model.retrieve([q], k=k, show_progress=False, n_threads=1)
        out = [(self.ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
        return out


# --------------------------------------------------------------------------- #
# Cross-encoder reranker
# --------------------------------------------------------------------------- #


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512):
        from .embeddings import resolve_device

        self.name = model_name
        self.device = resolve_device(device)
        self.max_length = max_length
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.name, device=self.device, max_length=self.max_length)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict([(query, p) for p in passages], batch_size=32, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Retrievers
# --------------------------------------------------------------------------- #


class VectorRetriever:
    name = "vector"

    def __init__(self, store: ChunkStore, embedder):
        self.store, self.embedder = store, embedder

    def search(self, query: str, k: int) -> list[Hit]:
        qvec = self.embedder.embed_query(query)
        return [
            Hit(cid, sim, r, {"cosine": sim}) for r, (cid, sim) in enumerate(self.store.vector_search(qvec, k), start=1)
        ]


class BM25Retriever:
    name = "bm25"

    def __init__(self, index: BM25Index):
        self.index = index

    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit(cid, s, r, {"bm25": s}) for r, (cid, s) in enumerate(self.index.search(query, k), start=1)]


class HybridRetriever:
    """BM25 + vector, fused with Reciprocal Rank Fusion over the top ``pool`` of each."""

    name = "hybrid"

    def __init__(self, vector: VectorRetriever, bm25: BM25Retriever, rrf_k: int = 60, pool: int = 50):
        self.vector, self.bm25, self.rrf_k, self.pool = vector, bm25, rrf_k, pool

    def search(self, query: str, k: int) -> list[Hit]:
        v = self.vector.search(query, self.pool)
        b = self.bm25.search(query, self.pool)
        v_rank = {h.chunk_id: h.rank for h in v}
        b_rank = {h.chunk_id: h.rank for h in b}
        fused = rrf_fuse([[h.chunk_id for h in v], [h.chunk_id for h in b]], k=self.rrf_k)[:k]
        hits = []
        for r, (cid, s) in enumerate(fused, start=1):
            detail = {"rrf": s}
            if cid in v_rank:
                detail["vector_rank"] = v_rank[cid]
            if cid in b_rank:
                detail["bm25_rank"] = b_rank[cid]
            hits.append(Hit(cid, s, r, detail))
        return hits


class RerankRetriever:
    """Hybrid candidates (top ``candidates``) re-scored by a cross-encoder."""

    name = "hybrid_rerank"

    def __init__(
        self, hybrid: HybridRetriever, reranker: CrossEncoderReranker, store: ChunkStore, candidates: int = 30
    ):
        self.hybrid, self.reranker, self.store, self.candidates = hybrid, reranker, store, candidates

    def search(self, query: str, k: int) -> list[Hit]:
        cands = self.hybrid.search(query, max(k, self.candidates))
        texts = [self.store.get(h.chunk_id).search_text for h in cands]
        scores = self.reranker.score(query, texts)
        order = sorted(range(len(cands)), key=lambda i: (-float(scores[i]), cands[i].rank))[:k]
        return [
            Hit(cands[i].chunk_id, float(scores[i]), r, {"rerank": float(scores[i]), "hybrid_rank": cands[i].rank})
            for r, i in enumerate(order, start=1)
        ]


def ids(hits: Iterable[Hit]) -> list[str]:
    return [h.chunk_id for h in hits]
