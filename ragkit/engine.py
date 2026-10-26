"""The RAG engine: retrievers + refusal gate + (optional) Claude answerer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .chunking import Chunk
from .config import NO_KEY_MESSAGE, RETRIEVERS, Settings, get_settings
from .llm import REFUSAL_TEXT, Answer, ClaudeAnswerer
from .retrieval import (
    BM25Index,
    BM25Retriever,
    CrossEncoderReranker,
    Hit,
    HybridRetriever,
    RerankRetriever,
    VectorRetriever,
)
from .store import ChunkStore

RETRIEVER_LABELS = {
    "vector": "Vector (bge cosine)",
    "bm25": "BM25",
    "hybrid": "Hybrid (BM25 + vector, RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}


@dataclass
class SearchResult:
    retriever: str
    hits: list[Hit]
    chunks: list[Chunk]
    latency_ms: float


def source_dict(hit: Hit, chunk: Chunk, relevance: float | None = None) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "chunk_id": chunk.chunk_id,
        "title": chunk.breadcrumb,
        "page_title": chunk.page_title,
        "url": chunk.url,
        "score": round(hit.score, 6),
        "relevance": None if relevance is None else round(relevance, 4),
        "detail": {k: round(v, 6) for k, v in hit.detail.items()},
        "text": chunk.text,
    }


class RAGEngine:
    def __init__(
        self,
        store: ChunkStore,
        embedder,
        settings: Settings | None = None,
        reranker: CrossEncoderReranker | None = None,
        answerer: ClaudeAnswerer | None = None,
    ):
        self.settings = settings or get_settings()
        self.store = store
        self.embedder = embedder
        self.bm25_index = BM25Index(store.chunks())
        vector = VectorRetriever(store, embedder)
        bm25 = BM25Retriever(self.bm25_index)
        hybrid = HybridRetriever(vector, bm25, rrf_k=self.settings.rrf_k)
        self.retrievers: dict[str, Any] = {"vector": vector, "bm25": bm25, "hybrid": hybrid}
        if reranker is not None:
            self.retrievers["hybrid_rerank"] = RerankRetriever(
                hybrid, reranker, store, candidates=self.settings.candidate_k
            )
        self.reranker = reranker
        self.answerer = answerer

    # ------------------------------------------------------------------ #
    @property
    def llm_enabled(self) -> bool:
        return self.answerer is not None

    @property
    def available_retrievers(self) -> list[str]:
        return [r for r in RETRIEVERS if r in self.retrievers]

    def resolve_retriever(self, name: str | None) -> str:
        name = name or self.settings.default_retriever
        if name not in self.retrievers:
            name = "hybrid" if name == "hybrid_rerank" and "hybrid" in self.retrievers else name
        if name not in self.retrievers:
            raise KeyError(f"unknown retriever {name!r}; choose from {self.available_retrievers}")
        return name

    def search(self, query: str, retriever: str | None = None, k: int | None = None) -> SearchResult:
        name = self.resolve_retriever(retriever)
        k = k or self.settings.top_k
        t0 = time.perf_counter()
        hits = self.retrievers[name].search(query, k)
        latency = (time.perf_counter() - t0) * 1000
        return SearchResult(name, hits, [self.store.get(h.chunk_id) for h in hits], latency)

    def relevance(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Cosine similarity between the question and each retrieved chunk (retriever-agnostic)."""
        return self.store.similarities(self.embedder.embed_query(query), chunk_ids)

    # ------------------------------------------------------------------ #
    def chat(self, question: str, retriever: str | None = None, k: int | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        res = self.search(question, retriever, k)
        rel = self.relevance(question, [h.chunk_id for h in res.hits])
        best = max(rel.values(), default=0.0)
        sources = [source_dict(h, c, rel.get(c.chunk_id)) for h, c in zip(res.hits, res.chunks)]
        gate_refuse = best < self.settings.min_relevance
        base = {
            "question": question,
            "retriever": res.retriever,
            "sources": sources,
            "retrieval_latency_ms": round(res.latency_ms, 1),
            "max_relevance": round(best, 4),
            "relevance_threshold": self.settings.min_relevance,
        }

        if not self.llm_enabled:
            return {
                **base,
                "mode": "retrieval_only",
                "banner": NO_KEY_MESSAGE,
                "answer": None,
                "refused": None,
                "would_refuse": gate_refuse,
                "refusal_reason": None,
                "citations": [],
                "usage": None,
                "llm_latency_ms": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }

        if gate_refuse:
            ans = Answer(REFUSAL_TEXT, True, "low_retrieval_score", [], None, 0.0, None)
        else:
            assert self.answerer is not None
            ans = self.answerer.answer(question, res.chunks)
        return {
            **base,
            "mode": "llm",
            "banner": None,
            "answer": ans.answer,
            "refused": ans.refused,
            "would_refuse": gate_refuse,
            "refusal_reason": ans.refusal_reason,
            "citations": [c.__dict__ for c in ans.citations],
            "usage": ans.usage.to_dict() if ans.usage else None,
            "fallback_used": ans.fallback_used,
            "stop_reason": ans.stop_reason,
            "llm_latency_ms": round(ans.latency_ms, 1),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def build_engine(settings: Settings | None = None, with_reranker: bool = True, with_llm: bool = True) -> RAGEngine:
    """Production wiring: pgvector store + sentence-transformers models + Claude (if a key is set)."""
    from .embeddings import SentenceTransformerEmbedder
    from .store import PgStore

    settings = settings or get_settings()
    store = PgStore(settings.database_url, settings.embed_model)
    embedder = SentenceTransformerEmbedder(settings.embed_model, settings.device)
    reranker = CrossEncoderReranker(settings.rerank_model, settings.device) if with_reranker else None
    answerer = ClaudeAnswerer(settings) if (with_llm and settings.llm_enabled) else None
    return RAGEngine(store, embedder, settings, reranker, answerer)
