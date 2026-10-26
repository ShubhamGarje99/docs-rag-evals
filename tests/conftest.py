from __future__ import annotations

from pathlib import Path

import pytest

from ragkit.chunking import chunk_corpus
from ragkit.config import Settings
from ragkit.embeddings import HashingEmbedder
from ragkit.engine import RAGEngine
from ragkit.store import MemoryStore

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture
def settings() -> Settings:
    # Explicit values so a developer's .env (API key, models...) never changes test behaviour.
    return Settings(
        anthropic_api_key="",
        corpus_dir=FIXTURE_CORPUS,
        claude_model="claude-sonnet-5-5",
        judge_model="claude-haiku-5-5",
        claude_effort="low",
        claude_fallbacks=True,
        default_retriever="hybrid_rerank",
        top_k=3,
        min_relevance=0.05,
    )


@pytest.fixture
def fixture_chunks(settings):
    return chunk_corpus(FIXTURE_CORPUS, max_tokens=120, overlap_tokens=20)


@pytest.fixture
def engine(settings, fixture_chunks) -> RAGEngine:
    """Tiny in-memory engine: fixture corpus + hashing embedder, no reranker, no LLM."""
    store = MemoryStore.from_chunks(fixture_chunks, HashingEmbedder())
    return RAGEngine(store, HashingEmbedder(), settings, reranker=None, answerer=None)
