"""End-to-end checks against the real stack (docker Postgres + downloaded models).

Skipped unless RUN_INTEGRATION=1 (see `make test-integration`); needs `make up && make ingest`.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 (make test-integration)"
    ),
]


@pytest.fixture(scope="module")
def real_engine():
    from ragkit.engine import build_engine

    return build_engine(with_llm=False)


@pytest.mark.parametrize("retriever", ["vector", "bm25", "hybrid", "hybrid_rerank"])
def test_cors_question_finds_cors_page(real_engine, retriever):
    res = real_engine.search("How do I allow my frontend origin with CORSMiddleware?", retriever, 5)
    assert any(c.doc_path == "tutorial/cors.md" for c in res.chunks)


def test_api_retrieval_only_against_real_index(real_engine):
    from fastapi.testclient import TestClient

    from ragkit.api import create_app

    with TestClient(create_app(engine=real_engine, warmup=False)) as c:
        body = c.post("/api/chat", json={"question": "What is the default max_age for CORS?"}).json()
    assert body["sources"] and body["max_relevance"] > 0
