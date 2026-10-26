"""API endpoints in retrieval-only mode (no API key, no DB, no model downloads)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from ragkit.api import create_app
from ragkit.engine import RAGEngine
from ragkit.llm import Answer, Citation, Usage


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine=engine, warmup=False)) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["llm_enabled"] is False and body["chunks"] > 0
    assert set(body["retrievers"]) == {"vector", "bm25", "hybrid"}


def test_config_reports_retrieval_only(client):
    cfg = client.get("/api/config").json()
    assert cfg["llm_enabled"] is False
    assert "ANTHROPIC_API_KEY" in cfg["banner"]
    assert cfg["default_retriever"] == "hybrid"  # hybrid_rerank falls back when no reranker is loaded
    assert [r["id"] for r in cfg["retrievers"]] == ["vector", "bm25", "hybrid"]


@pytest.mark.parametrize("retriever", ["vector", "bm25", "hybrid"])
def test_search(client, retriever):
    r = client.get("/api/search", params={"q": "CORSMiddleware allow_origins", "retriever": retriever, "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["retriever"] == retriever and len(body["results"]) <= 2
    top = body["results"][0]
    assert top["chunk_id"].startswith("tutorial/cors.md#")
    assert top["url"].startswith("https://fastapi.tiangolo.com/tutorial/cors/")
    assert {"rank", "title", "score", "relevance", "text", "detail"} <= set(top)


def test_search_rejects_unknown_retriever(client):
    r = client.get("/api/search", params={"q": "x", "retriever": "magic"})
    assert r.status_code == 422


def test_chat_retrieval_only_mode(client):
    r = client.post("/api/chat", json={"question": "How do I run a task after the response?", "retriever": "bm25"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "retrieval_only"
    assert body["answer"] is None and body["citations"] == []
    assert "ANTHROPIC_API_KEY" in body["banner"]
    assert body["sources"][0]["chunk_id"].startswith("tutorial/background-tasks.md#")
    assert isinstance(body["would_refuse"], bool)
    assert body["latency_ms"] >= body["retrieval_latency_ms"] >= 0


def test_chat_validation(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "x", "k": 99}).status_code == 422


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "FastAPI Docs Copilot" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_chat_llm_mode_with_fake_answerer(settings, engine):
    chunk = engine.store.chunks()[0]

    class FakeAnswerer:
        def answer(self, question, chunks):
            return Answer(
                answer="Use BackgroundTasks[1].",
                refused=False,
                refusal_reason=None,
                citations=[Citation(1, chunk.chunk_id, chunk.breadcrumb, chunk.url, "quote", ["quote"])],
                usage=Usage("claude-sonnet-5-5", 500, 50, cost_usd=0.0015),
                latency_ms=5.0,
                stop_reason="end_turn",
            )

    eng = RAGEngine(engine.store, engine.embedder, settings, answerer=FakeAnswerer())
    with TestClient(create_app(engine=eng, warmup=False)) as c:
        body = c.post("/api/chat", json={"question": "background tasks after response"}).json()
    assert body["mode"] == "llm" and body["refused"] is False
    assert body["citations"][0]["chunk_id"] == chunk.chunk_id
    assert body["usage"]["cost_usd"] == pytest.approx(0.0015)


def test_degraded_when_index_missing(monkeypatch):
    import ragkit.engine as eng_mod

    def boom(*a, **k):
        raise RuntimeError("no db")

    monkeypatch.setattr(eng_mod, "build_engine", boom)
    with TestClient(create_app(warmup=False)) as c:
        h = c.get("/healthz").json()
        assert h["status"] == "degraded" and "make up" in h["error"]
        assert c.post("/api/chat", json={"question": "hi"}).status_code == 503
        assert c.get("/").status_code == 200
    assert NS  # keep import used
