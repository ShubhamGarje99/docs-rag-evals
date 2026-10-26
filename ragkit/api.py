"""FastAPI backend: chat with citations, retrieval debug search, health, and the static UI.

uv run uvicorn ragkit.api:app --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import NO_KEY_MESSAGE, RETRIEVERS, get_settings
from .engine import RETRIEVER_LABELS, RAGEngine, source_dict
from .llm import LLMError

log = logging.getLogger("ragkit.api")
STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retriever: str | None = Field(default=None, description=f"one of {', '.join(RETRIEVERS)}")
    k: int | None = Field(default=None, ge=1, le=10)


def _engine(request: Request) -> RAGEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, getattr(request.app.state, "engine_error", None) or "Index not loaded")
    return engine


def _check_retriever(engine: RAGEngine, name: str | None) -> str:
    try:
        return engine.resolve_retriever(name)
    except KeyError as e:
        raise HTTPException(422, str(e.args[0])) from e


def create_app(engine: RAGEngine | None = None, warmup: bool | None = None) -> FastAPI:
    """App factory. Pass a prebuilt ``engine`` (tests) or let the lifespan build the production one."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine
        app.state.engine_error = None
        if app.state.engine is None:
            try:
                from .engine import build_engine

                app.state.engine = build_engine()
            except Exception as e:  # keep serving /healthz and the UI with a clear error
                log.error("Could not build the RAG engine: %s", e)
                app.state.engine_error = f"{type(e).__name__}: {e}. Run `make up && make ingest`."
        do_warmup = warmup if warmup is not None else os.environ.get("WARMUP", "1") != "0"
        if app.state.engine is not None and do_warmup and engine is None:
            for r in app.state.engine.available_retrievers:  # load models before the first demo query
                app.state.engine.search("warm up", r, 3)
        yield

    app = FastAPI(
        title="FastAPI Docs RAG",
        version="0.1.0",
        summary="Retrieval-augmented QA over the FastAPI docs with Claude citations and a refusal guard.",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def healthz(request: Request) -> dict[str, Any]:
        eng: RAGEngine | None = getattr(request.app.state, "engine", None)
        s = eng.settings if eng else get_settings()
        return {
            "status": "ok" if eng else "degraded",
            "index_loaded": eng is not None,
            "error": getattr(request.app.state, "engine_error", None),
            "chunks": len(eng.store.chunks()) if eng else 0,
            "llm_enabled": bool(eng and eng.llm_enabled),
            "embed_model": s.embed_model,
            "rerank_model": s.rerank_model if eng and eng.reranker else None,
            "claude_model": s.claude_model if eng and eng.llm_enabled else None,
            "retrievers": eng.available_retrievers if eng else [],
        }

    @app.get("/api/config")
    def config(request: Request) -> dict[str, Any]:
        eng = _engine(request)
        s = eng.settings
        return {
            "retrievers": [{"id": r, "label": RETRIEVER_LABELS[r]} for r in eng.available_retrievers],
            "default_retriever": eng.resolve_retriever(None),
            "top_k": s.top_k,
            "llm_enabled": eng.llm_enabled,
            "banner": None if eng.llm_enabled else NO_KEY_MESSAGE,
            "claude_model": s.claude_model if eng.llm_enabled else None,
            "embed_model": s.embed_model,
            "rerank_model": s.rerank_model if eng.reranker else None,
            "min_relevance": s.min_relevance,
            "chunks": len(eng.store.chunks()),
        }

    @app.get("/api/search")
    def search(
        request: Request,
        q: Annotated[str, Query(min_length=1, max_length=2000, description="search query")],
        retriever: Annotated[str | None, Query(description=f"one of {', '.join(RETRIEVERS)}")] = None,
        k: Annotated[int, Query(ge=1, le=50)] = 5,
    ) -> dict[str, Any]:
        eng = _engine(request)
        name = _check_retriever(eng, retriever)
        res = eng.search(q, name, k)
        rel = eng.relevance(q, [h.chunk_id for h in res.hits])
        return {
            "query": q,
            "retriever": res.retriever,
            "latency_ms": round(res.latency_ms, 1),
            "results": [source_dict(h, c, rel.get(c.chunk_id)) for h, c in zip(res.hits, res.chunks)],
        }

    @app.post("/api/chat")
    def chat(request: Request, body: ChatRequest) -> dict[str, Any]:
        eng = _engine(request)
        name = _check_retriever(eng, body.retriever)
        try:
            return eng.chat(body.question.strip(), name, body.k)
        except LLMError as e:
            raise HTTPException(502, str(e)) from e

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
