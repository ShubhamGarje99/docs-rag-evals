"""Runtime configuration, read from environment variables (and `.env` if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# `.env` never overrides variables that are already exported in the shell.
load_dotenv(ROOT / ".env", override=False)

RETRIEVERS = ("vector", "bm25", "hybrid", "hybrid_rerank")


def _env(name: str, default: str) -> str:
    val = os.environ.get(name, "")
    return val.strip() if val.strip() else default


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"))
    corpus_dir: Path = field(default_factory=lambda: Path(_env("CORPUS_DIR", str(ROOT / "corpus"))))

    # Retrieval models (local, via sentence-transformers)
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    device: str = field(default_factory=lambda: _env("EMBED_DEVICE", "auto"))  # auto|mps|cpu|cuda

    # Chunking (token counts are approximate, see ragkit.chunking.count_tokens)
    chunk_max_tokens: int = field(default_factory=lambda: int(_env("CHUNK_MAX_TOKENS", "400")))
    chunk_overlap_tokens: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP_TOKENS", "50")))

    # Retrieval
    default_retriever: str = field(default_factory=lambda: _env("RETRIEVER", "hybrid"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    candidate_k: int = field(default_factory=lambda: int(_env("CANDIDATE_K", "30")))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    # Refusal guard: if the best cosine similarity between the question and the
    # retrieved chunks is below this, refuse without calling Claude.
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.60")))

    # Claude
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    claude_model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-sonnet-5-5"))
    judge_model: str = field(default_factory=lambda: _env("CLAUDE_JUDGE_MODEL", "claude-haiku-5-5"))
    claude_effort: str = field(default_factory=lambda: _env("CLAUDE_EFFORT", "low"))
    claude_max_tokens: int = field(default_factory=lambda: int(_env("CLAUDE_MAX_TOKENS", "16000")))
    claude_fallbacks: bool = field(
        default_factory=lambda: _env("CLAUDE_FALLBACKS", "default").lower() not in {"off", "0", "false", "no"}
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


def get_settings() -> Settings:
    return Settings()


NO_KEY_MESSAGE = (
    "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example) to enable "
    "Claude answers; until then the app runs in retrieval-only mode."
)
