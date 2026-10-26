"""Chunk stores: Postgres/pgvector for the app, in-memory numpy for tests."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .chunking import Chunk


class NotIngestedError(RuntimeError):
    pass


class ChunkStore(Protocol):
    embed_model: str

    def chunks(self) -> list[Chunk]: ...

    def get(self, chunk_id: str) -> Chunk: ...

    def vector_search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def similarities(self, qvec: np.ndarray, chunk_ids: Sequence[str]) -> dict[str, float]: ...


class MemoryStore:
    """Exact cosine search over an in-memory matrix (tests / offline smoke runs)."""

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embed_model: str = "memory"):
        assert len(chunks) == len(embeddings)
        self._chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self._chunks}
        self._idx = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        self._emb = np.asarray(embeddings, dtype=np.float32)
        self.embed_model = embed_model

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embedder) -> MemoryStore:
        emb = embedder.embed_documents([c.search_text for c in chunks])
        return cls(chunks, emb, embedder.name)

    def chunks(self) -> list[Chunk]:
        return self._chunks

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def vector_search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        sims = self._emb @ qvec
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self._chunks[i].chunk_id, float(sims[i])) for i in order]

    def similarities(self, qvec: np.ndarray, chunk_ids: Sequence[str]) -> dict[str, float]:
        return {cid: float(self._emb[self._idx[cid]] @ qvec) for cid in chunk_ids if cid in self._idx}


class PgStore:
    """pgvector-backed store. Chunk metadata (a few hundred rows) is cached in memory."""

    def __init__(self, database_url: str, embed_model: str, ef_search: int = 100):
        from .db import connect, table_for_model

        self.database_url = database_url
        self.embed_model = embed_model
        self._lock = threading.Lock()
        self._conn = connect(database_url)
        self._table = table_for_model(self._conn, embed_model) if self._has_schema() else None
        if not self._table:
            raise NotIngestedError(f"No embeddings for {embed_model} yet - run `make ingest`.")
        self._conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")
        rows = self._conn.execute(
            f"SELECT c.chunk_id, c.doc_path, c.page_title, c.heading_path, c.anchor, c.url,"
            f" c.ordinal, c.position, c.content, c.n_tokens"
            f" FROM chunks c JOIN {self._table} e USING (chunk_id)"
            f" ORDER BY c.doc_path, c.position"
        ).fetchall()
        if not rows:
            raise NotIngestedError("The chunks table is empty - run `make ingest`.")
        self._chunks = [
            Chunk(
                chunk_id=r[0],
                doc_path=r[1],
                page_title=r[2],
                heading_path=list(r[3]),
                anchor=r[4],
                url=r[5],
                ordinal=r[6],
                position=r[7],
                text=r[8],
                n_tokens=r[9],
            )
            for r in rows
        ]
        self._by_id = {c.chunk_id: c for c in self._chunks}

    def _has_schema(self) -> bool:
        return bool(self._conn.execute("SELECT to_regclass('public.embedding_spaces') IS NOT NULL").fetchone()[0])

    def chunks(self) -> list[Chunk]:
        return self._chunks

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def _execute(self, sql: str, params: tuple):
        with self._lock:
            try:
                return self._conn.execute(sql, params).fetchall()
            except Exception:
                # One reconnect attempt (e.g. the db container was restarted).
                from .db import connect

                self._conn = connect(self.database_url)
                self._conn.execute("SET hnsw.ef_search = 100")
                return self._conn.execute(sql, params).fetchall()

    def vector_search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        rows = self._execute(
            f"SELECT chunk_id, 1 - (embedding <=> %s) AS sim FROM {self._table} ORDER BY embedding <=> %s LIMIT %s",
            (qvec, qvec, k),
        )
        return [(r[0], float(r[1])) for r in rows]

    def similarities(self, qvec: np.ndarray, chunk_ids: Sequence[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        rows = self._execute(
            f"SELECT chunk_id, 1 - (embedding <=> %s) FROM {self._table} WHERE chunk_id = ANY(%s)",
            (qvec, list(chunk_ids)),
        )
        return {r[0]: float(r[1]) for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
