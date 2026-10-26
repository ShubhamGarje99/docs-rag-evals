"""Postgres + pgvector helpers: connection, migrations, per-model embedding tables."""

from __future__ import annotations

import re
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from .config import ROOT

MIGRATIONS_DIR = ROOT / "db" / "migrations"


def connect(database_url: str, register: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=True, connect_timeout=5)
    if register:
        try:
            register_vector(conn)
        except psycopg.ProgrammingError:
            # The extension does not exist yet (fresh database, before migrations).
            pass
    return conn


def migrate(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending ``db/migrations/*.sql`` files in order. Returns the applied versions."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    applied = []
    for path in sorted(migrations_dir.glob("*.sql")):
        version = path.stem
        if version in done:
            continue
        with conn.transaction():
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        applied.append(version)
    register_vector(conn)
    return applied


def embedding_table_name(model: str) -> str:
    return "emb_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def ensure_embedding_table(conn: psycopg.Connection, model: str, dim: int) -> str:
    table = embedding_table_name(model)
    row = conn.execute("SELECT table_name, dim FROM embedding_spaces WHERE model = %s", (model,)).fetchone()
    if row and row[1] != dim:
        raise RuntimeError(f"{model} is registered with dim={row[1]}, got {dim}; drop {row[0]} first")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {table} ("
        " chunk_id text PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,"
        f" embedding vector({dim}) NOT NULL,"
        " content_sha256 text NOT NULL)"
    )
    # HNSW with cosine distance; ef_search is raised per query in PgStore.
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx ON {table} "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    conn.execute(
        "INSERT INTO embedding_spaces (model, table_name, dim) VALUES (%s, %s, %s) ON CONFLICT (model) DO NOTHING",
        (model, table, dim),
    )
    return table


def table_for_model(conn: psycopg.Connection, model: str) -> str | None:
    row = conn.execute("SELECT table_name FROM embedding_spaces WHERE model = %s", (model,)).fetchone()
    return row[0] if row else None
