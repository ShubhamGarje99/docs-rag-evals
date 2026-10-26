"""Ingest the vendored corpus into Postgres/pgvector (idempotent, incremental).

    uv run python -m ragkit.ingest            # migrate, chunk, embed what changed
    uv run python -m ragkit.ingest --reset    # drop all chunks/embeddings first

Only new or changed chunks are embedded, so re-running on an up-to-date
database takes a couple of seconds and never loads the embedding model.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

from .chunking import Chunk, chunk_corpus, load_manifest
from .config import get_settings


def chunk_hash(c: Chunk) -> str:
    h = hashlib.sha256()
    for part in (c.chunk_id, c.url, c.breadcrumb, c.text):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest corpus/ into Postgres + pgvector")
    ap.add_argument("--reset", action="store_true", help="delete all chunks and embeddings first")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    import psycopg

    from .db import connect, embedding_table_name, ensure_embedding_table, migrate

    s = get_settings()
    t0 = time.perf_counter()
    try:
        conn = connect(s.database_url, register=False)
    except psycopg.OperationalError as e:
        print(f"Cannot connect to {s.database_url}: {e}\nStart the database with `make up`.", file=sys.stderr)
        return 2
    applied = migrate(conn)
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")

    if args.reset:
        conn.execute("TRUNCATE chunks CASCADE")
        print("Reset: all chunks and embeddings deleted")

    chunks = chunk_corpus(s.corpus_dir, s.chunk_max_tokens, s.chunk_overlap_tokens)
    hashes = {c.chunk_id: chunk_hash(c) for c in chunks}
    print(f"Chunked corpus: {len(chunks)} chunks from {len({c.doc_path for c in chunks})} pages")

    existing = dict(conn.execute("SELECT chunk_id, content_sha256 FROM chunks").fetchall())
    stale = [cid for cid in existing if cid not in hashes]
    changed = [c for c in chunks if existing.get(c.chunk_id) != hashes[c.chunk_id]]
    with conn.transaction():
        if stale:
            conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (stale,))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (chunk_id, doc_path, page_title, heading_path, anchor, url, ordinal,
                                       position, content, n_tokens, content_sha256)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO UPDATE SET
                     doc_path = EXCLUDED.doc_path, page_title = EXCLUDED.page_title,
                     heading_path = EXCLUDED.heading_path, anchor = EXCLUDED.anchor, url = EXCLUDED.url,
                     ordinal = EXCLUDED.ordinal, position = EXCLUDED.position, content = EXCLUDED.content,
                     n_tokens = EXCLUDED.n_tokens, content_sha256 = EXCLUDED.content_sha256,
                     ingested_at = now()""",
                [
                    (
                        c.chunk_id,
                        c.doc_path,
                        c.page_title,
                        c.heading_path,
                        c.anchor,
                        c.url,
                        c.ordinal,
                        c.position,
                        c.text,
                        c.n_tokens,
                        hashes[c.chunk_id],
                    )
                    for c in changed
                ],
            )
    print(f"Chunks: {len(changed)} upserted, {len(stale)} deleted, {len(chunks) - len(changed)} unchanged")

    table = embedding_table_name(s.embed_model)
    have_table = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()[0]
    embedded = dict(conn.execute(f"SELECT chunk_id, content_sha256 FROM {table}").fetchall()) if have_table else {}
    todo = [c for c in chunks if embedded.get(c.chunk_id) != hashes[c.chunk_id]]
    if todo or not have_table:
        from pgvector.psycopg import register_vector

        from .embeddings import SentenceTransformerEmbedder

        register_vector(conn)
        emb = SentenceTransformerEmbedder(s.embed_model, s.device, batch_size=args.batch_size)
        print(f"Embedding {len(todo)} chunks with {s.embed_model} on {emb.device} ...")
        table = ensure_embedding_table(conn, s.embed_model, emb.dim)
        te = time.perf_counter()
        vecs = emb.embed_documents([c.search_text for c in todo], show_progress=len(todo) > 64) if todo else []
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {table} (chunk_id, embedding, content_sha256) VALUES (%s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding,
                    content_sha256 = EXCLUDED.content_sha256""",
                [(c.chunk_id, v, hashes[c.chunk_id]) for c, v in zip(todo, vecs)],
            )
        print(f"Embedded {len(todo)} chunks in {time.perf_counter() - te:.1f}s")
    else:
        print(f"Embeddings ({s.embed_model}): all {len(chunks)} up to date")

    manifest = load_manifest(s.corpus_dir) if (s.corpus_dir / "manifest.json").exists() else {}
    with conn.transaction():
        for k in ("source_repo", "tag", "commit"):
            if k in manifest:
                conn.execute(
                    "INSERT INTO corpus_info (key, value) VALUES (%s, %s)"
                    " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (k, str(manifest[k])),
                )
    n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"Done in {time.perf_counter() - t0:.1f}s: {n} chunks indexed in {table} (HNSW, cosine)")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
