-- 001_init: base schema for docs-rag-evals.
-- Embedding vectors live in one table per embedding model (created by
-- ragkit.db.ensure_embedding_table) so that e.g. bge-small (384-d) and bge-m3
-- (1024-d) indexes can coexist and be A/B tested without re-ingesting text.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        text PRIMARY KEY,           -- e.g. tutorial/body.md#create-your-data-model-00
    doc_path        text        NOT NULL,       -- path inside corpus/docs
    page_title      text        NOT NULL,
    heading_path    text[]      NOT NULL,       -- breadcrumb
    anchor          text        NOT NULL,
    url             text        NOT NULL,       -- canonical https://fastapi.tiangolo.com/... deep link
    ordinal         integer     NOT NULL,       -- chunk index within its section
    position        integer     NOT NULL,       -- chunk index within its page
    content         text        NOT NULL,
    n_tokens        integer     NOT NULL,
    content_sha256  text        NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_doc_path_idx ON chunks (doc_path);

CREATE TABLE IF NOT EXISTS embedding_spaces (
    model       text PRIMARY KEY,               -- e.g. BAAI/bge-small-en-v1.5
    table_name  text        NOT NULL UNIQUE,    -- e.g. emb_baai_bge_small_en_v1_5
    dim         integer     NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS corpus_info (
    key    text PRIMARY KEY,
    value  text NOT NULL
);
