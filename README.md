# FastAPI Docs Copilot: RAG with citations, measured

**Ask the FastAPI docs a question. You get an answer in which every claim links to a quoted passage, or a plain
"I couldn't find that in the FastAPI docs." The retrieval behind it was chosen by a reproducible ablation, not by
taste.**

Python 3.12 · FastAPI · Postgres + pgvector · BM25 · Claude (native Citations) · pytest · MIT

The headline result is offline and reproducible with `make eval`, no API key needed. It covers 48 answerable golden
questions over 616 chunks of the FastAPI 0.162.0 docs; full details are in [EVAL_REPORT.md](EVAL_REPORT.md).

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Mean latency* |
|---|---:|---:|---:|---:|---:|
| Vector only (bge-small-en-v1.5, pgvector HNSW) | 52.1% | 85.4% | 89.6% | 0.664 | ≈10 ms |
| BM25 only (bm25s, stemmed) | 50.0% | 79.2% | 85.4% | 0.622 | <1 ms |
| **Hybrid: BM25 + vector with Reciprocal Rank Fusion (default)** | **52.1%** | **87.5%** | **93.8%** | **0.671** | ≈10 ms |
| Hybrid + MS MARCO MiniLM cross-encoder rerank | 50.0% | 87.5% | 93.8% | 0.651 | ≈300–400 ms |
| Hybrid + `bge-reranker-base` rerank (separate run) | 47.9% | 91.7% | 93.8% | 0.639 | seconds |

<sub>*Per query on an 8 GB Apple M3 (MPS), including query embedding; rerank latency varies from run to run, and the exact figures are in the report. With 48 questions, one question is worth 2.1
points; the 95% bootstrap CIs are in the report.</sub>

What the table says:
- Hybrid search beats either retriever alone at no extra latency.
- Neither cross-encoder improves Hit@1 or MRR on these long, code-heavy chunks. The larger one only gets more gold
  chunks into the top 5, at seconds per query. The app therefore defaults to plain hybrid, and the reranker stays
  one click away for the A/B.
- Paraphrased questions, where the user describes the problem without API terms, are the weakest slice for every
  retriever. That's where to spend effort next (see [How I'd fix your RAG](#how-id-fix-your-rag)).

## Screenshots

> _Placeholder: add `docs/screenshot-answer.png` (an answer with citation chips and the sources panel) and
> `docs/screenshot-refusal.png` (an out-of-corpus question refused). Record them with an API key set._

## What it does

- **Grounded answers with native Claude Citations.** Every retrieved chunk is sent as its own `document` block
  (`citations: {enabled: true}`, title = chunk id). Claude returns `char_location` citations whose `document_index`
  maps exactly back to a chunk id, a docs URL and the quoted sentence. The UI shows them as `[1] [2]` chips that open
  the quote and link to fastapi.tiangolo.com.
- **Three-layer refusal guard.**
  1. A retrieval gate skips the LLM call for off-topic questions (max question-to-chunk cosine < `MIN_RELEVANCE`).
  2. The prompt tells Claude to reply exactly "I couldn't find that in the FastAPI docs." when the excerpts don't
     answer the question.
  3. A citation gate replaces any answer with zero citations by the refusal.
- **Four retrievers, switchable live:** vector (pgvector cosine), BM25, hybrid (RRF), and hybrid + cross-encoder
  rerank. Each answer shows its latency, tokens and cost.
- **Works without a key.** With no `ANTHROPIC_API_KEY` the UI and `/api/chat` run in *retrieval-only mode*: you get
  the top sources, a banner, and what the refusal gate would do. The retrieval demo and every offline eval run
  without spending anything.
- **An eval harness you can hand to a client.**
  - 55-question golden set: factual, how-to, multi-chunk, paraphrased, and 7 unanswerable questions.
  - A sanity checker for the golden set.
  - Hit@k / MRR / Recall with bootstrap CIs, plus a per-type breakdown.
  - An offline analysis of the refusal gate.
  - Generation metrics (citation correctness, key-fact coverage, Claude Haiku 5.5 LLM judge, refusal accuracy, cost)
    behind `make eval-full`.

## Architecture

```mermaid
flowchart LR
  subgraph Ingest["Ingest (make ingest)"]
    A["corpus/docs/*.md<br/>FastAPI 0.162.0, MIT"] --> B["Heading-aware chunker<br/>≤400 tok, 50 overlap<br/>stable ids"]
    B --> C["bge-small-en-v1.5<br/>embeddings (local)"]
    B --> P[("Postgres 16 + pgvector<br/>chunks + HNSW index")]
    C --> P
  end

  subgraph Query["Query (POST /api/chat)"]
    Q["Question"] --> V["Vector search<br/>pgvector cosine"]
    Q --> K["BM25<br/>bm25s"]
    V --> F["RRF fusion"]
    K --> F
    F --> R["Cross-encoder rerank<br/>(optional)"]
    F --> G{"Retrieval gate<br/>max cosine ≥ MIN_RELEVANCE?"}
    R --> G
    G -- no --> X["Refuse<br/>(no LLM call)"]
    G -- yes --> L["Claude Sonnet 5.5<br/>chunks as Citations documents"]
    L --> CG{"≥1 citation?"}
    CG -- no --> X
    CG -- yes --> AN["Answer + [n] citations<br/>+ latency / tokens / cost"]
  end
  P -.-> V
  P -.-> K

  subgraph Eval["Evals (make eval / eval-full)"]
    GS["golden.jsonl<br/>55 questions"] --> RA["Retrieval ablation<br/>Hit@k, MRR, CIs, latency"]
    GS --> GE["Generation eval<br/>citations, key facts,<br/>Haiku 5.5 judge, refusals"]
    RA --> REP["EVAL_REPORT.md"]
    GE --> REP
  end
  Query -.-> Eval
```

## Quickstart

Prerequisites: Docker (Docker Desktop or colima), [uv](https://docs.astral.sh/uv/), `make`. The first run downloads
about 250 MB of local models (bge-small and MiniLM) from Hugging Face.

```bash
cp .env.example .env        # optional: add ANTHROPIC_API_KEY for Claude answers
make demo                   # uv sync -> Postgres (port 5433) -> ingest if needed -> http://127.0.0.1:8000
```

`make demo` runs these steps:
1. Creates `.venv` with Python 3.12 from `uv.lock`.
2. Starts `pgvector/pgvector:pg16` with `docker compose`.
3. Applies the SQL migrations.
4. Chunks and embeds the corpus. This takes about a minute the first time and is incremental after that.
5. Serves the API and UI with uvicorn.

Stop the server with Ctrl+C. `make down` stops the database and keeps its volume; `make nuke` deletes the volume too.

| Command | What it does |
|---|---|
| `make demo` | Everything above in one command |
| `make up` / `make down` / `make nuke` | Start or stop the database / delete its volume |
| `make ingest` | Migrate, chunk and embed (idempotent; re-embeds only changed chunks) |
| `make serve` | API + UI only (database must be up and ingested) |
| `make test` | 44 unit and API tests (no db, no downloads, no key) |
| `make test-integration` | Tests against the real Postgres index and models |
| `make eval` | Retrieval ablation → `EVAL_REPORT.md` (offline) |
| `make eval-full` | Adds generation metrics with Claude (needs a key) |
| `make eval-rerankers` | Compares MiniLM vs `BAAI/bge-reranker-base` (1.1 GB download) |
| `make golden-check` | Validates the golden set against the corpus |
| `make golden-candidates` | Claude proposes new golden questions for human review (needs a key) |
| `make corpus` | Re-downloads the pinned FastAPI tag and rebuilds `corpus/` |
| `make lint` | ruff |

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` `{question, retriever?, k?}` | Answer + `citations[{n, chunk_id, title, url, quote}]` + `sources` + latency / tokens / cost. Retrieval-only mode without a key. |
| `GET /api/search?q=&retriever=&k=` | Retrieval debugging: ranked chunks with per-retriever scores (`vector_rank`, `bm25_rank`, `rrf`, `rerank`, cosine) |
| `GET /api/config` | Available retrievers, default, LLM status (used by the UI) |
| `GET /healthz` | Index and LLM status |
| `GET /` | Single-page UI (static HTML/JS, no Node build); interactive OpenAPI docs at `/docs` |

```bash
curl -s "localhost:8000/api/search?q=how+do+I+enable+CORS&retriever=hybrid&k=3" | jq '.results[].chunk_id'
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
     -d '{"question": "How do I run code after the response is sent?", "retriever": "hybrid"}' | jq
```

## Configuration

All settings come from environment variables or `.env` (see [.env.example](.env.example)).

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | _(empty)_ | Empty means retrieval-only mode |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Answer model |
| `CLAUDE_JUDGE_MODEL` | `claude-haiku-5-5` | LLM judge in `make eval-full` |
| `CLAUDE_EFFORT` | `low` | `output_config.effort` for answers; no sampling params are sent (Sonnet/Haiku 5.5 reject them) |
| `CLAUDE_FALLBACKS` | `default` | Server-side refusal fallback on Sonnet (`fallbacks="default"`); `off` disables it. Never sent to Haiku. |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5433/rag` | Compose maps Postgres to host port `PG_PORT` (5433) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | `BAAI/bge-m3` also works (1024-d; each model gets its own table) |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Any sentence-transformers cross-encoder |
| `EMBED_DEVICE` | `auto` | `mps` / `cuda` / `cpu` |
| `RETRIEVER` | `hybrid` | `vector` / `bm25` / `hybrid` / `hybrid_rerank` |
| `TOP_K` / `CANDIDATE_K` / `RRF_K` | `5` / `30` / `60` | Context chunks / rerank candidates / RRF constant |
| `MIN_RELEVANCE` | `0.60` | Retrieval refusal gate (see the threshold sweep in the report) |
| `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `400` / `50` | Chunker budget |

## Evals

```bash
make eval        # offline: retrieval ablation, per-type breakdown, refusal-gate analysis -> EVAL_REPORT.md
make eval-full   # + generation metrics (needs ANTHROPIC_API_KEY in .env)
```

- **Golden set** ([evals/golden.jsonl](evals/golden.jsonl)): 48 answerable questions (12 factual, 14 how-to,
  8 multi-chunk, 14 paraphrased), each with a reference answer, 1-3 gold chunk ids and verbatim key facts. There
  are also 7 plausible but unanswerable questions (rate limiting, Strawberry GraphQL, AWS Lambda, Heroku…), and the
  checker verifies their terms appear nowhere in the corpus.
  - The questions were written by hand without an API key, deliberately avoiding the chunks' wording (mean
    question-to-chunk lexical overlap is 0.38 for the paraphrased type and 0.47 overall).
  - Gold labels were completed by pooling the top-5 of all retrievers.
  - `scripts/generate_golden.py` is the Claude-assisted way to grow the set; its output goes to human review.
- **Checker** (`make golden-check`): every gold id exists, every key fact occurs in its gold chunks, no
  (near-)duplicate questions, and unanswerable terms are absent from the corpus. It also reports lexical overlap per
  type.
- **Retrieval metrics** (offline): Hit@1/3/5/10, MRR@10, Recall@5 and 95% bootstrap CIs, overall and per question
  type, plus mean/p95 latency per retriever. There is also a threshold sweep for the refusal gate, with off-topic
  probes.
- **Generation metrics: pending.** They need `make eval-full` with `ANTHROPIC_API_KEY` set, and no generation numbers
  are reported until that run exists. It measures:
  - citation correctness: every cited chunk was in the provided context and at least one is gold;
  - key-fact coverage;
  - faithfulness, correctness, completeness and citation quality from a Claude Haiku 5.5 judge with structured
    output and per-criterion rationales;
  - refusal accuracy on unanswerable questions and the false-refusal rate on answerable ones;
  - total run cost at list prices (Sonnet 5.5 $2/$10, Haiku 5.5 $0.10/$0.50 per MTok).

  The run is about 55 Sonnet calls with roughly 2-3k input tokens each, plus about 48 Haiku judge calls, so it
  should cost well under a dollar. The actual cost is printed and written into the report.

## How I'd fix your RAG

The checklist this repo demonstrates, in the order I'd apply it to a client system:

1. **Build a golden set before you touch the pipeline.**
   - 50+ real questions, tagged by type, with gold chunk ids and key facts.
   - Include unanswerable questions; a bot that never says "I don't know" is a liability.
   - Write paraphrased questions on purpose, because keyword-overlap questions flatter BM25.
2. **Pool your labels.** Review the top-k of every retriever and add chunks that also answer. Unpooled gold sets
   undercount every system and make close comparisons meaningless.
3. **Run the ablation and report uncertainty.** Measure vector vs BM25 vs hybrid vs rerank on Hit@k and MRR, with
   CIs and latency. Here, hybrid was a free win and the reranker was not; on 48 questions a few points is noise.
4. **Read the failures by type.** Here that surfaced three things:
   - a chunker bug: merged sections hide content behind the wrong heading;
   - a vocabulary gap on paraphrased questions, which calls for query rewriting or HyDE;
   - fusion burying strong lexical hits, which calls for weighted RRF on identifier-heavy queries.
5. **Ground generation and measure it separately.**
   - Use native citations mapped to chunk ids, not "please cite [n]" prompting.
   - Score citation correctness and faithfulness with an LLM judge that uses a fixed rubric and structured output.
   - Track the false-refusal rate next to refusal accuracy.
6. **Gate cheaply, refuse honestly.** An embedding-similarity gate removes off-topic traffic before the LLM call.
   In-domain gaps need the prompt plus a no-citation → refusal rule, and they have to be measured.
7. **Put cost and latency in every answer.** Then the "better" variant has a price tag.

## Corpus, attribution and licenses

- **Source:** [fastapi/fastapi](https://github.com/fastapi/fastapi), `docs/en/docs/` at tag **`0.162.0`** (commit
  `7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4`).
- **Upstream license:** MIT. The upstream LICENSE text was checked and is copied verbatim to
  [corpus/LICENSE](corpus/LICENSE).
- **Subset:** 99 pages (`tutorial/`, `advanced/`, `deployment/`, plus `async.md`, `python-types.md`, `features.md`,
  `fastapi-cli.md`). Attribution and the list of modifications are in [corpus/NOTICE.md](corpus/NOTICE.md); the
  per-page provenance (upstream sha256, canonical URL) is in `corpus/manifest.json`.
- **Rebuilding it:** `make corpus` re-downloads the tag, verifies the commit and LICENSE, and rebuilds the snapshot
  deterministically.
- **mkdocs syntax:** `{* ../../docs_src/... *}` includes are replaced by the referenced example code; admonitions,
  tabs and inline HTML become plain Markdown.

This project is not affiliated with or endorsed by the FastAPI project.

## Project layout

```
ragkit/            chunking.py · embeddings.py · db.py · store.py · retrieval.py (BM25, RRF, rerank)
                   llm.py (Claude + Citations + refusal guard + cost) · judge.py · engine.py · api.py · static/
db/migrations/     SQL schema (pgvector extension, chunks, per-model embedding tables)
corpus/            vendored FastAPI docs subset + LICENSE + NOTICE + manifest
evals/             golden.jsonl · check_golden.py · run.py · report_notes.py · results/ (aggregates)
scripts/           fetch_corpus.py · generate_golden.py
tests/             chunker, RRF, metrics, citation mapping / refusal guard (mocked Claude), API
```

## Limitations

- Generation quality is **unmeasured until `make eval-full` runs with a key**. The citation mapping, refusal guard
  and request shape are covered by unit tests with a mocked Claude response, not by a live call from this
  environment.
- 48 answerable questions give wide CIs (about ±10 points on Hit@5), so close retrievers can't be separated.
- The token budget uses a tokenizer-free proxy. Chunks are ≤400 proxy tokens, which keeps most under bge-small's
  512-token limit, but some code-heavy chunks may be truncated when embedded.
- The UI is single-turn: each question is answered independently, without chat memory.

## License

Code: [MIT](LICENSE) © 2026 Shubham Garje. The documentation corpus in `corpus/` stays under FastAPI's MIT license
(© 2018 Sebastián Ramírez); see [corpus/LICENSE](corpus/LICENSE) and [corpus/NOTICE.md](corpus/NOTICE.md).
