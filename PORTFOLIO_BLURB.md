# Portfolio entry

**Title:** FastAPI Docs Copilot: RAG chatbot with Claude citations and a measured retrieval ablation

**Description (3 lines):**
1. A RAG chatbot over the FastAPI docs. Every claim links to a quoted source passage (native Claude Citations), and
   a three-layer guard refuses questions the docs don't answer.
2. It runs on a FastAPI + Postgres/pgvector backend and compares vector, BM25, hybrid (RRF) and cross-encoder rerank
   retrieval live in the UI. Each answer shows its latency, tokens and cost.
3. It ships with evals: a 55-question golden set and a reproducible ablation (hybrid scores 87.5% Hit@5 vs 79.2%
   for BM25, and the reranker didn't pay for its latency), plus an LLM-judge harness for citation correctness,
   faithfulness and refusals.

**Skills tags:** RAG · Retrieval-Augmented Generation · LLM Evaluation · Claude API · Anthropic · Python · FastAPI ·
PostgreSQL · pgvector · Vector Search · BM25 · Hybrid Search · Reranking · Sentence Transformers · Embeddings ·
LLM-as-a-Judge · Prompt Engineering · Docker · pytest · REST API
