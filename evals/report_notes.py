"""Hand-written failure analysis rendered into EVAL_REPORT.md by evals/run.py.

Kept as code (not generated) because it is human judgement about the numbers. Re-read it whenever the
ablation changes; it refers to question ids in evals/golden.jsonl.
"""

ANALYSIS = """
_Hand-written notes, read against the run above (question ids refer to [evals/golden.jsonl](evals/golden.jsonl); per-question ranks are in `evals/results/retrieval_summary.json`)._

**1. Merging small sections hides content behind the wrong heading (q05, q22). All four retrievers miss both.**
The chunker merges small neighbouring sections to avoid tiny chunks. On `tutorial/metadata.md` the short
"OpenAPI URL" section, which is where `openapi_url=None` lives, ended up inside a chunk whose breadcrumb is
*Metadata for tags > Use your tags*. On `advanced/path-operation-advanced-configuration.md`, "Exclude from OpenAPI"
(`include_in_schema=False`) shares a chunk that opens with `operationId` material. Retrieval can't find a section
whose heading isn't what the chunk appears to be about. *Fix:* stop merging across H2 boundaries, and add every
heading inside a chunk to its search text. Then re-pool the gold labels and re-run.

**2. Paraphrased questions are the weakest type for every retriever (q18, q27, q28).**
Users describe a symptom instead of the API: "every reply should carry how many seconds the server spent" means
middleware plus `X-Process-Time`; "my signup endpoint echoes back the secret" means `response_model`; "React dev
server on port 3000 is blocked" means CORS (here the reranker pushed the CORS chunk from 4th to 9th, preferring
proxy and port chunks). Neither embeddings nor BM25 bridge that gap at this model size. *Fix:* rewrite the query
into the docs' vocabulary before retrieving (HyDE, or a Claude Haiku 5.5 rewrite costing a fraction of a cent).
Another option is doc2query: add generated questions to each chunk at ingest time. Score the change on the
paraphrased slice.

**3. Fusion can bury a strong lexical hit (q33, q08).**
For "my container takes ten seconds to stop", BM25 ranks the Docker *exec form* chunk 1st and vector search doesn't
find it in its top-10. Rank fusion averages the two lists, so the chunk lands 4th. q08 looks the same: BM25 ranks it
1st, vector 2nd, hybrid 4th. *Fix:* weighted RRF, with BM25 weighted up for queries that contain identifiers
(backticks, snake_case, CLI flags), or a smaller RRF `k`. Tune it on a dev split, not on this set.

**4. A reranker isn't automatically better.**
The MS MARCO MiniLM cross-encoder costs tens of times the latency of plain hybrid and doesn't improve any metric here
(Hit@1/MRR are slightly lower, within noise). It was trained on short web passages, and these chunks are long and
code-heavy. In the reranker comparison, the larger `bge-reranker-base` gets more gold chunks into the top 5, which is
what Claude sees, but doesn't improve Hit@1 or MRR, and it takes seconds per query on this laptop. That's why the app
defaults to plain hybrid. Measure on your own data before you ship a reranker. Fine-tuning one on synthetic
(question, chunk) pairs from this corpus would be the next experiment.

**5. The cosine refusal gate catches off-topic questions, not in-domain gaps.**
Clearly off-topic questions score far below the threshold. Plausible FastAPI questions the docs don't cover (Heroku,
gRPC, request timeouts) score as high as many answerable ones, so no threshold separates them without blocking real
questions. Refusing those is left to the prompt and the citation gate, and `make eval-full` measures that. The gate
threshold stays conservative on purpose.

**Labeling note.** After the first run, the top-5 of every retriever was pooled. A retrieved chunk was added as gold
only if it answers the question on its own (16 chunks across 13 questions; the 1-3 gold limit still holds). This is
standard IR pooling. Without it the ablation undercounted every retriever, because several questions are answered on
more than one page. Many client eval sets have the same blind spot.

**What I'd fix next, in order:** (1) chunk at H2 boundaries and index every heading in a chunk, which targets
q05/q22; (2) query rewriting for paraphrased questions; (3) identifier-aware weighted fusion; (4) grow the golden
set to 150+ with `scripts/generate_golden.py` plus human review, because ±10-point CIs are too wide to separate
close retrievers; (5) run `make eval-full`, then tune the answer prompt and the gate on generation metrics.
"""
