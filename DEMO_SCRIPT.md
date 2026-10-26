# 90-second Loom demo script

**Before recording:**
- Put `ANTHROPIC_API_KEY` in `.env`.
- Run `make eval-full` once, so EVAL_REPORT.md has real generation numbers.
- Run `make demo` and wait for `Uvicorn running`. The server warms the models at startup.
- Ask one throwaway question so the first answer you record isn't a cold start.
- Browser at 125% zoom, dark theme, a clean chat. Keep EVAL_REPORT.md open on GitHub in a second tab.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:10 | The app's empty state. The retriever selector is visible, set to **Hybrid**. | "Most RAG chatbots answer confidently even when they're wrong. This one answers from the FastAPI docs with clickable citations, says 'not found' when it should, and every design choice is backed by an eval." |
| 0:10–0:30 | Type *"How do I let my React app on localhost:3000 call my API?"* Click a `[1]` chip so the quote popover opens, then click "Open in the FastAPI docs". | "That's Claude's native Citations feature. Every retrieved chunk goes in as its own document, so each claim maps to an exact quoted sentence and a docs link. On the right are the retrieved chunks with their scores, and the cited ones are highlighted. Under the answer: latency, tokens, cost." |
| 0:30–0:45 | Type *"How do I add rate limiting of 100 requests per minute per IP?"* The refusal bubble appears with its tag. | "That isn't in the FastAPI docs, so it says so. There are three guards: a retrieval-score gate, the prompt, and a rule that an answer without citations becomes a refusal. It doesn't invent a decorator." |
| 0:45–1:00 | Switch to **Vector** and ask *"My container takes about ten seconds to stop and my shutdown hooks never fire. What's wrong?"* Then switch to **Hybrid** and ask it again. | "Same question, two retrievers. Vector search only finds generic lifespan pages. Hybrid adds BM25 and pulls in the Docker 'use the exec form of CMD' section, which is the actual fix." |
| 1:00–1:20 | The EVAL_REPORT.md ablation table, then the per-type table, then the generation metrics. | "And this is how I chose the default. 55 golden questions, including unanswerable ones. Hybrid beats either retriever alone, and the reranker didn't pay for its latency. Paraphrased questions are the weak spot, and the report says what I'd fix next. Generation is scored on citation correctness, faithfulness and refusals, with the run's cost." |
| 1:20–1:30 | The README "How I'd fix your RAG" checklist. | "If your RAG bot makes things up, or you can't tell whether a change helped, I'll set up this kind of eval harness on your data and fix what it finds. I build production agents that ship with evals." |

**Fallback without a key:** the app runs in retrieval-only mode. Skip the 0:10–0:45 answer beats, show the sources
panel and the "would refuse / would pass" line instead, and say that generation metrics are pending a key.
