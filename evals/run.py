"""Run the evals and regenerate EVAL_REPORT.md.

    uv run python evals/run.py            # retrieval ablation (offline, no API key)   == make eval
    uv run python evals/run.py --full     # + generation metrics via Claude             == make eval-full

Outputs
  evals/runs/<timestamp>/*.json          per-question details (git-ignored run artifacts)
  evals/results/retrieval_summary.json   aggregated retrieval numbers behind the report (committed)
  evals/results/generation_summary.json  aggregated generation numbers (written by --full)
  EVAL_REPORT.md                         regenerated from the two summaries + evals/analysis.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.chunking import load_manifest  # noqa: E402
from ragkit.config import NO_KEY_MESSAGE, RETRIEVERS, Settings, get_settings  # noqa: E402
from ragkit.golden import QUESTION_TYPES, GoldenItem, load_golden  # noqa: E402
from ragkit.metrics import (  # noqa: E402
    auroc,
    citation_checks,
    first_relevant_rank,
    hit_at_k,
    key_fact_coverage,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

RESULTS_DIR = ROOT / "evals" / "results"
RUNS_DIR = ROOT / "evals" / "runs"
REPORT_PATH = ROOT / "EVAL_REPORT.md"
K_VALUES = (1, 3, 5, 10)
EVAL_K = 10
LABELS = {
    "vector": "Vector only (bge-small, pgvector cosine)",
    "bm25": "BM25 only",
    "hybrid": "Hybrid (BM25 + vector, RRF k=60)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}
# Clearly out-of-domain questions: what the cosine gate is actually for.
OFF_TOPIC_PROBES = (
    "What's a good recipe for vegetarian lasagna?",
    "Who won the FIFA World Cup in 2018?",
    "How do I center a div with CSS grid?",
)
SHORT = {"vector": "Vector", "bm25": "BM25", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid+rerank"}


def label(r: str) -> str:
    if r.startswith("hybrid_rerank:"):
        return f"Hybrid + `{r.split(':', 1)[1]}` rerank"
    return LABELS.get(r, r)


def short(r: str) -> str:
    return r.split("/", 1)[-1] if r.startswith("hybrid_rerank:") else SHORT.get(r, r)


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 13) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean, resampling questions (seeded => reproducible)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def load_analysis() -> str:
    from evals.report_notes import ANALYSIS

    return ANALYSIS


def golden_hash() -> str:
    return hashlib.sha256((ROOT / "evals" / "golden.jsonl").read_bytes()).hexdigest()[:12]


def git_rev() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True
        ).stdout
        return rev + ("+dirty" if dirty.strip() else "")
    except Exception:
        return "uncommitted"


def r4(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 4)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def run_retrieval(engine, items: list[GoldenItem], retrievers: list[str], gate_retriever: str) -> tuple[dict, list]:
    s: Settings = engine.settings
    for r in retrievers:  # warm-up: load models / caches so latency is steady-state
        for _ in range(2):
            engine.search("How do I declare a path parameter?", r, EVAL_K)

    rows = []
    for n, it in enumerate(items, start=1):
        row = {
            "id": it.id,
            "type": it.type,
            "answerable": it.answerable,
            "question": it.question,
            "gold": it.gold_chunk_ids,
            "results": {},
        }
        for r in retrievers:
            res = engine.search(it.question, r, EVAL_K)
            ranked = [h.chunk_id for h in res.hits]
            row["results"][r] = {
                "ranked": ranked,
                "scores": [round(h.score, 5) for h in res.hits],
                "latency_ms": round(res.latency_ms, 2),
                "first_gold_rank": first_relevant_rank(ranked, it.gold_chunk_ids) if it.answerable else None,
            }
        gate = row["results"][gate_retriever]
        top = gate["ranked"][: s.top_k]
        rel = engine.relevance(it.question, top)
        ce = row["results"].get("hybrid_rerank")
        row["gate"] = {
            "max_cosine": round(max(rel.values(), default=0.0), 5),
            # cross-encoder logit of the best candidate: an alternative gate signal
            "top_score": ce["scores"][0] if ce and ce["scores"] else None,
        }
        rows.append(row)
        if n % 10 == 0:
            print(f"  retrieval: {n}/{len(items)} questions")

    answerable = [r for r in rows if r["answerable"]]
    overall = {}
    for r in retrievers:
        lat = [row["results"][r]["latency_ms"] for row in rows]
        m = {
            f"hit@{k}": r4(mean(hit_at_k(x["results"][r]["ranked"], x["gold"], k) for x in answerable))
            for k in K_VALUES
        }
        m["mrr@10"] = r4(mean(reciprocal_rank(x["results"][r]["ranked"], x["gold"], 10) for x in answerable))
        m["recall@5"] = r4(mean(recall_at_k(x["results"][r]["ranked"], x["gold"], 5) for x in answerable))
        lo, hi = bootstrap_ci([hit_at_k(x["results"][r]["ranked"], x["gold"], 5) for x in answerable])
        m["hit@5_ci95"] = [r4(lo), r4(hi)]
        lo, hi = bootstrap_ci([reciprocal_rank(x["results"][r]["ranked"], x["gold"], 10) for x in answerable])
        m["mrr@10_ci95"] = [r4(lo), r4(hi)]
        m["latency_ms_mean"] = round(mean(lat), 1)
        m["latency_ms_p50"] = round(percentile(lat, 50), 1)
        m["latency_ms_p95"] = round(percentile(lat, 95), 1)
        overall[r] = m

    by_type: dict[str, dict] = {}
    for t in QUESTION_TYPES:
        sub = [x for x in answerable if x["type"] == t]
        if not sub:
            continue
        by_type[t] = {"n": len(sub)}
        for r in retrievers:
            by_type[t][r] = {
                "hit@1": r4(mean(hit_at_k(x["results"][r]["ranked"], x["gold"], 1) for x in sub)),
                "hit@5": r4(mean(hit_at_k(x["results"][r]["ranked"], x["gold"], 5) for x in sub)),
                "mrr@10": r4(mean(reciprocal_rank(x["results"][r]["ranked"], x["gold"], 10) for x in sub)),
            }

    # Refusal-gate signal separability (answerable vs unanswerable), offline.
    unans = [r for r in rows if not r["answerable"]]
    pos_cos = [x["gate"]["max_cosine"] for x in answerable]
    neg_cos = [x["gate"]["max_cosine"] for x in unans]
    pos_top = [x["gate"]["top_score"] for x in answerable if x["gate"]["top_score"] is not None]
    neg_top = [x["gate"]["top_score"] for x in unans if x["gate"]["top_score"] is not None]
    thr = s.min_relevance
    gate = {
        "retriever": gate_retriever,
        "top_k": s.top_k,
        "threshold": thr,
        "max_cosine": {
            "answerable": {"min": r4(min(pos_cos)), "p10": r4(percentile(pos_cos, 10)), "mean": r4(mean(pos_cos))},
            "unanswerable": {
                "min": r4(min(neg_cos)) if neg_cos else None,
                "mean": r4(mean(neg_cos)),
                "max": r4(max(neg_cos)) if neg_cos else None,
            },
            "auroc": r4(auroc(pos_cos, neg_cos)),
            "answerable_blocked_at_threshold": sum(v < thr for v in pos_cos),
            "unanswerable_caught_at_threshold": sum(v < thr for v in neg_cos),
        },
        "top_score": {
            "label": "Cross-encoder top logit (MiniLM)",
            "answerable": {"min": r4(min(pos_top)) if pos_top else None, "mean": r4(mean(pos_top))},
            "unanswerable": {"mean": r4(mean(neg_top)), "max": r4(max(neg_top)) if neg_top else None},
            "auroc": r4(auroc(pos_top, neg_top)),
        },
        "n_answerable": len(pos_cos),
        "n_unanswerable": len(neg_cos),
        "sweep": [
            {
                "threshold": t,
                "answerable_blocked": sum(v < t for v in pos_cos),
                "unanswerable_caught": sum(v < t for v in neg_cos),
            }
            for t in (0.60, 0.65, 0.70, 0.72, 0.75, 0.78)
        ],
        "off_topic_probes": [
            {
                "question": q,
                "max_cosine": r4(
                    max(
                        engine.relevance(
                            q, [h.chunk_id for h in engine.search(q, gate_retriever, s.top_k).hits]
                        ).values(),
                        default=0.0,
                    )
                ),
            }
            for q in OFF_TOPIC_PROBES
        ],
    }

    # Where reranking helped / hurt (gold inside top-5 or not).
    rerank_delta = None
    if "hybrid" in retrievers and "hybrid_rerank" in retrievers:
        helped = [
            x["id"]
            for x in answerable
            if not hit_at_k(x["results"]["hybrid"]["ranked"], x["gold"], 5)
            and hit_at_k(x["results"]["hybrid_rerank"]["ranked"], x["gold"], 5)
        ]
        hurt = [
            x["id"]
            for x in answerable
            if hit_at_k(x["results"]["hybrid"]["ranked"], x["gold"], 5)
            and not hit_at_k(x["results"]["hybrid_rerank"]["ranked"], x["gold"], 5)
        ]
        rerank_delta = {"helped_at_5": helped, "hurt_at_5": hurt}

    summary = {
        "overall": overall,
        "by_type": by_type,
        "refusal_gate": gate,
        "rerank_delta": rerank_delta,
        "per_question": [
            {
                "id": x["id"],
                "type": x["type"],
                "first_gold_rank": {r: x["results"][r]["first_gold_rank"] for r in retrievers},
                "max_cosine": x["gate"]["max_cosine"],
                "top_score": x["gate"]["top_score"],
            }
            for x in rows
        ],
    }
    return summary, rows


# --------------------------------------------------------------------------- #
# Generation (needs ANTHROPIC_API_KEY)
# --------------------------------------------------------------------------- #


def run_generation(engine, items: list[GoldenItem], retriever: str) -> tuple[dict, list]:
    from ragkit.judge import Judge
    from ragkit.llm import LLMError

    s: Settings = engine.settings
    judge = Judge(s)
    rows, errors = [], 0
    for n, it in enumerate(items, start=1):
        try:
            out = engine.chat(it.question, retriever, s.top_k)
        except LLMError as e:
            errors += 1
            rows.append({"id": it.id, "type": it.type, "error": str(e)})
            print(f"  {it.id}: ERROR {e}")
            if errors >= 5:
                raise SystemExit("Too many Claude API errors, aborting the generation eval.") from e
            continue
        context_ids = [x["chunk_id"] for x in out["sources"]]
        cited = [c["chunk_id"] for c in out["citations"]]
        row = {
            "id": it.id,
            "type": it.type,
            "answerable": it.answerable,
            "question": it.question,
            "answer": out["answer"],
            "refused": out["refused"],
            "refusal_reason": out["refusal_reason"],
            "context": context_ids,
            "cited": cited,
            "citations": out["citations"],
            "usage": out["usage"],
            "latency_ms": out["latency_ms"],
            "fallback_used": out.get("fallback_used"),
            "judge": None,
            "judge_usage": None,
        }
        if it.answerable and not out["refused"]:
            row["citation"] = citation_checks(cited, context_ids, it.gold_chunk_ids)
            row["key_fact_coverage"] = key_fact_coverage(out["answer"], it.key_facts)
            chunks = [engine.store.get(c) for c in context_ids]
            verdict, jusage, _, jerr = judge.grade(
                it.question, it.reference_answer, it.key_facts, chunks, out["answer"], out["citations"]
            )
            row["judge"] = verdict.model_dump() if verdict else None
            row["judge_error"] = jerr
            row["judge_usage"] = jusage.to_dict() if jusage else None
        elif it.answerable:
            row["key_fact_coverage"] = 0.0
        rows.append(row)
        if n % 5 == 0:
            print(f"  generation: {n}/{len(items)} questions")

    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    ans_answered = [r for r in ans if not r["refused"]]
    unans = [r for r in ok if not r["answerable"]]
    judged = [r for r in ans_answered if r.get("judge")]

    def crit(name: str) -> float | None:
        return r4(mean(r["judge"][name]["score"] for r in judged)) if judged else None

    answer_cost = sum((r["usage"] or {}).get("cost_usd") or 0.0 for r in ok)
    judge_cost = sum((r["judge_usage"] or {}).get("cost_usd") or 0.0 for r in ok)
    tokens_in = sum((r["usage"] or {}).get("input_tokens", 0) for r in ok)
    tokens_out = sum((r["usage"] or {}).get("output_tokens", 0) for r in ok)
    summary = {
        "retriever": retriever,
        "top_k": s.top_k,
        "answer_model": s.claude_model,
        "judge_model": s.judge_model,
        "effort": s.claude_effort,
        "n": len(ok),
        "errors": errors,
        "answerable": {
            "n": len(ans),
            "false_refusal_rate": r4(mean(1.0 if r["refused"] else 0.0 for r in ans)) if ans else None,
            "citation_within_context_rate": r4(mean(r["citation"]["within_context"] for r in ans_answered))
            if ans_answered
            else None,
            "citation_cites_gold_rate": r4(mean(r["citation"]["cites_gold"] for r in ans_answered))
            if ans_answered
            else None,
            "citation_correct_rate": r4(mean(r["citation"]["correct"] for r in ans_answered)) if ans_answered else None,
            "key_fact_coverage": r4(mean(r["key_fact_coverage"] for r in ans)) if ans else None,
            "judge": {
                "n": len(judged),
                "faithfulness": crit("faithfulness"),
                "correctness": crit("correctness"),
                "completeness": crit("completeness"),
                "citation_quality": crit("citation_quality"),
                "pass_rate_answered": r4(mean(1.0 if r["judge"]["overall_pass"] else 0.0 for r in judged))
                if judged
                else None,
                "pass_rate_all_answerable": r4(sum(1 for r in judged if r["judge"]["overall_pass"]) / len(ans))
                if ans
                else None,
            },
        },
        "unanswerable": {
            "n": len(unans),
            "refusal_accuracy": r4(mean(1.0 if r["refused"] else 0.0 for r in unans)) if unans else None,
            "refusal_reasons": {
                k: sum(1 for r in unans if r["refusal_reason"] == k) for k in {r["refusal_reason"] for r in unans}
            },
        },
        "fallback_used": sum(1 for r in ok if r.get("fallback_used")),
        "latency_ms_mean": round(mean(r["latency_ms"] for r in ok), 1) if ok else None,
        "cost_usd": {
            "answers": round(answer_cost, 4),
            "judge": round(judge_cost, 4),
            "total": round(answer_cost + judge_cost, 4),
        },
        "tokens": {"answer_input": tokens_in, "answer_output": tokens_out},
    }
    return summary, rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def fmt(x, nd=3) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def bold_best(values: dict[str, float | None], key: str, higher: bool = True) -> set[str]:
    vals = {r: v[key] for r, v in values.items() if v.get(key) is not None}
    if not vals:
        return set()
    best = max(vals.values()) if higher else min(vals.values())
    return {r for r, v in vals.items() if v == best}


def render_report(ret: dict, gen: dict | None, meta: dict, analysis: str, rr: dict | None = None) -> str:
    o = ret["overall"]
    retrievers = list(o.keys())
    L = []
    L.append("# Eval report: FastAPI docs RAG\n")
    L.append(
        f"_Generated by `evals/run.py` on {meta['date']} (repo rev `{meta['rev']}`). "
        "Every number below was produced by that run; nothing is hand-entered._\n"
    )

    L.append("## Setup\n")
    L.append("| | |\n|---|---|")
    L.append(
        f"| Corpus | FastAPI docs `{meta['corpus_tag']}` (commit `{meta['corpus_commit'][:12]}`), {meta['pages']} pages, MIT |"
    )
    L.append(
        f"| Chunking | heading-aware Markdown, ≤{meta['chunk_max']} tokens, {meta['chunk_overlap']}-token overlap within a section → {meta['chunks']} chunks |"
    )
    L.append(
        f"| Embeddings | `{meta['embed_model']}` (query instruction prefix, cosine), Postgres 16 + pgvector HNSW |"
    )
    L.append("| BM25 | `bm25s` Okapi BM25 (k1=1.5, b=0.75), English stopwords + Snowball stemming, same chunk text |")
    L.append(f"| Hybrid | Reciprocal Rank Fusion (k={meta['rrf_k']}) over the top-50 of BM25 and vector |")
    L.append(
        f"| Reranker | `{meta['rerank_model']}` cross-encoder over the top-{meta['candidate_k']} hybrid candidates |"
    )
    L.append(
        f"| Golden set | {meta['n_items']} questions: "
        + ", ".join(f"{v} {k}" for k, v in meta["type_counts"].items())
        + f" ([evals/golden.jsonl](evals/golden.jsonl), sha256 `{meta['golden_hash']}`); 1-3 gold chunks each, completed by pooling (see below) |"
    )
    L.append(
        "| Metrics | Hit@k = any gold chunk in top-k; MRR@10; Recall@5 = share of gold chunks in top-5 (answerable questions only) |"
    )
    L.append(
        f"| Hardware | {meta['hardware']}; latency = wall-clock per query incl. query embedding (and reranking), after warm-up |"
    )
    L.append("")

    L.append(f"## Retrieval ablation ({meta['n_answerable']} answerable questions)\n")
    L.append("| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Recall@5 | Latency mean | Latency p95 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    best = {k: bold_best(o, k) for k in ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "recall@5"]}
    for r in retrievers:
        m = o[r]

        def cell(k, f, m=m, r=r):
            s = f(m[k])
            return f"**{s}**" if r in best[k] else s

        L.append(
            f"| {label(r)} | {cell('hit@1', pct)} | {cell('hit@3', pct)} | {cell('hit@5', pct)} | "
            f"{cell('hit@10', pct)} | {cell('mrr@10', fmt)} | {cell('recall@5', pct)} | "
            f"{m['latency_ms_mean']:.0f} ms | {m['latency_ms_p95']:.0f} ms |"
        )
    L.append("")
    ci = "; ".join(
        f"{short(r)} Hit@5 {pct(o[r]['hit@5_ci95'][0])}–{pct(o[r]['hit@5_ci95'][1])}, MRR {fmt(o[r]['mrr@10_ci95'][0], 2)}–{fmt(o[r]['mrr@10_ci95'][1], 2)}"
        for r in retrievers
        if o[r].get("hit@5_ci95")
    )
    L.append(
        f"95% bootstrap CIs (1,000 resamples of the {meta['n_answerable']} questions): {ci}. "
        f"One question is worth {100 / meta['n_answerable']:.1f} points of Hit@k, so differences of a few points are within noise.\n"
    )
    L.append(f"The app's default retriever is `{meta['default_retriever']}` (`RETRIEVER` in `.env`).\n")
    if ret.get("rerank_delta"):
        d = ret["rerank_delta"]
        L.append(
            f"Reranking moved the gold chunk into the top-5 for {len(d['helped_at_5'])} question(s) "
            f"({', '.join(d['helped_at_5']) or '–'}) and out of it for {len(d['hurt_at_5'])} "
            f"({', '.join(d['hurt_at_5']) or '–'}), compared with plain hybrid.\n"
        )

    L.append("### By question type (Hit@5 / MRR@10, bold = best Hit@5)\n")
    L.append("| Type | n | " + " | ".join(short(r) for r in retrievers) + " |")
    L.append("|---|---:|" + "---:|" * len(retrievers))
    for t, d in ret["by_type"].items():
        cells = []
        b = {r for r in retrievers if d[r]["hit@5"] == max(d[x]["hit@5"] for x in retrievers)}
        for r in retrievers:
            c = f"{pct(d[r]['hit@5'])} / {fmt(d[r]['mrr@10'], 2)}"
            cells.append(f"**{c}**" if r in b else c)
        L.append(f"| {t} | {d['n']} | " + " | ".join(cells) + " |")
    L.append("")

    g = ret["refusal_gate"]
    mc, ts = g["max_cosine"], g["top_score"]
    L.append("## Refusal gate signal (offline)\n")
    L.append(
        f"Before calling Claude, the app refuses if the best cosine similarity between the question and its top-{g['top_k']} "
        f"`{g['retriever']}` chunks is below `MIN_RELEVANCE={g['threshold']}`. How well does that signal separate the "
        f"{g['n_answerable']} answerable from the {g['n_unanswerable']} unanswerable questions?\n"
    )
    L.append("| Signal | Answerable (min / mean) | Unanswerable (mean / max) | AUROC |")
    L.append("|---|---:|---:|---:|")
    L.append(
        f"| Max cosine (bge-small) | {fmt(mc['answerable']['min'])} / {fmt(mc['answerable']['mean'])} | "
        f"{fmt(mc['unanswerable']['mean'])} / {fmt(mc['unanswerable']['max'])} | {fmt(mc['auroc'], 2)} |"
    )
    L.append(
        f"| {ts['label']} | {fmt(ts['answerable']['min'], 2)} / {fmt(ts['answerable']['mean'], 2)} | "
        f"{fmt(ts['unanswerable']['mean'], 2)} / {fmt(ts['unanswerable']['max'], 2)} | {fmt(ts['auroc'], 2)} |"
    )
    L.append("")
    L.append(
        f"At the configured threshold the gate blocks {mc['answerable_blocked_at_threshold']} answerable question(s) "
        f"and catches {mc['unanswerable_caught_at_threshold']}/{g['n_unanswerable']} of the in-domain unanswerable ones "
        "before any LLM call. Threshold sweep (answerable blocked / unanswerable caught):\n"
    )
    L.append("| MIN_RELEVANCE | " + " | ".join(f"{x['threshold']:.2f}" for x in g["sweep"]) + " |")
    L.append("|---|" + "---:|" * len(g["sweep"]))
    L.append(
        "| blocked / caught | "
        + " | ".join(
            f"{x['answerable_blocked']}/{g['n_answerable']} · {x['unanswerable_caught']}/{g['n_unanswerable']}"
            for x in g["sweep"]
        )
        + " |"
    )
    L.append("")
    probes = ", ".join(f"“{p['question']}” {fmt(p['max_cosine'])}" for p in g.get("off_topic_probes", []))
    L.append(
        f"Clearly off-topic probes score far lower ({probes}), so the cheap gate is a garbage filter; refusing "
        "*plausible* FastAPI questions the docs don't cover is left to the prompt + citation gate "
        "(measured in the generation section).\n"
    )

    if rr:
        stale = rr["meta"].get("golden_hash") != meta["golden_hash"]
        L.append("### Reranker comparison\n")
        L.append(
            f"Separate run on {rr['meta']['date']} (`make eval-rerankers`, rev `{rr['meta']['rev']}`)"
            + (" — **stale: the golden set changed since, re-run it**" if stale else ", same golden set")
            + ". Same hybrid candidates (top-"
            + str(meta["candidate_k"])
            + "), different cross-encoders. Latencies come from that run, with the 1.1 GB model loaded "
            "on an 8 GB laptop, so they aren't comparable with the main table:\n"
        )
        L.append("| Retriever | Hit@1 | Hit@5 | MRR@10 | Latency mean |")
        L.append("|---|---:|---:|---:|---:|")
        for r, m in rr["overall"].items():
            name = f"Hybrid + `{meta['rerank_model']}` rerank" if r == "hybrid_rerank" else label(r)
            L.append(
                f"| {name} | {pct(m['hit@1'])} | {pct(m['hit@5'])} | {fmt(m['mrr@10'])} | {m['latency_ms_mean']:.0f} ms |"
            )
        L.append("")

    L.append("## Failure analysis\n")
    L.append(analysis.strip() + "\n" if analysis.strip() else "_No analysis notes yet._\n")

    L.append("## Generation metrics (Claude)\n")
    if gen is None:
        L.append(
            "**Pending — run `make eval-full` with `ANTHROPIC_API_KEY` set.** No generation numbers are reported "
            "until that run exists. It will measure, on the same golden set with the default retriever "
            f"(`{meta['default_retriever']}`, top-5 chunks as Citations documents):\n"
        )
        L.append("| Metric | Definition | Value |\n|---|---|---|")
        for name, desc in [
            ("Citation correctness", "every cited chunk was in the provided context AND ≥1 cited chunk is gold"),
            ("Key-fact coverage", "share of the golden key facts present in the answer"),
            ("Faithfulness (LLM judge)", f"1-5, `{meta['judge_model']}` rubric with structured output"),
            ("Answer correctness (LLM judge)", "1-5 vs. the reference answer"),
            ("Refusal accuracy", "unanswerable questions answered with the refusal"),
            ("False-refusal rate", "answerable questions that were refused"),
            ("Run cost", "answer + judge tokens × list prices"),
        ]:
            L.append(f"| {name} | {desc} | pending |")
        L.append("")
    else:
        a, u, j = gen["answerable"], gen["unanswerable"], gen["answerable"]["judge"]
        L.append(
            f"Answer model `{gen['answer_model']}` (effort `{gen['effort']}`), judge `{gen['judge_model']}`, "
            f"retriever `{gen['retriever']}`, top-{gen['top_k']} chunks, {gen['n']} questions, run on {gen.get('date', meta['date'])}.\n"
        )
        L.append("| Metric | Value |\n|---|---:|")
        L.append(f"| Citation correctness (in-context AND cites gold) | {pct(a['citation_correct_rate'])} |")
        L.append(f"| Citations within provided context | {pct(a['citation_within_context_rate'])} |")
        L.append(f"| Answers citing ≥1 gold chunk | {pct(a['citation_cites_gold_rate'])} |")
        L.append(f"| Key-fact coverage | {pct(a['key_fact_coverage'])} |")
        L.append(f"| Judge: faithfulness (1-5) | {fmt(j['faithfulness'], 2)} |")
        L.append(f"| Judge: correctness (1-5) | {fmt(j['correctness'], 2)} |")
        L.append(f"| Judge: completeness (1-5) | {fmt(j['completeness'], 2)} |")
        L.append(f"| Judge: citation quality (1-5) | {fmt(j['citation_quality'], 2)} |")
        L.append(
            f"| Judge pass rate (answered / all answerable) | {pct(j['pass_rate_answered'])} / {pct(j['pass_rate_all_answerable'])} |"
        )
        L.append(f"| Refusal accuracy on unanswerable ({u['n']}) | {pct(u['refusal_accuracy'])} |")
        L.append(f"| False-refusal rate on answerable ({a['n']}) | {pct(a['false_refusal_rate'])} |")
        L.append(f"| Mean end-to-end latency | {gen['latency_ms_mean']:.0f} ms |")
        L.append(
            f"| Run cost (answers + judge) | ${gen['cost_usd']['total']:.4f} (${gen['cost_usd']['answers']:.4f} + ${gen['cost_usd']['judge']:.4f}) |"
        )
        L.append("")

    L.append("## Reproduce\n")
    L.append(
        "```bash\nmake up && make ingest   # Postgres + pgvector, embed the corpus\nmake eval                # this report's retrieval section (offline, ~1 min)\n"
        "make eval-full           # + generation metrics (needs ANTHROPIC_API_KEY in .env)\n```\n"
    )
    L.append(
        "Raw per-question results go to `evals/runs/<timestamp>/` (git-ignored); the aggregated numbers are in "
        "[`evals/results/`](evals/results/).\n"
    )
    return "\n".join(L)


def build_meta(engine, items: list[GoldenItem]) -> dict:
    s: Settings = engine.settings
    manifest = load_manifest(s.corpus_dir)
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it.type] += 1
    device = getattr(engine.embedder, "device", "cpu")
    return {
        "date": dt.date.today().isoformat(),
        "rev": git_rev(),
        "corpus_tag": manifest["tag"],
        "corpus_commit": manifest["commit"],
        "pages": len(manifest["pages"]),
        "chunks": len(engine.store.chunks()),
        "chunk_max": s.chunk_max_tokens,
        "chunk_overlap": s.chunk_overlap_tokens,
        "embed_model": s.embed_model,
        "rerank_model": s.rerank_model,
        "rrf_k": s.rrf_k,
        "candidate_k": s.candidate_k,
        "judge_model": s.judge_model,
        "default_retriever": s.default_retriever,
        "golden_hash": golden_hash(),
        "n_items": len(items),
        "n_answerable": sum(1 for it in items if it.answerable),
        "type_counts": dict(counts),
        "hardware": f"{platform.machine()} {platform.system()} ({device}), Python {platform.python_version()}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run retrieval (and optionally generation) evals")
    ap.add_argument("--full", action="store_true", help="also run generation metrics with Claude (needs a key)")
    ap.add_argument("--retrievers", default=",".join(RETRIEVERS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N golden items (smoke runs)")
    ap.add_argument("--no-report", action="store_true", help="don't rewrite EVAL_REPORT.md / evals/results")
    ap.add_argument(
        "--extra-rerankers",
        default="",
        help="comma-separated cross-encoder models to compare "
        "against the default reranker (writes evals/results/reranker_summary.json)",
    )
    args = ap.parse_args()

    s = get_settings()
    if args.full and not s.llm_enabled:
        print(
            f"ERROR: {NO_KEY_MESSAGE}\n`make eval-full` needs ANTHROPIC_API_KEY; `make eval` runs offline.",
            file=sys.stderr,
        )
        return 2

    from ragkit.engine import build_engine
    from ragkit.store import NotIngestedError

    try:
        engine = build_engine(s, with_reranker=True, with_llm=args.full)
    except NotIngestedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # psycopg OperationalError etc.
        print(f"ERROR: cannot open the index ({e}). Is the database up (`make up`)?", file=sys.stderr)
        return 2

    items = load_golden()
    if args.limit:
        items = items[: args.limit]
    retrievers = [r for r in args.retrievers.split(",") if r]
    extras = [m for m in args.extra_rerankers.split(",") if m]
    if extras:
        from ragkit.retrieval import CrossEncoderReranker, RerankRetriever

        retrievers = ["hybrid", "hybrid_rerank"]
        for model in extras:
            name = f"hybrid_rerank:{model}"
            engine.retrievers[name] = RerankRetriever(
                engine.retrievers["hybrid"], CrossEncoderReranker(model, s.device), engine.store, s.candidate_k
            )
            retrievers.append(name)
    gate_retriever = s.default_retriever if s.default_retriever in retrievers else retrievers[-1]
    run_dir = RUNS_DIR / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = build_meta(engine, items)

    print(f"Retrieval eval: {len(items)} questions x {len(retrievers)} retrievers (k={EVAL_K})")
    t0 = time.perf_counter()
    ret_summary, ret_rows = run_retrieval(engine, items, retrievers, gate_retriever)
    ret_summary["meta"] = meta
    (run_dir / "retrieval.json").write_text(json.dumps({"summary": ret_summary, "rows": ret_rows}, indent=1))
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print(f"\n{'retriever':<15} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'hit@10':>7} {'mrr@10':>7} {'lat ms':>7}")
    for r, m in ret_summary["overall"].items():
        print(
            f"{r:<15} {m['hit@1']:>7.3f} {m['hit@3']:>7.3f} {m['hit@5']:>7.3f} {m['hit@10']:>7.3f} "
            f"{m['mrr@10']:>7.3f} {m['latency_ms_mean']:>7.1f}"
        )

    rr_path = RESULTS_DIR / "reranker_summary.json"
    if extras and not args.limit:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        rr = {
            "meta": {k: meta[k] for k in ("date", "rev", "golden_hash", "candidate_k")},
            "overall": {
                r: {k: v for k, v in m.items() if not k.endswith("_ci95")} for r, m in ret_summary["overall"].items()
            },
        }
        rr_path.write_text(json.dumps(rr, indent=1) + "\n")
        print(f"Wrote {rr_path.relative_to(ROOT)}")
        ret_path = RESULTS_DIR / "retrieval_summary.json"
        if ret_path.exists():  # re-render the report from the main ablation + this comparison
            main_ret = json.loads(ret_path.read_text())
            gen_old = (
                json.loads((RESULTS_DIR / "generation_summary.json").read_text())
                if (RESULTS_DIR / "generation_summary.json").exists()
                else None
            )
            analysis = load_analysis()
            REPORT_PATH.write_text(render_report(main_ret, gen_old, main_ret["meta"], analysis, rr), encoding="utf-8")
            print(f"Re-rendered {REPORT_PATH.relative_to(ROOT)}")
        return 0

    gen_summary = None
    gen_path = RESULTS_DIR / "generation_summary.json"
    if args.full:
        print(f"\nGeneration eval with {s.claude_model} (judge {s.judge_model}) ...")
        gen_summary, gen_rows = run_generation(engine, items, s.default_retriever)
        gen_summary["date"] = meta["date"]
        gen_summary["rev"] = meta["rev"]
        (run_dir / "generation.json").write_text(json.dumps({"summary": gen_summary, "rows": gen_rows}, indent=1))
        print(json.dumps({k: gen_summary[k] for k in ("answerable", "unanswerable", "cost_usd")}, indent=1))
    elif gen_path.exists() and not args.limit:
        gen_summary = json.loads(gen_path.read_text())  # keep the last real generation run in the report

    if args.no_report or args.limit:
        print(f"\nRun artifacts: {run_dir.relative_to(ROOT)} (report not rewritten)")
        return 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_summary.json").write_text(json.dumps(ret_summary, indent=1) + "\n")
    if args.full and gen_summary is not None:
        gen_path.write_text(json.dumps(gen_summary, indent=1) + "\n")
    analysis = load_analysis()
    rr = json.loads(rr_path.read_text()) if rr_path.exists() else None
    REPORT_PATH.write_text(render_report(ret_summary, gen_summary, meta, analysis, rr), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)} and evals/results/; run artifacts in {run_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
