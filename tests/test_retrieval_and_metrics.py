from __future__ import annotations

import math

import pytest

from ragkit.metrics import (
    auroc,
    citation_checks,
    first_relevant_rank,
    hit_at_k,
    key_fact_coverage,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from ragkit.retrieval import rrf_fuse

# --------------------------------------------------------------------------- RRF


def test_rrf_scores_match_the_formula():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_order_and_deterministic_ties():
    order = [d for d, _ in rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)]
    assert order == ["a", "c", "b", "d"]
    # x and y have identical scores; tie broken by best rank, then id
    tied = [d for d, _ in rrf_fuse([["x", "y"], ["y", "x"]], k=60)]
    assert tied == ["x", "y"]


def test_rrf_small_k_rewards_top_ranks_more():
    big = dict(rrf_fuse([["a", "b"]], k=60))
    small = dict(rrf_fuse([["a", "b"]], k=1))
    assert small["a"] / small["b"] > big["a"] / big["b"]


def test_rrf_empty():
    assert rrf_fuse([[], []]) == []


# --------------------------------------------------------------------------- metrics

RANKED = ["d1", "d2", "d3", "d4", "d5"]


def test_hit_at_k_and_rank():
    assert first_relevant_rank(RANKED, {"d3"}) == 3
    assert hit_at_k(RANKED, {"d3"}, 1) == 0.0
    assert hit_at_k(RANKED, {"d3"}, 3) == 1.0
    assert hit_at_k(RANKED, {"zz"}, 5) == 0.0


def test_mrr_uses_first_relevant_within_cutoff():
    assert reciprocal_rank(RANKED, {"d2", "d4"}) == 0.5
    assert reciprocal_rank(RANKED, {"d5"}, k=4) == 0.0
    mrr = (reciprocal_rank(RANKED, {"d1"}) + reciprocal_rank(RANKED, {"d4"}) + reciprocal_rank(RANKED, {"x"})) / 3
    assert mrr == pytest.approx((1 + 0.25 + 0) / 3)


def test_recall_at_k_for_multi_chunk_gold():
    assert recall_at_k(RANKED, {"d1", "d9"}, 5) == 0.5
    assert recall_at_k(RANKED, {"d1", "d2"}, 1) == 0.5
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_auroc_and_percentile():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5
    assert math.isnan(auroc([], [1.0]))
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([5], 95) == 5


def test_key_fact_coverage_normalises_markdown():
    ans = 'Use the **exec** form: `CMD ["fastapi", "run"]` and set `max_age` to 600.'
    assert key_fact_coverage(ans, ["exec form", "max_age", "600"]) == 1.0
    assert key_fact_coverage(ans, ["exec form", "Celery"]) == 0.5


def test_citation_checks():
    ctx = ["a", "b", "c"]
    assert citation_checks(["a"], ctx, ["a"]) == {"within_context": True, "cites_gold": True, "correct": True}
    assert citation_checks(["b"], ctx, ["a"])["correct"] is False
    assert citation_checks(["z"], ctx, ["z"])["within_context"] is False
    assert citation_checks([], ctx, ["a"])["correct"] is False


# --------------------------------------------------------------------------- retrievers on the fixture corpus


def test_retrievers_on_fixture(engine):
    for name in ("vector", "bm25", "hybrid"):
        res = engine.search("How do I configure CORSMiddleware allowed origins?", name, 3)
        assert res.hits, name
        assert res.hits[0].chunk_id.startswith("tutorial/cors.md#"), (name, res.hits[0].chunk_id)
        assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    hybrid = engine.search("Celery heavy background computation", "hybrid", 3)
    assert hybrid.hits[0].detail.get("bm25_rank") == 1


def test_rerank_falls_back_to_hybrid_without_reranker(engine):
    assert engine.resolve_retriever("hybrid_rerank") == "hybrid"
    with pytest.raises(KeyError):
        engine.resolve_retriever("nope")
