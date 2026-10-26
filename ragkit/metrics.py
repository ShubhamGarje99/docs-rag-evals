"""Retrieval and generation metrics (pure functions, unit-tested)."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence


def first_relevant_rank(ranked: Sequence[str], gold: Iterable[str], k: int | None = None) -> int | None:
    """1-based rank of the first gold item within the top-k, or None."""
    gold = set(gold)
    for i, doc in enumerate(ranked[:k] if k else ranked, start=1):
        if doc in gold:
            return i
    return None


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if any gold item is in the top-k (a.k.a. success@k / hit rate)."""
    return 1.0 if first_relevant_rank(ranked, gold, k) is not None else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str], k: int = 10) -> float:
    """1/rank of the first gold item within the top-k, else 0 (averaged -> MRR@k)."""
    r = first_relevant_rank(ranked, gold, k)
    return 1.0 / r if r else 0.0


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Share of gold items found in the top-k (matters for multi-chunk questions)."""
    gold = set(gold)
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in [0, 100])."""
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1))
    return s[idx]


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Probability a random positive scores higher than a random negative (ties count 0.5)."""
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def key_fact_coverage(answer: str, key_facts: Sequence[str]) -> float:
    """Share of key facts that literally appear (normalised) in the answer."""
    from .golden import normalize

    if not key_facts:
        return float("nan")
    a = normalize(answer)
    return sum(1 for f in key_facts if normalize(f) in a) / len(key_facts)


def citation_checks(cited: Sequence[str], context: Sequence[str], gold: Sequence[str]) -> dict[str, bool]:
    """Citation correctness for one answer.

    * ``within_context``: every cited chunk was actually provided to the model;
    * ``cites_gold``: at least one cited chunk is a gold chunk;
    * ``correct``: both of the above (and at least one citation).
    """
    cited_set = set(cited)
    within = bool(cited_set) and cited_set <= set(context)
    gold_hit = bool(cited_set & set(gold))
    return {"within_context": within, "cites_gold": gold_hit, "correct": within and gold_hit}
