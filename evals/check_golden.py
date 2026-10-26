"""Sanity-check evals/golden.jsonl against the chunked corpus.

Checks (any ERROR exits non-zero):
  * schema: required fields, known type, 1-3 gold chunks for answerable items,
    none for unanswerable ones;
  * every gold chunk id exists in the current chunking of corpus/;
  * every key fact appears (normalised) in at least one of the item's gold chunks;
  * no duplicate / near-duplicate questions;
  * unanswerable items: none of their `absent_terms` occurs anywhere in the corpus.
Reports (informational): question <-> gold-chunk lexical overlap per type, so we
can see whether the set is biased towards keyword matching (which would hand
BM25 an unfair win).

    uv run python evals/check_golden.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragkit.chunking import chunk_corpus  # noqa: E402
from ragkit.config import get_settings  # noqa: E402
from ragkit.golden import QUESTION_TYPES, content_tokens, lexical_overlap, load_golden, normalize  # noqa: E402

HIGH_OVERLAP = 0.85


def main() -> int:
    s = get_settings()
    chunks = {c.chunk_id: c for c in chunk_corpus(s.corpus_dir, s.chunk_max_tokens, s.chunk_overlap_tokens)}
    corpus_norm = normalize("\n".join(c.text for c in chunks.values()))
    items = load_golden()
    errors: list[str] = []
    warnings: list[str] = []

    seen_ids: set[str] = set()
    for it in items:
        if it.id in seen_ids:
            errors.append(f"{it.id}: duplicate id")
        seen_ids.add(it.id)
        if it.type not in QUESTION_TYPES:
            errors.append(f"{it.id}: unknown type {it.type!r}")
        if it.type == "unanswerable":
            if it.gold_chunk_ids or it.key_facts:
                errors.append(f"{it.id}: unanswerable items must have empty gold_chunk_ids/key_facts")
            for term in it.absent_terms:
                if normalize(term) in corpus_norm:
                    errors.append(f"{it.id}: 'absent' term {term!r} occurs in the corpus")
            continue
        if not 1 <= len(it.gold_chunk_ids) <= 3:
            errors.append(f"{it.id}: answerable items need 1-3 gold chunks")
        if not it.key_facts:
            errors.append(f"{it.id}: no key facts")
        missing = [cid for cid in it.gold_chunk_ids if cid not in chunks]
        for cid in missing:
            errors.append(f"{it.id}: gold chunk {cid!r} not found")
        gold_text = normalize("\n".join(chunks[c].search_text for c in it.gold_chunk_ids if c in chunks))
        for fact in it.key_facts:
            if normalize(fact) not in gold_text:
                errors.append(f"{it.id}: key fact {fact!r} not found in its gold chunks")
        if it.type == "multi-chunk" and len(it.gold_chunk_ids) < 2:
            warnings.append(f"{it.id}: multi-chunk item with a single gold chunk")

    # duplicate / near-duplicate questions
    qs = [(it.id, content_tokens(it.question), normalize(it.question)) for it in items]
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            if qs[i][2] == qs[j][2]:
                errors.append(f"{qs[i][0]} / {qs[j][0]}: duplicate question")
                continue
            a, b = qs[i][1], qs[j][1]
            jac = len(a & b) / max(1, len(a | b))
            if jac >= 0.6:
                warnings.append(f"{qs[i][0]} / {qs[j][0]}: near-duplicate questions (Jaccard {jac:.2f})")

    # lexical overlap report
    by_type: dict[str, list[float]] = defaultdict(list)
    high = []
    for it in items:
        if not it.answerable or any(c not in chunks for c in it.gold_chunk_ids):
            continue
        ov = max(lexical_overlap(it.question, chunks[c].search_text) for c in it.gold_chunk_ids)
        by_type[it.type].append(ov)
        if ov >= HIGH_OVERLAP:
            high.append((it.id, ov))

    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it.type] += 1
    print(f"Golden set: {len(items)} items  " + "  ".join(f"{t}={counts[t]}" for t in QUESTION_TYPES))
    print(f"Corpus: {len(chunks)} chunks\n")
    print("Question <-> best gold chunk lexical overlap (share of question content words found in the chunk):")
    print(f"  {'type':<13} {'n':>3} {'mean':>6} {'min':>6} {'max':>6}")
    all_ov = []
    for t in QUESTION_TYPES:
        v = by_type.get(t)
        if not v:
            continue
        all_ov += v
        print(f"  {t:<13} {len(v):>3} {statistics.mean(v):>6.2f} {min(v):>6.2f} {max(v):>6.2f}")
    if all_ov:
        print(f"  {'all':<13} {len(all_ov):>3} {statistics.mean(all_ov):>6.2f} {min(all_ov):>6.2f} {max(all_ov):>6.2f}")
    if high:
        warnings.append(
            f"high lexical overlap (>= {HIGH_OVERLAP:.2f}): " + ", ".join(f"{i} ({o:.2f})" for i, o in high)
        )
    print()
    for w in warnings:
        print(f"WARNING {w}")
    for e in errors:
        print(f"ERROR   {e}")
    print(f"\n{'FAIL' if errors else 'OK'}: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
