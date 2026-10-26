"""Propose golden-set candidates with Claude (needs ANTHROPIC_API_KEY).

For a seeded, page-stratified sample of chunks, Claude writes questions a real
user might ask *without* reusing the chunk's wording, plus a reference answer
and key facts copied verbatim from the chunk. Candidates are filtered
automatically (key facts must occur in the chunk, lexical overlap must be
below a ceiling so BM25 doesn't get a free win) and written for HUMAN REVIEW to
``evals/runs/golden_candidates-<timestamp>.jsonl``. Nothing is added to
``evals/golden.jsonl`` automatically.

The committed golden set was written by hand without a key (same procedure,
then checked with ``evals/check_golden.py``); this script is how you would grow it.

    uv run python scripts/generate_golden.py --n-chunks 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragkit.chunking import Chunk, chunk_corpus  # noqa: E402
from ragkit.config import NO_KEY_MESSAGE, get_settings  # noqa: E402
from ragkit.golden import lexical_overlap, normalize  # noqa: E402
from ragkit.llm import FALLBACK_BETA, make_client, supports_server_fallback, usage_from_response  # noqa: E402


class Candidate(BaseModel):
    type: Literal["factual", "how-to", "paraphrased"]
    question: str = Field(description="What a developer would type into a docs chatbot.")
    reference_answer: str = Field(description="1-3 sentences, answerable from the excerpt alone.")
    key_facts: list[str] = Field(description="1-3 short phrases copied VERBATIM from the excerpt.")


class Candidates(BaseModel):
    candidates: list[Candidate]


PROMPT = """You are building an evaluation set for a retrieval system over the FastAPI documentation.

<excerpt id="{chunk_id}" section="{breadcrumb}">
{text}
</excerpt>

Write {n} different questions that this excerpt fully answers, as a real developer would ask them:
- one "factual" (a specific fact, value or name), one "how-to" (a task), and, if possible, one "paraphrased" question that describes the problem in the developer's own words and avoids the excerpt's distinctive terms (no API names, no headings).
- Questions must be answerable from this excerpt alone and must not mention "the excerpt" or "the docs section".
- key_facts: 1-3 short phrases copied exactly, character for character, from the excerpt (identifiers, values, short terms) that a correct answer must contain.
If the excerpt has nothing worth asking about (navigation text, a lone heading), return an empty list."""


def sample_chunks(chunks: list[Chunk], n: int, seed: int) -> list[Chunk]:
    rng = random.Random(seed)
    by_page: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.n_tokens >= 80:  # skip stubs
            by_page[c.doc_path].append(c)
    pages = sorted(by_page)
    rng.shuffle(pages)
    return [rng.choice(by_page[p]) for p in pages[:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-chunks", type=int, default=20)
    ap.add_argument("--per-chunk", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-overlap", type=float, default=0.7)
    args = ap.parse_args()

    s = get_settings()
    if not s.llm_enabled:
        print(f"ERROR: {NO_KEY_MESSAGE}", file=sys.stderr)
        return 2
    client = make_client(s)
    chunks = sample_chunks(
        chunk_corpus(s.corpus_dir, s.chunk_max_tokens, s.chunk_overlap_tokens), args.n_chunks, args.seed
    )

    out_dir = ROOT / "evals" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"golden_candidates-{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    kept = dropped = 0
    cost = 0.0
    with out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=1):
            kwargs = dict(
                model=s.claude_model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            chunk_id=c.chunk_id, breadcrumb=c.breadcrumb, text=c.text, n=args.per_chunk
                        ),
                    }
                ],
                output_config={"effort": "medium"},
                output_format=Candidates,
            )
            if s.claude_fallbacks and supports_server_fallback(s.claude_model):
                resp = client.beta.messages.parse(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                resp = client.messages.parse(**kwargs)
            cost += usage_from_response(resp, s.claude_model).cost_usd or 0.0
            if resp.stop_reason == "refusal" or resp.parsed_output is None:
                print(f"[{i}/{len(chunks)}] {c.chunk_id}: no output ({resp.stop_reason})")
                continue
            text = normalize(c.search_text)
            for cand in resp.parsed_output.candidates:
                facts_ok = cand.key_facts and all(normalize(k) in text for k in cand.key_facts)
                overlap = lexical_overlap(cand.question, c.search_text)
                if not facts_ok or overlap > args.max_overlap:
                    dropped += 1
                    continue
                kept += 1
                f.write(
                    json.dumps(
                        {
                            "id": f"cand-{kept:03d}",
                            "type": cand.type,
                            "difficulty": "unrated",
                            "question": cand.question,
                            "reference_answer": cand.reference_answer,
                            "gold_chunk_ids": [c.chunk_id],
                            "key_facts": cand.key_facts,
                            "_lexical_overlap": round(overlap, 2),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"[{i}/{len(chunks)}] {c.chunk_id}: kept {kept}, dropped {dropped}")
    print(f"\nWrote {kept} candidates ({dropped} dropped by filters) to {out.relative_to(ROOT)}; cost ≈ ${cost:.4f}")
    print(
        "Review them by hand, add any extra gold chunks that also answer the question, "
        "then append the good ones to evals/golden.jsonl and run `make golden-check`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
