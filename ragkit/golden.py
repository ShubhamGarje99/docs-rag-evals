"""Golden-set loading and text normalisation shared by the eval scripts and tests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT

GOLDEN_PATH = ROOT / "evals" / "golden.jsonl"
QUESTION_TYPES = ("factual", "how-to", "multi-chunk", "paraphrased", "unanswerable")

STOPWORDS = set(
    """a an and are as at be but by can could do does for from had has have how i if in into is it its
    me my of on or our should so that the their them then there these they this to up us was we what
    when where which while who why will with would you your i'm i'd don't doesn't can't won't it's what's
    via just like e g eg one some any all also than too very only own same other such no not""".split()
)


@dataclass
class GoldenItem:
    id: str
    type: str
    difficulty: str
    question: str
    reference_answer: str
    gold_chunk_ids: list[str]
    key_facts: list[str]
    absent_terms: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.gold_chunk_ids)


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        d = json.loads(line)
        try:
            items.append(GoldenItem(**d))
        except TypeError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
    return items


def normalize(text: str) -> str:
    """Lowercase, drop Markdown emphasis/backticks, unify quotes, collapse whitespace."""
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[*`]", "", text)
    return " ".join(text.split())


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{2,}", normalize(text)) if t not in STOPWORDS}


def lexical_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that literally occur in the passage."""
    q = content_tokens(question)
    if not q:
        return 0.0
    return len(q & content_tokens(passage)) / len(q)
