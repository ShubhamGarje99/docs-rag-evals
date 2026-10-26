"""LLM-as-judge rubric for generated answers (Claude Haiku 5.5, structured outputs).

Uses ``client.messages.parse(..., output_format=JudgeVerdict)`` so the verdict is
schema-validated JSON. No citations here (citations and structured outputs are
mutually exclusive), no sampling parameters, and no ``fallbacks`` (Haiku 5.5 has
no server-side fallback).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from .chunking import Chunk
from .config import Settings
from .llm import Usage, make_client, usage_from_response

Score = Literal[1, 2, 3, 4, 5]


class Criterion(BaseModel):
    rationale: str = Field(description="One or two sentences justifying the score. Quote the problem if any.")
    score: Score


class JudgeVerdict(BaseModel):
    faithfulness: Criterion = Field(
        description="Is every claim in the answer supported by the provided context excerpts? "
        "5 = fully supported, 3 = minor unsupported details, 1 = mostly unsupported or contradicted."
    )
    correctness: Criterion = Field(
        description="Does the answer agree with the reference answer on the facts that matter? "
        "5 = fully consistent, 1 = wrong or contradicts it."
    )
    completeness: Criterion = Field(
        description="Does the answer cover the key facts the question needs? 5 = all, 1 = almost none."
    )
    citation_quality: Criterion = Field(
        description="Do the cited quotes actually support the sentences they are attached to? "
        "5 = all citations relevant and supporting, 1 = citations irrelevant or missing."
    )
    overall_pass: bool = Field(
        description="True only if the answer is faithful (>=4), correct (>=4) and properly cited (>=4)."
    )


JUDGE_SYSTEM = """You are a strict evaluator of a retrieval-augmented documentation assistant for the FastAPI framework.
You receive: the user question, a reference answer and key facts written by a human, the context excerpts that were given to the assistant, and the assistant's answer with its citations.
Grade only what is on the page. Do not use outside knowledge to excuse unsupported claims: a claim that is true in general but absent from the context excerpts is unfaithful.

Rubric (score each criterion 1-5, keep rationales short):
- faithfulness: every claim in the answer is supported by the context excerpts. 5 = fully supported; 3 = minor unsupported details; 1 = mostly unsupported or contradicted.
- correctness: the answer agrees with the reference answer on the facts that matter. 5 = fully consistent; 1 = wrong or contradicts it.
- completeness: the answer covers the key facts the question needs. 5 = all; 1 = almost none.
- citation_quality: the cited quotes support the sentences they are attached to. 5 = all relevant and supporting; 1 = irrelevant or missing.
- overall_pass: true only if faithfulness, correctness and citation_quality are all >= 4."""


def build_judge_prompt(
    question: str,
    reference_answer: str,
    key_facts: Sequence[str],
    context: Sequence[Chunk],
    answer: str,
    citations: Sequence[dict],
) -> str:
    ctx = "\n\n".join(f'<excerpt id="{c.chunk_id}">\n{c.text}\n</excerpt>' for c in context)
    cites = "\n".join(f'[{c["n"]}] {c["chunk_id"]}: "{c.get("quote", "")}"' for c in citations) or "(none)"
    facts = "\n".join(f"- {f}" for f in key_facts) or "(none)"
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_answer>\n{reference_answer}\n</reference_answer>\n\n"
        f"<key_facts>\n{facts}\n</key_facts>\n\n"
        f"<context>\n{ctx}\n</context>\n\n"
        f"<assistant_answer>\n{answer}\n</assistant_answer>\n\n"
        f"<assistant_citations>\n{cites}\n</assistant_citations>"
    )


class Judge:
    def __init__(self, settings: Settings, client: Any | None = None, effort: str = "medium"):
        self.settings = settings
        self.client = client if client is not None else make_client(settings)
        self.effort = effort

    def grade(
        self,
        question: str,
        reference_answer: str,
        key_facts: Sequence[str],
        context: Sequence[Chunk],
        answer: str,
        citations: Sequence[dict],
    ) -> tuple[JudgeVerdict | None, Usage | None, float, str | None]:
        """Returns (verdict, usage, latency_ms, error)."""
        t0 = time.perf_counter()
        resp = self.client.messages.parse(
            model=self.settings.judge_model,
            max_tokens=4096,
            system=JUDGE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": build_judge_prompt(question, reference_answer, key_facts, context, answer, citations),
                }
            ],
            output_config={"effort": self.effort},
            output_format=JudgeVerdict,
        )
        latency = (time.perf_counter() - t0) * 1000
        usage = usage_from_response(resp, self.settings.judge_model)
        if resp.stop_reason == "refusal":
            return None, usage, latency, "judge refused (stop_reason=refusal)"
        if resp.stop_reason == "max_tokens":
            return None, usage, latency, "judge hit max_tokens"
        return resp.parsed_output, usage, latency, None
