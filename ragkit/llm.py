"""Claude answering with native Citations, plus the shared pricing / usage helpers.

Every retrieved chunk is sent as its own plain-text ``document`` block with
``citations: {enabled: true}`` and ``title = chunk_id``. Claude's citations come
back as ``char_location`` objects whose ``document_index`` points into that list,
so mapping a citation to a chunk is exact (no string matching).

Refusal guard (three layers):
1. retrieval gate - the engine refuses before calling Claude when the best
   question/chunk cosine similarity is below ``MIN_RELEVANCE``;
2. prompt - Claude is told to answer only from the documents and otherwise to
   reply with ``REFUSAL_TEXT``;
3. citation gate - an answer that cites nothing is treated as ungrounded and
   replaced by ``REFUSAL_TEXT``.

Model notes (see the claude-api skill): Sonnet 5.5 / Haiku 5.5 reject non-default
sampling params, ``budget_tokens`` and prefill, so none are sent; effort is set
explicitly via ``output_config``; ``stop_reason == "refusal"`` is checked before
reading content; Sonnet requests opt into server-side fallback
(``fallbacks="default"``) unless ``CLAUDE_FALLBACKS=off``. Citations cannot be
combined with ``output_config.format`` so the answer call uses none.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .chunking import Chunk
from .config import NO_KEY_MESSAGE, Settings

REFUSAL_TEXT = "I couldn't find that in the FastAPI docs."
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens (input, output). Cache writes bill at 1.25x input, reads at 0.1x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),  # server-side fallback target for Sonnet 5.5
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}

SYSTEM_PROMPT = f"""You are a documentation assistant for the FastAPI web framework.
The user's message contains excerpts from the FastAPI docs, one document per excerpt (the document title is the excerpt id), followed by a question.

Answer using only these excerpts:
- Base every statement on the excerpts and cite them. Do not add facts from outside knowledge, even if you are confident they are true.
- If the excerpts do not contain the information needed to answer, reply with exactly this sentence and nothing else: {REFUSAL_TEXT}
- If they answer only part of the question, answer that part and say briefly what the docs excerpts don't cover.
- Be concise: a short paragraph or a few bullet points. Show a short code example only when an excerpt contains one that answers the question.
"""


class LLMNotConfigured(RuntimeError):
    def __init__(self, message: str = NO_KEY_MESSAGE):
        super().__init__(message)


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def price_usage(model: str, inp: int, out: int, cache_write: int = 0, cache_read: int = 0) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    pin, pout = p
    return (inp * pin + out * pout + cache_write * pin * 1.25 + cache_read * pin * 0.10) / 1_000_000


def usage_from_response(resp: Any, requested_model: str) -> Usage:
    u = resp.usage
    model = getattr(resp, "model", None) or requested_model
    usage = Usage(
        model=model,
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )
    iterations = getattr(u, "iterations", None) or []
    if iterations:  # with server-side fallback each model's share is billed at its own rate
        total, known = 0.0, True
        for it in iterations:
            c = price_usage(
                getattr(it, "model", None) or model,
                getattr(it, "input_tokens", 0) or 0,
                getattr(it, "output_tokens", 0) or 0,
                getattr(it, "cache_creation_input_tokens", 0) or 0,
                getattr(it, "cache_read_input_tokens", 0) or 0,
            )
            if c is None:
                known = False
                break
            total += c
        usage.cost_usd = total if known else None
    if usage.cost_usd is None:
        usage.cost_usd = price_usage(
            model if model in PRICES else requested_model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
        )
    return usage


def supports_server_fallback(model: str) -> bool:
    # Haiku 5.5 has no server-side fallback; don't send the parameter there.
    return "haiku" not in model


def make_client(settings: Settings):
    if not settings.llm_enabled:
        raise LLMNotConfigured()
    import anthropic

    return anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=2, timeout=90.0)


# --------------------------------------------------------------------------- #
# Answer generation
# --------------------------------------------------------------------------- #


@dataclass
class Citation:
    n: int
    chunk_id: str
    title: str
    url: str
    quote: str
    quotes: list[str] = field(default_factory=list)


@dataclass
class Answer:
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    usage: Usage | None
    latency_ms: float
    stop_reason: str | None
    fallback_used: bool = False
    raw_answer: str | None = None  # model text before the citation gate replaced it

    @property
    def cited_chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.citations]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = self.usage.to_dict() if self.usage else None
        return d


def build_documents(chunks: Sequence[Chunk]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c.text},
            "title": c.chunk_id,
            "context": f"Page: {c.page_title}. Section: {c.breadcrumb}. URL: {c.url}",
            "citations": {"enabled": True},
        }
        for c in chunks
    ]


def build_request(settings: Settings, question: str, chunks: Sequence[Chunk]) -> dict:
    return {
        "model": settings.claude_model,
        "max_tokens": settings.claude_max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [*build_documents(chunks), {"type": "text", "text": question}],
            }
        ],
        "output_config": {"effort": settings.claude_effort},
    }


def _is_refusal_text(text: str) -> bool:
    norm = " ".join(text.lower().replace("’", "'").split())
    return "couldn't find that in the fastapi docs" in norm


def parse_response(resp: Any, chunks: Sequence[Chunk], requested_model: str, latency_ms: float) -> Answer:
    """Map a Messages API response (with citations) onto an ``Answer``."""
    usage = usage_from_response(resp, requested_model) if getattr(resp, "usage", None) else None
    blocks = list(getattr(resp, "content", None) or [])
    fallback_used = any(getattr(b, "type", "") == "fallback" for b in blocks) or any(
        getattr(it, "type", "") == "fallback_message"
        for it in (getattr(getattr(resp, "usage", None), "iterations", None) or [])
    )

    if resp.stop_reason == "refusal":
        return Answer(
            answer=REFUSAL_TEXT,
            refused=True,
            refusal_reason="model_safety_refusal",
            citations=[],
            usage=usage,
            latency_ms=latency_ms,
            stop_reason=resp.stop_reason,
            fallback_used=fallback_used,
        )

    numbers: dict[str, int] = {}
    cites: dict[str, Citation] = {}
    parts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) != "text":
            continue
        parts.append(block.text)
        marks: list[int] = []
        for c in getattr(block, "citations", None) or []:
            idx = getattr(c, "document_index", None)
            if idx is None or not (0 <= idx < len(chunks)):
                continue  # cannot happen for valid responses; never trust blindly
            chunk = chunks[idx]
            if chunk.chunk_id not in numbers:
                numbers[chunk.chunk_id] = len(numbers) + 1
                cites[chunk.chunk_id] = Citation(
                    n=numbers[chunk.chunk_id],
                    chunk_id=chunk.chunk_id,
                    title=chunk.breadcrumb,
                    url=chunk.url,
                    quote="",
                )
            quote = (getattr(c, "cited_text", "") or "").strip()
            cit = cites[chunk.chunk_id]
            if quote and quote not in cit.quotes:
                cit.quotes.append(quote)
                cit.quote = " … ".join(cit.quotes)
            n = numbers[chunk.chunk_id]
            if n not in marks:
                marks.append(n)
        if marks:
            parts.append("".join(f"[{n}]" for n in marks))
    text = "".join(parts).strip()
    citations = sorted(cites.values(), key=lambda c: c.n)

    if resp.stop_reason == "max_tokens":
        text += "\n\n(Answer truncated: max_tokens reached.)"

    if _is_refusal_text(text) and not citations:
        return Answer(REFUSAL_TEXT, True, "model_declined", [], usage, latency_ms, resp.stop_reason, fallback_used)
    if not citations:
        return Answer(
            REFUSAL_TEXT, True, "no_citations", [], usage, latency_ms, resp.stop_reason, fallback_used, raw_answer=text
        )
    return Answer(text, False, None, citations, usage, latency_ms, resp.stop_reason, fallback_used)


class ClaudeAnswerer:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self.client = client if client is not None else make_client(settings)

    def answer(self, question: str, chunks: Sequence[Chunk]) -> Answer:
        import anthropic

        req = build_request(self.settings, question, chunks)
        t0 = time.perf_counter()
        try:
            if self.settings.claude_fallbacks and supports_server_fallback(self.settings.claude_model):
                resp = self.client.beta.messages.create(**req, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                resp = self.client.messages.create(**req)
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY in .env.") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Claude API rejected the request: {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Rate limited by the Claude API, retry shortly.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Claude API (network error).") from e
        latency = (time.perf_counter() - t0) * 1000
        return parse_response(resp, chunks, self.settings.claude_model, latency)
