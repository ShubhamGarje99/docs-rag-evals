"""Citation mapping, refusal guard and request shape, with a mocked Claude client (no network)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from ragkit.engine import RAGEngine
from ragkit.llm import (
    FALLBACK_BETA,
    REFUSAL_TEXT,
    ClaudeAnswerer,
    build_request,
    parse_response,
    price_usage,
)


def usage(inp=1000, out=100, model_iters=None):
    return NS(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        iterations=model_iters,
    )


def cite(idx, title, text):
    return NS(
        type="char_location",
        document_index=idx,
        document_title=title,
        cited_text=text,
        start_char_index=0,
        end_char_index=len(text),
    )


def response(blocks, stop="end_turn", model="claude-sonnet-5-5", u=None):
    return NS(content=blocks, stop_reason=stop, model=model, usage=u or usage())


class FakeClient:
    """Records the kwargs of messages.create / beta.messages.create and returns a canned response."""

    def __init__(self, resp):
        self.calls = []
        outer = self

        class _Messages:
            def __init__(self, kind):
                self.kind = kind

            def create(self, **kw):
                outer.calls.append((self.kind, kw))
                return resp

        self.messages = _Messages("messages")
        self.beta = NS(messages=_Messages("beta"))


@pytest.fixture
def chunks(fixture_chunks):
    return fixture_chunks[:3]


def test_citations_map_to_chunk_ids_with_markers(chunks):
    a, b = chunks[0], chunks[1]
    resp = response(
        [
            NS(type="thinking", thinking=""),
            NS(
                type="text",
                text="Use background tasks",
                citations=[cite(0, a.chunk_id, "run after returning a response")],
            ),
            NS(
                type="text",
                text=" and the middleware",
                citations=[cite(1, b.chunk_id, "q1"), cite(0, a.chunk_id, "q2")],
            ),
            NS(type="text", text=".", citations=None),
        ]
    )
    ans = parse_response(resp, chunks, "claude-sonnet-5-5", 12.0)
    assert not ans.refused
    assert ans.answer == "Use background tasks[1] and the middleware[2][1]."
    assert [c.chunk_id for c in ans.citations] == [a.chunk_id, b.chunk_id]
    assert ans.citations[0].quotes == ["run after returning a response", "q2"]
    assert ans.citations[0].url == a.url
    assert ans.usage.cost_usd == pytest.approx(price_usage("claude-sonnet-5-5", 1000, 100))


def test_out_of_range_document_index_is_ignored(chunks):
    resp = response([NS(type="text", text="x", citations=[cite(7, "bogus", "zzz")])])
    ans = parse_response(resp, chunks, "claude-sonnet-5-5", 1.0)
    assert ans.refused and ans.refusal_reason == "no_citations"


def test_model_declines_with_refusal_sentence(chunks):
    resp = response([NS(type="text", text="I couldn’t find that in the FastAPI docs.", citations=None)])
    ans = parse_response(resp, chunks, "claude-sonnet-5-5", 1.0)
    assert ans.refused and ans.refusal_reason == "model_declined" and ans.answer == REFUSAL_TEXT


def test_uncited_answer_is_replaced_by_refusal(chunks):
    resp = response([NS(type="text", text="FastAPI supports rate limiting via @limit.", citations=[])])
    ans = parse_response(resp, chunks, "claude-sonnet-5-5", 1.0)
    assert ans.refused and ans.refusal_reason == "no_citations"
    assert ans.answer == REFUSAL_TEXT
    assert "rate limiting" in ans.raw_answer


def test_safety_refusal_stop_reason_is_checked_first(chunks):
    resp = response([NS(type="text", text="partial", citations=[cite(0, chunks[0].chunk_id, "q")])], stop="refusal")
    ans = parse_response(resp, chunks, "claude-sonnet-5-5", 1.0)
    assert ans.refused and ans.refusal_reason == "model_safety_refusal" and not ans.citations


def test_request_shape_documents_and_no_sampling_params(settings, chunks):
    req = build_request(settings, "How?", chunks)
    assert req["model"] == "claude-sonnet-5-5"
    assert req["output_config"] == {"effort": "low"}
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in req
    content = req["messages"][0]["content"]
    docs, question = content[:-1], content[-1]
    assert question == {"type": "text", "text": "How?"}
    assert len(docs) == len(chunks)
    for d, c in zip(docs, chunks):
        assert d["type"] == "document" and d["title"] == c.chunk_id
        assert d["citations"] == {"enabled": True}
        assert d["source"] == {"type": "text", "media_type": "text/plain", "data": c.text}
    assert req["messages"][-1]["role"] == "user"  # no assistant prefill


def test_sonnet_uses_server_side_fallback_beta(settings, chunks):
    client = FakeClient(response([NS(type="text", text="ok", citations=[cite(0, chunks[0].chunk_id, "q")])]))
    ClaudeAnswerer(settings, client=client).answer("q", chunks)
    kind, kw = client.calls[0]
    assert kind == "beta"
    assert kw["fallbacks"] == "default" and kw["betas"] == [FALLBACK_BETA]


def test_fallbacks_off_and_haiku_use_plain_messages(settings, chunks):
    from dataclasses import replace

    resp = response([NS(type="text", text="ok", citations=[cite(0, chunks[0].chunk_id, "q")])])
    for s in (replace(settings, claude_fallbacks=False), replace(settings, claude_model="claude-haiku-5-5")):
        client = FakeClient(resp)
        ClaudeAnswerer(s, client=client).answer("q", chunks)
        kind, kw = client.calls[0]
        assert kind == "messages" and "fallbacks" not in kw and "betas" not in kw


def test_fallback_iterations_are_priced_per_model(chunks):
    iters = [
        NS(
            type="message",
            model="claude-sonnet-5-5",
            input_tokens=1000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        NS(
            type="fallback_message",
            model="claude-sonnet-5",
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    ]
    resp = response(
        [NS(type="text", text="ok", citations=[cite(0, chunks[0].chunk_id, "q")])],
        model="claude-sonnet-5",
        u=usage(2000, 200, iters),
    )
    ans = parse_response(resp, chunks, "claude-sonnet-5-5", 1.0)
    assert ans.fallback_used
    assert ans.usage.cost_usd == pytest.approx((1000 * 2 + 1000 * 2 + 200 * 10) / 1e6)


def test_engine_retrieval_gate_refuses_without_calling_claude(settings, fixture_chunks):
    from dataclasses import replace

    from ragkit.embeddings import HashingEmbedder
    from ragkit.store import MemoryStore

    class ExplodingAnswerer:
        def answer(self, *a, **k):
            raise AssertionError("Claude must not be called when the gate refuses")

    strict = replace(settings, min_relevance=0.99)
    store = MemoryStore.from_chunks(fixture_chunks, HashingEmbedder())
    eng = RAGEngine(store, HashingEmbedder(), strict, answerer=ExplodingAnswerer())
    out = eng.chat("How do I add rate limiting per IP address?", "hybrid")
    assert out["mode"] == "llm" and out["refused"] and out["refusal_reason"] == "low_retrieval_score"
    assert out["answer"] == REFUSAL_TEXT and out["citations"] == [] and out["usage"] is None
