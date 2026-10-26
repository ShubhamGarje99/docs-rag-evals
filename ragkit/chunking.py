"""Heading-aware Markdown chunking with stable chunk ids.

Design
------
1. A page is split into *sections* at Markdown headings (headings inside fenced
   code are ignored). Each section knows its heading breadcrumb
   (``Request Body > Create your data model``) and its anchor (the explicit
   ``{ #anchor }`` from the FastAPI docs, else a slug of the heading).
2. A section body is split into *blocks*: paragraphs / lists separated by blank
   lines, with fenced code blocks kept whole. Blocks are never cut unless a
   single block is larger than the chunk budget (then code is split by lines and
   prose by sentences).
3. Blocks are packed greedily into chunks of at most ``max_tokens``. Small
   neighbouring sections are merged into one chunk, a new chunk starts at a
   section boundary when the next section does not fit, and when one section
   spans several chunks consecutive chunks share ~``overlap_tokens`` of trailing
   prose (overlap never crosses a section boundary).

Chunk ids are ``<page path>#<anchor>-<NN>`` where ``NN`` is the ordinal of the
chunk within that section, e.g. ``tutorial/body.md#create-your-data-model-00``.
They depend only on the page text, so re-ingesting the same corpus snapshot
always yields the same ids (the golden eval set references them).

Token counts use a tokenizer-free proxy (words + punctuation marks), which
tracks BPE / WordPiece counts closely enough for budgeting and keeps the
chunker free of model downloads.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([A-Za-z0-9_.:-]+)\s*\})?\s*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z`*\[(\"'])")


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def slugify(text: str) -> str:
    text = re.sub(r"[`*_]", "", text.lower())
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return re.sub(r"[\s-]+", "-", text).strip("-") or "section"


@dataclass
class Chunk:
    chunk_id: str
    doc_path: str
    page_title: str
    heading_path: list[str]
    anchor: str
    url: str
    ordinal: int
    position: int
    text: str
    n_tokens: int

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def search_text(self) -> str:
        """Text used for both BM25 and embeddings: breadcrumb + chunk body."""
        return f"{self.breadcrumb}\n\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Section:
    level: int
    title: str
    anchor: str
    heading_path: list[str]
    heading_line: str | None
    lines: list[str] = field(default_factory=list)


@dataclass
class _Block:
    text: str
    tokens: int
    is_code: bool
    section: int  # index into the page's section list
    overlap: bool = False  # repeated from the previous chunk


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _parse_sections(markdown: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current = _Section(level=0, title="", anchor="", heading_path=[], heading_line=None)
    in_fence = False
    fence = ""
    for line in markdown.split("\n"):
        fm = FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)
            if not in_fence:
                in_fence, fence = True, marker
            elif marker.startswith(fence[:3]) and line.strip() == marker:
                in_fence = False
            current.lines.append(line)
            continue
        hm = None if in_fence else HEADING_RE.match(line)
        if hm:
            if current.lines or current.heading_line:
                sections.append(current)
            level = len(hm.group(1))
            title = hm.group(2).strip()
            anchor = hm.group(3) or slugify(title)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = _Section(
                level=level,
                title=title,
                anchor=anchor,
                heading_path=[t for _, t in stack],
                heading_line=f"{hm.group(1)} {title}",
            )
            continue
        current.lines.append(line)
    if current.lines or current.heading_line:
        sections.append(current)
    # Drop an empty preamble before the first heading.
    return [s for s in sections if s.heading_line or any(ln.strip() for ln in s.lines)]


def _split_blocks(lines: list[str]) -> list[tuple[str, bool]]:
    blocks: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_fence = False
    fence = ""

    def flush(is_code: bool = False) -> None:
        nonlocal buf
        text = "\n".join(buf).strip("\n")
        if text.strip():
            blocks.append((text, is_code))
        buf = []

    for line in lines:
        fm = FENCE_RE.match(line)
        if in_fence:
            buf.append(line)
            if fm and fm.group(1).startswith(fence[:3]) and line.strip() == fm.group(1):
                in_fence = False
                flush(is_code=True)
            continue
        if fm:
            flush()
            in_fence, fence = True, fm.group(1)
            buf.append(line)
            continue
        if not line.strip():
            flush()
            continue
        buf.append(line)
    flush(is_code=in_fence)
    return blocks


def _split_oversized(text: str, is_code: bool, max_tokens: int) -> list[str]:
    """Split a single block that exceeds the budget."""
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    if is_code:
        lines = text.split("\n")
        opener, closer = lines[0], lines[-1] if FENCE_RE.match(lines[-1]) else "```"
        body = lines[1:-1] if FENCE_RE.match(lines[-1]) else lines[1:]
        budget = max_tokens - count_tokens(opener) - count_tokens(closer)
        cur: list[str] = []
        cur_tok = 0
        for ln in body:
            t = count_tokens(ln)
            if cur and cur_tok + t > budget:
                pieces.append("\n".join([opener, *cur, closer]))
                cur, cur_tok = [], 0
            cur.append(ln)
            cur_tok += t
        if cur:
            pieces.append("\n".join([opener, *cur, closer]))
        return pieces
    units = SENTENCE_SPLIT_RE.split(text)
    expanded: list[str] = []
    for u in units:
        if count_tokens(u) <= max_tokens:
            expanded.append(u)
        else:  # pathological sentence: split by words
            words = u.split()
            step = max(1, max_tokens // 2)
            expanded.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    cur, cur_tok = [], 0
    for u in expanded:
        t = count_tokens(u)
        if cur and cur_tok + t > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += t
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _overlap_tail(blocks: list[_Block], overlap_tokens: int) -> list[_Block]:
    """Trailing prose of the previous chunk (<= overlap_tokens) to repeat in the next one."""
    if overlap_tokens <= 0 or not blocks:
        return []
    tail: list[_Block] = []
    budget = overlap_tokens
    for b in reversed(blocks):
        if b.is_code:
            break
        if b.tokens <= budget:
            tail.insert(0, _Block(b.text, b.tokens, False, b.section, overlap=True))
            budget -= b.tokens
            continue
        # Take the last sentences of this block that fit.
        sents = SENTENCE_SPLIT_RE.split(b.text)
        picked: list[str] = []
        for s in reversed(sents):
            t = count_tokens(s)
            if t > budget:
                break
            picked.insert(0, s)
            budget -= t
        if picked:
            txt = " ".join(picked)
            tail.insert(0, _Block(txt, count_tokens(txt), False, b.section, overlap=True))
        break
    return tail


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def chunk_markdown(
    markdown: str,
    doc_path: str,
    page_url: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    min_tokens: int = 80,
) -> list[Chunk]:
    sections = _parse_sections(markdown)
    if not sections:
        return []
    page_title = next((s.title for s in sections if s.level == 1), sections[0].title or doc_path)

    # Turn every section into a list of blocks (the heading line is the first block).
    sec_blocks: list[list[_Block]] = []
    for i, sec in enumerate(sections):
        blocks: list[_Block] = []
        raw = ([(sec.heading_line, False)] if sec.heading_line else []) + _split_blocks(sec.lines)
        for text, is_code in raw:
            for piece in _split_oversized(text, is_code, max_tokens):
                blocks.append(_Block(piece, count_tokens(piece), is_code, i))
        sec_blocks.append(blocks)

    raw_chunks: list[tuple[int, list[_Block]]] = []  # (owner section, blocks)
    buf: list[_Block] = []
    buf_tok = 0
    owner = 0

    def flush() -> list[_Block]:
        nonlocal buf, buf_tok
        done = buf
        # A chunk made only of an overlap tail carries no new content.
        if any(not b.overlap for b in buf):
            raw_chunks.append((owner, buf))
        buf, buf_tok = [], 0
        return done

    for i, blocks in enumerate(sec_blocks):
        sec_tok = sum(b.tokens for b in blocks)
        # Prefer to start a new chunk at a section boundary, unless the current
        # chunk is still tiny (e.g. a one-line page intro), then keep filling it.
        if buf and buf_tok + sec_tok > max_tokens and buf_tok >= min_tokens:
            flush()
        if not buf:
            owner = i
        for b in blocks:
            if buf and buf_tok + b.tokens > max_tokens:
                prev = flush()
                tail = _overlap_tail([x for x in prev if x.section == i], overlap_tokens)
                owner = i
                buf = list(tail)
                buf_tok = sum(t.tokens for t in tail)
                if buf and buf_tok + b.tokens > max_tokens:  # overlap would overflow: drop it
                    buf, buf_tok = [], 0
            buf.append(b)
            buf_tok += b.tokens
    if buf:
        flush()

    chunks: list[Chunk] = []
    ordinals: dict[str, int] = {}
    seen_ids: set[str] = set()
    for position, (sec_idx, blocks) in enumerate(raw_chunks):
        sec = sections[sec_idx]
        anchor = sec.anchor or slugify(page_title)
        n = ordinals.get(anchor, 0)
        ordinals[anchor] = n + 1
        chunk_id = f"{doc_path}#{anchor}-{n:02d}"
        while chunk_id in seen_ids:  # duplicate anchors on one page (should not happen)
            n += 1
            chunk_id = f"{doc_path}#{anchor}-{n:02d}"
        seen_ids.add(chunk_id)
        text = "\n\n".join(b.text for b in blocks).strip()
        url = page_url if sec.level <= 1 else f"{page_url}#{anchor}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_path=doc_path,
                page_title=page_title,
                heading_path=sec.heading_path or [page_title],
                anchor=anchor,
                url=url,
                ordinal=n,
                position=position,
                text=text,
                n_tokens=count_tokens(text),
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# Corpus helpers
# --------------------------------------------------------------------------- #


def load_manifest(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))


def chunk_corpus(corpus_dir: Path, max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    """Chunk every page listed in ``corpus_dir/manifest.json`` (or every .md file if absent)."""
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    if manifest_path.exists():
        pages = [(p["path"], p["url"]) for p in load_manifest(corpus_dir)["pages"]]
    else:
        pages = [(str(p.relative_to(corpus_dir / "docs")), "") for p in sorted((corpus_dir / "docs").rglob("*.md"))]
    chunks: list[Chunk] = []
    for rel, url in pages:
        text = (corpus_dir / "docs" / rel).read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, rel, url, max_tokens, overlap_tokens))
    return chunks
