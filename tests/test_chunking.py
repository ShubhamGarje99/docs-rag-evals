from __future__ import annotations

import re

from ragkit.chunking import _parse_sections, chunk_markdown, count_tokens, slugify

PAGE = """# Request Body { #request-body }

Intro paragraph about request bodies.

## Create your data model { #create-your-data-model }

Declare a class that inherits from `BaseModel`.

### Optional fields

Use `None` as the default to make a field optional.

```python
# Not a heading: this line is inside a code fence
class Item(BaseModel):
    name: str
```

## Results

FastAPI will read the body as JSON.
"""


def _long_section(n_paragraphs: int = 12) -> str:
    paras = [
        f"Paragraph {i} explains detail number {i} about dependencies. It has two sentences for overlap tests."
        for i in range(n_paragraphs)
    ]
    return "# Big Page\n\n## Long section { #long }\n\n" + "\n\n".join(paras) + "\n"


def test_ids_are_stable_and_unique():
    a = chunk_markdown(PAGE, "tutorial/body.md", "https://fastapi.tiangolo.com/tutorial/body/", max_tokens=40)
    b = chunk_markdown(PAGE, "tutorial/body.md", "https://fastapi.tiangolo.com/tutorial/body/", max_tokens=40)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    for c in a:
        assert re.fullmatch(r"tutorial/body\.md#[a-z0-9-]+-\d{2}", c.chunk_id), c.chunk_id


def test_explicit_anchor_slug_fallback_and_breadcrumbs():
    secs = {s.anchor: s for s in _parse_sections(PAGE)}
    assert list(secs) == ["request-body", "create-your-data-model", "optional-fields", "results"]
    assert secs["request-body"].title == "Request Body"  # { #anchor } stripped from the title
    assert secs["optional-fields"].heading_path == ["Request Body", "Create your data model", "Optional fields"]
    assert secs["results"].heading_path == ["Request Body", "Results"]  # level-3 popped on a new level-2
    # the fenced "# Not a heading" line stays inside the optional-fields section
    assert any("# Not a heading" in ln for ln in secs["optional-fields"].lines)


def test_chunk_urls_and_anchor_markup():
    words = " ".join(f"word{i}" for i in range(30))
    page = f"# Page {{ #page }}\n\n{words}\n\n## Second part {{ #second }}\n\n{words}\n\n## Third\n\n{words}\n"
    chunks = chunk_markdown(page, "x/p.md", "https://x.dev/x/p/", max_tokens=40, overlap_tokens=0)
    assert [c.chunk_id for c in chunks] == ["x/p.md#page-00", "x/p.md#second-00", "x/p.md#third-00"]
    assert chunks[0].url == "https://x.dev/x/p/"  # H1 -> page URL
    assert chunks[1].url == "https://x.dev/x/p/#second"
    assert chunks[2].heading_path == ["Page", "Third"]
    assert all("{ #" not in c.text for c in chunks)


def test_fenced_code_block_kept_whole():
    chunks = chunk_markdown(PAGE, "p.md", "https://x/", max_tokens=60, min_tokens=0)
    fenced = next(c for c in chunks if "class Item" in c.text)
    assert "# Not a heading" in fenced.text and fenced.text.count("```") == 2


def test_small_sections_are_merged_into_one_chunk():
    chunks = chunk_markdown(PAGE, "p.md", "https://x/", max_tokens=400)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "p.md#request-body-00"
    assert "## Results" in chunks[0].text


def test_budget_and_overlap_within_a_section():
    chunks = chunk_markdown(_long_section(), "big.md", "https://x/big/", max_tokens=80, overlap_tokens=25)
    long = [c for c in chunks if c.anchor == "long"]
    assert len(long) >= 3
    assert [c.ordinal for c in long] == list(range(len(long)))
    assert all(c.n_tokens <= 80 for c in chunks)
    for prev, nxt in zip(long, long[1:]):
        tail = prev.text.split("\n\n")[-1]
        assert nxt.text.startswith(tail[-40:]) or tail in nxt.text, "consecutive chunks should share trailing text"


def test_no_overlap_when_disabled():
    chunks = chunk_markdown(_long_section(), "big.md", "https://x/big/", max_tokens=80, overlap_tokens=0)
    paras = [p for c in chunks for p in c.text.split("\n\n") if p.startswith("Paragraph")]
    assert len(paras) == len(set(paras)) == 12


def test_oversized_code_block_is_split_with_fences():
    code = "```python\n" + "\n".join(f"value_{i} = compute({i}, {i + 1})" for i in range(60)) + "\n```"
    chunks = chunk_markdown(f"# Code page\n\n{code}\n", "c.md", "https://x/", max_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c.n_tokens <= 100
        assert c.text.count("```") % 2 == 0


def test_helpers():
    assert slugify("Use `CORSMiddleware`") == "use-corsmiddleware"
    assert count_tokens("app.add_middleware(CORSMiddleware)") == 6
