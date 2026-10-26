"""Reproduce the vendored FastAPI docs snapshot under ``corpus/``.

Downloads the pinned FastAPI release tarball from GitHub, verifies the commit
and the MIT LICENSE text, selects a curated subset of ``docs/en/docs/`` and
rewrites the mkdocs-specific syntax into plain Markdown:

* ``{* ../../docs_src/... ln[..] hl[..] *}`` include directives are replaced by
  the referenced example file (honouring ``ln[...]`` line ranges) as a fenced
  ``python`` block, because the code examples are a large part of what users ask.
* ``/// tip | Title`` admonitions become a bold label line plus the body.
* ``//// tab | Label`` groups keep a bold label; "non-Annotated" duplicate
  variants are dropped.
* Inline HTML (``<abbr>``, ``<dfn>``, ``<font>``, ``<img>``, ``<div class="termy">``...)
  is reduced to its text; ``{.internal-link target=_blank}`` attribute lists
  are removed. Heading anchors ``{ #anchor }`` are kept, the chunker uses them
  to build deep links to https://fastapi.tiangolo.com.

The tarball is treated as untrusted data: it is only read, never executed.

Usage:
    uv run python scripts/fetch_corpus.py            # download + rebuild corpus/
    uv run python scripts/fetch_corpus.py --tarball path/to/fastapi.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = "https://github.com/fastapi/fastapi"
TAG = "0.162.0"
COMMIT = "7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4"
TARBALL_URL = f"https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/{TAG}"
DOCS_BASE_URL = "https://fastapi.tiangolo.com/"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "corpus"

# Curated subset: the whole tutorial + advanced user guide + deployment, plus a
# few high-traffic top-level pages. how-to/, reference/ (API autodoc), release
# notes, translations and community pages are intentionally excluded.
INCLUDE_DIRS = ("tutorial/", "advanced/", "deployment/")
INCLUDE_FILES = ("async.md", "python-types.md", "features.md", "fastapi-cli.md")

LICENSE_MUST_CONTAIN = (
    "The MIT License (MIT)",
    "Copyright (c) 2018 Sebastián Ramírez",
    "Permission is hereby granted, free of charge",
    'THE SOFTWARE IS PROVIDED "AS IS"',
)


def doc_url(rel: str) -> str:
    """docs/en/docs-relative path -> canonical page URL (mkdocs directory URLs)."""
    p = rel[:-3]  # strip .md
    if p == "index":
        return DOCS_BASE_URL
    if p.endswith("/index"):
        return DOCS_BASE_URL + p[: -len("index")]
    return DOCS_BASE_URL + p + "/"


# --------------------------------------------------------------------------- #
# Markdown cleaning
# --------------------------------------------------------------------------- #

INCLUDE_RE = re.compile(r"^\{\*\s*(?P<path>\S+)(?P<opts>.*?)\*\}\s*$")
ADMONITION_OPEN_RE = re.compile(r"^///\s*(?P<kind>[a-z]+)\s*(?:\|\s*(?P<title>.*))?$")
TAB_OPEN_RE = re.compile(r"^////\s*tab\s*\|\s*(?P<label>.*)$")
FENCE_RE = re.compile(r"^(\s*)(```+|~~~+)\s*(?P<info>.*)$")
ATTR_LIST_RE = re.compile(r"\)\{[^}\n]*\}")
ABBR_RE = re.compile(r'<(abbr|dfn)\s+title="([^"]*)"\s*>(.*?)</\1>', re.S)
STRIP_TAGS_RE = re.compile(
    r"</?(?:font|span|b|u|div|details|summary|small|strong|a|em|i|br|p|"
    r"table|thead|tbody|tr|td|th|dfn|abbr|code)\b[^>]*>",
    re.I,
)
IMG_RE = re.compile(r"<img\b[^>]*>", re.I)
MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _parse_ranges(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(part), int(part)))
    return out


def _select_lines(lines: list[str], spec: str) -> str:
    pieces = ["\n".join(lines[a - 1 : b]) for a, b in _parse_ranges(spec)]
    return "\n# ...\n".join(p for p in pieces if p.strip())


def render_include(path: str, opts: str, read_src, seen: set[str]) -> str:
    """Render an include directive.

    The upstream docs re-include the same example file many times on one page,
    each time highlighting different lines (``hl[...]``). To avoid near-duplicate
    chunks we inline the full file (or its ``ln[...]`` range) the first time it
    appears on a page, and only the highlighted lines on later references.
    """
    rel = re.sub(r"^(\.\./)+", "", path)
    code = read_src(rel)
    if code is None:
        return ""
    lines = code.rstrip("\n").split("\n")
    ln = re.search(r"ln\[([^\]]*)\]", opts)
    hl = re.search(r"hl\[([^\]]*)\]", opts)
    if ln:
        body = _select_lines(lines, ln.group(1))
    elif rel not in seen:
        body = "\n".join(lines)
    elif hl:
        body = _select_lines(lines, hl.group(1))
    else:
        body = ""
    seen.add(rel)
    return f"```python\n{body}\n```" if body.strip() else ""


def clean_inline_html(line: str) -> str:
    line = IMG_RE.sub("", line)
    line = MD_IMG_RE.sub("", line)
    line = ABBR_RE.sub(lambda m: f"{STRIP_TAGS_RE.sub('', m.group(3))} ({m.group(2)})", line)
    line = re.sub(r"</?code>", "`", line)
    line = STRIP_TAGS_RE.sub("", line)
    line = ATTR_LIST_RE.sub(")", line)
    return line


def clean_markdown(text: str, read_src) -> str:
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    fence_is_console = False
    skip_tab = False  # inside a dropped "non-Annotated" tab
    seen_includes: set[str] = set()
    for raw in text.split("\n"):
        line = raw.rstrip()
        fm = FENCE_RE.match(line)
        if in_fence:
            if fm and fm.group(2).startswith(fence_marker[:3]) and not fm.group("info"):
                in_fence = False
                if not skip_tab:
                    out.append(fm.group(1) + fence_marker)
                continue
            if not skip_tab:
                out.append(STRIP_TAGS_RE.sub("", line) if fence_is_console else line)
            continue

        if fm:
            in_fence = True
            fence_marker = fm.group(2)
            info = fm.group("info").strip()
            lang = info.split()[0].lower() if info else ""
            fence_is_console = lang in {"console", "bash", "shell", ""}
            if not skip_tab:
                out.append(f"{fm.group(1)}{fence_marker}{lang}")
            continue

        tab = TAB_OPEN_RE.match(line)
        if tab:
            label = tab.group("label").strip()
            skip_tab = "non-annotated" in label.lower()
            if not skip_tab:
                out.append(f"**{label.strip('`')}:**")
            continue
        if line.strip() == "////":
            skip_tab = False
            continue
        if skip_tab:
            continue

        inc = INCLUDE_RE.match(line.strip())
        if inc:
            rendered = render_include(inc.group("path"), inc.group("opts"), read_src, seen_includes)
            if rendered:
                out.extend(rendered.split("\n"))
            continue

        adm = ADMONITION_OPEN_RE.match(line.strip())
        if adm:
            kind = adm.group("kind").capitalize()
            title = (adm.group("title") or "").strip()
            label = f"{kind}: {title}" if title and title.lower() != kind.lower() else kind
            out.append(f"**{label}:**")
            continue
        if line.strip() == "///":
            continue

        out.append(clean_inline_html(line))

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def selected(rel: str) -> bool:
    return rel.endswith(".md") and (rel.startswith(INCLUDE_DIRS) or rel in INCLUDE_FILES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tarball", type=Path, help="use a local copy of the release tarball")
    args = ap.parse_args()

    if args.tarball:
        data = args.tarball.read_bytes()
    else:
        print(f"Downloading {TARBALL_URL} ...")
        with urllib.request.urlopen(TARBALL_URL, timeout=120) as r:  # noqa: S310 (pinned https URL)
            data = r.read()
    tar_sha = hashlib.sha256(data).hexdigest()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        commit = tar.pax_headers.get("comment", "")
        if commit != COMMIT:
            print(f"ERROR: tarball commit {commit!r} != pinned {COMMIT}", file=sys.stderr)
            return 1
        members = {m.name.split("/", 1)[1]: m for m in tar.getmembers() if m.isfile() and "/" in m.name}

        def read(rel: str) -> str | None:
            m = members.get(rel)
            if m is None:
                return None
            f = tar.extractfile(m)
            return f.read().decode("utf-8") if f else None

        license_text = read("LICENSE") or ""
        for needle in LICENSE_MUST_CONTAIN:
            if needle not in license_text:
                print(f"ERROR: upstream LICENSE missing {needle!r}", file=sys.stderr)
                return 1

        docs_prefix = "docs/en/docs/"
        pages = sorted(
            rel[len(docs_prefix) :]
            for rel in members
            if rel.startswith(docs_prefix) and selected(rel[len(docs_prefix) :])
        )

        tmp = Path(tempfile.mkdtemp(prefix="corpus-"))
        manifest = []
        missing_includes: list[str] = []

        def read_src(rel: str) -> str | None:
            txt = read(rel)
            if txt is None:
                missing_includes.append(rel)
            return txt

        for rel in pages:
            raw = read(docs_prefix + rel) or ""
            cleaned = clean_markdown(raw, read_src)
            dest = tmp / "docs" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(cleaned, encoding="utf-8")
            title_m = re.search(r"^#\s+(.+?)(?:\s*\{\s*#[^}]*\})?\s*$", cleaned, re.M)
            manifest.append(
                {
                    "path": rel,
                    "title": title_m.group(1).strip() if title_m else rel,
                    "url": doc_url(rel),
                    "upstream_path": docs_prefix + rel,
                    "upstream_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "words": len(cleaned.split()),
                }
            )

    if missing_includes:
        print(f"WARNING: {len(missing_includes)} include(s) not found: {missing_includes[:5]}")

    (tmp / "LICENSE").write_text(license_text, encoding="utf-8")
    (tmp / "manifest.json").write_text(
        json.dumps(
            {
                "source_repo": REPO,
                "tag": TAG,
                "commit": COMMIT,
                "tarball_url": TARBALL_URL,
                "tarball_sha256": tar_sha,
                "license": "MIT (see corpus/LICENSE)",
                "pages": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp / "NOTICE.md").write_text(
        f"""# Corpus attribution

The Markdown files under `corpus/docs/` are derived from the FastAPI documentation
(`docs/en/docs/`) in {REPO}, release tag `{TAG}` (commit `{COMMIT}`).

FastAPI and its documentation are Copyright (c) 2018 Sebastián Ramírez and are
distributed under the MIT License, reproduced verbatim in `corpus/LICENSE`.

Modifications made by `scripts/fetch_corpus.py` (reproducible, deterministic):

- Only a subset of pages is included ({len(manifest)} pages: `tutorial/`, `advanced/`,
  `deployment/`, and {", ".join(f"`{f}`" for f in INCLUDE_FILES)}).
- mkdocs include directives (`{{* ../../docs_src/... *}}`) were replaced by the
  referenced example source files from the same release, as fenced Python blocks.
- Admonitions, content tabs, inline HTML and attribute lists were converted to
  plain Markdown; images were removed. "non-Annotated" duplicate code tabs were dropped.

This corpus is used here only as retrieval data for a demo / evaluation project.
It is not affiliated with or endorsed by the FastAPI project.
""",
        encoding="utf-8",
    )

    if OUT.exists():
        for child in ("docs", "LICENSE", "manifest.json", "NOTICE.md"):
            p = OUT / child
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    OUT.mkdir(exist_ok=True)
    for child in tmp.iterdir():
        shutil.move(str(child), OUT / child.name)
    shutil.rmtree(tmp, ignore_errors=True)
    words = sum(p["words"] for p in manifest)
    print(f"Wrote {len(manifest)} pages ({words:,} words) to {OUT.relative_to(ROOT)}/ from {TAG} @ {COMMIT[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
