# Corpus attribution

The Markdown files under `corpus/docs/` are derived from the FastAPI documentation
(`docs/en/docs/`) in https://github.com/fastapi/fastapi, release tag `0.162.0` (commit `7a0a3f0b286c6c865cf4398ea0e957a87b8bffa4`).

FastAPI and its documentation are Copyright (c) 2018 Sebastián Ramírez and are
distributed under the MIT License, reproduced verbatim in `corpus/LICENSE`.

Modifications made by `scripts/fetch_corpus.py` (reproducible, deterministic):

- Only a subset of pages is included (99 pages: `tutorial/`, `advanced/`,
  `deployment/`, and `async.md`, `python-types.md`, `features.md`, `fastapi-cli.md`).
- mkdocs include directives (`{* ../../docs_src/... *}`) were replaced by the
  referenced example source files from the same release, as fenced Python blocks.
- Admonitions, content tabs, inline HTML and attribute lists were converted to
  plain Markdown; images were removed. "non-Annotated" duplicate code tabs were dropped.

This corpus is used here only as retrieval data for a demo / evaluation project.
It is not affiliated with or endorsed by the FastAPI project.
