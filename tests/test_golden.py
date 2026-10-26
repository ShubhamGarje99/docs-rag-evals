"""The committed golden set must stay consistent with the vendored corpus and the chunker."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from ragkit.golden import QUESTION_TYPES, load_golden

ROOT = Path(__file__).resolve().parents[1]


def test_golden_schema():
    items = load_golden()
    assert len(items) >= 50
    assert {it.type for it in items} <= set(QUESTION_TYPES)
    unans = [it for it in items if it.type == "unanswerable"]
    assert 5 <= len(unans) <= 8 and all(not it.gold_chunk_ids for it in unans)
    assert all(1 <= len(it.gold_chunk_ids) <= 3 for it in items if it.type != "unanswerable")


def test_check_golden_passes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_golden.py"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "evals" / "check_golden.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert exc.value.code == 0, out
    assert "OK: 0 error(s)" in out
