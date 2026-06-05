"""Tests for the probe-file loaders."""
import json
from pathlib import Path

import pytest

from pathmark.probes import (
    load_probe_file,
    filter_by_char_count,
    filter_by_token_count,
)


def _write_json(tmp_path: Path, data) -> str:
    p = tmp_path / "probes.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_load_probe_file_returns_list_of_strings(tmp_path):
    p = _write_json(tmp_path, ["alpha", "beta", "gamma"])
    out = load_probe_file(p)
    assert out == ["alpha", "beta", "gamma"]


def test_load_probe_file_rejects_non_list(tmp_path):
    p = _write_json(tmp_path, {"not": "a list"})
    with pytest.raises(ValueError):
        load_probe_file(p)


def test_load_probe_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_probe_file("/non/existent/probes.json")


def test_filter_by_char_count():
    texts = ["short", "x" * 100, "x" * 1000]
    out = filter_by_char_count(texts, min_chars=10, max_chars=500)
    assert len(out) == 1
    assert len(out[0]) == 100


def test_filter_by_token_count_uses_tokenizer():
    class _Tok:
        def encode(self, t, add_special_tokens=False):
            return list(t)
    texts = ["aaa", "aaaaaaaaaa", "aa"]
    out = filter_by_token_count(texts, _Tok(), min_tokens=3, max_tokens=5)
    assert out == ["aaa"]
