"""Probe-file IO.

Probe files are JSON arrays of strings shipped under `probes/`. We keep the
loading helpers here (rather than in `pathmark.data`) so they're easy to
reuse from any eval module without pulling in tokenizer machinery.
"""
import json
from pathlib import Path
from typing import List


def load_probe_file(path: str) -> List[str]:
    """Read a JSON list-of-strings; raises if the file isn't well-formed."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Probe file not found: {p}")
    with p.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {p}, got {type(data).__name__}")
    return [str(x) for x in data if isinstance(x, (str, bytes))]


def filter_by_token_count(
    texts: List[str],
    tokenizer,
    min_tokens: int,
    max_tokens: int,
    limit: int = None,
) -> List[str]:
    """Drop strings whose tokenized length is outside [min_tokens, max_tokens]."""
    out = []
    for t in texts:
        t = (t or "").strip()
        if not t: continue
        n = len(tokenizer.encode(t, add_special_tokens=False))
        if n < min_tokens or n > max_tokens: continue
        out.append(t)
        if limit and len(out) >= limit:
            break
    return out


def filter_by_char_count(
    texts: List[str],
    min_chars: int = 0,
    max_chars: int = None,
    limit: int = None,
) -> List[str]:
    """Filter without tokenizing — handy for raw length cuts."""
    out = []
    for t in texts:
        n = len(t)
        if n < min_chars: continue
        if max_chars and n > max_chars: continue
        out.append(t)
        if limit and len(out) >= limit:
            break
    return out
