"""studies/_shared/structure/decode_fingerprint.py compare logic (the full fingerprint runs manually, ~40 s)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('decode_fingerprint', ROOT / 'studies' / '_shared' / 'structure' / 'decode_fingerprint.py')
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def test_compare_reports_field_level_differences():
    old = {'cells': {'a': {'tokens': [1, 2], 'accepted': 1}, 'b': {'tokens': [3]}}}
    new = {'cells': {'a': {'tokens': [1, 2], 'accepted': 2}, 'c': {'tokens': [3]}}}
    assert fp.compare(old, old) == []
    assert fp.compare(old, new) == ["a: ['accepted']", "b: ['tokens']", "c: ['tokens']"]
