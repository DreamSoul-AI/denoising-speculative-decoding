"""Actual-file hashing works without Python 3.11's file_digest API."""
import hashlib
from importlib import import_module

import pytest


@pytest.mark.parametrize('module,name', [
    ('spec_mtp.structure.artifact.provenance', 'file_sha256'),
    ('studies._shared.structure.evidence', 'file_hash'),
    ('studies._shared.structure.window_diagnostics', 'file_hash'),
    ('studies._shared.structure.geometry_4b', 'sha'),
    ('studies.output_space_small_matrix.analyze_transfers', 'sha'),
    ('studies.output_space_followup.analyze_results', 'sha'),
])
def test_file_hash_supports_python310_and_real_bytes(tmp_path, monkeypatch, module, name):
    payload = b'actual\x00content' * 100000
    path = tmp_path / 'source.bin'
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    actual_hash = getattr(import_module(module), name)
    assert actual_hash(path) == expected
    if module != 'studies._shared.structure.window_diagnostics':
        with pytest.raises(FileNotFoundError):
            actual_hash(tmp_path / 'missing.bin')
