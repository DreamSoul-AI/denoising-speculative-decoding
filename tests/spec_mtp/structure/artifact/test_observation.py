"""Untrusted observation envelopes fail as scientific verdicts."""



import pytest

from spec_mtp.structure.artifact.observation import audit_run


@pytest.mark.parametrize('raw', [[[], {'event': 'end'}], [None], [True], {}, ''])
def test_non_object_or_missing_envelope_is_rejected(raw):
    assert audit_run(raw, {}).argmax_coverage is False

def test_model_sources_reject_resolved_members_outside_root(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from spec_mtp.structure.artifact.observation import verify_model_files

    root = tmp_path / 'model'
    root.mkdir()
    member = root / 'config.json'
    member.write_bytes(b'{}')
    outside = tmp_path / 'external-config.json'
    original = Path.resolve
    # Simulate an external symlink resolution without requiring OS symlink privileges.
    monkeypatch.setattr(Path, 'resolve', lambda self, *a, **k: outside if self == member else original(self, *a, **k))
    with pytest.raises(ValueError, match='escapes'):
        verify_model_files(root, {'config.json': hashlib.sha256(b'{}').hexdigest()})
