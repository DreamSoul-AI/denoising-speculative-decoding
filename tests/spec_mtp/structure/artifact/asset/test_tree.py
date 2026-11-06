"""Diagnostic part layout admission reads metadata only."""
from pathlib import Path
import pytest
from spec_mtp.structure.artifact.asset.tree import diagnostic_files
from tests.conftest import saved_diagnostic


@pytest.mark.parametrize('layout', ['gap', 'duplicate', 'missing_anchor', 'nonfile',
                                   'escape', 'duplicate_identity', 'inbounds_alias'])
def test_diagnostic_segments_validate_named_layout_without_reading_bodies(tmp_path, monkeypatch, layout):
    parent = tmp_path / 'logs'
    parent.mkdir()
    anchor = parent / 'events.jsonl'
    if layout != 'missing_anchor': anchor.write_text(saved_diagnostic())
    part = parent / ('events.000002.jsonl' if layout == 'gap' else 'events.000001.jsonl')
    if layout == 'nonfile': part.mkdir()
    else: part.write_text(saved_diagnostic('part'))
    if layout == 'duplicate': (parent / 'events.1.jsonl').write_text(saved_diagnostic('duplicate'))
    outside = tmp_path / 'outside.jsonl'
    outside.write_text(saved_diagnostic('outside'))
    history = parent / 'historical.jsonl'
    history.write_text(saved_diagnostic('history'))
    resolve, stat = Path.resolve, Path.stat
    destination = outside if layout == 'escape' else history
    def resolved(path, *args, **kwargs):
        return destination if path == part and layout in ('escape', 'inbounds_alias') else resolve(path, *args, **kwargs)
    original_identity = anchor.stat() if anchor.exists() else None
    history_identity = history.stat()
    def stated(path, *args, **kwargs):
        if path == part and layout == 'duplicate_identity': return original_identity
        if path == part and layout == 'inbounds_alias': return history_identity
        return stat(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'resolve', resolved)
        patch.setattr(Path, 'stat', stated)
        patch.setattr(Path, 'open', lambda *a, **k: pytest.fail('enumeration must not read byte bodies'))
        with pytest.raises(ValueError): diagnostic_files(anchor)
