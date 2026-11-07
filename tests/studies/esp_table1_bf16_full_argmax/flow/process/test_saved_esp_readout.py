"""Saved-count IO boundaries, publication and input identity; no model execution."""

import hashlib
import json
from pathlib import Path

import pytest

from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout


@pytest.mark.parametrize('fault', ['existing', 'inside', 'ancestor', 'checkout'])
def test_saved_esp_readout_rejects_output_overlap_and_existing_roots(tmp_path, fault):
    source, output = tmp_path / 'source', tmp_path / 'new'
    source.mkdir()
    if fault == 'existing':
        output.mkdir()
        (output / 'keep').write_bytes(b'old')
    elif fault == 'inside':
        output = source / 'new'
    elif fault == 'ancestor':
        output = tmp_path
    else:
        (tmp_path / 'foreign' / '.git').mkdir(parents=True)
        output = tmp_path / 'foreign' / 'new'
    with pytest.raises((ValueError, FileExistsError)):
        SavedReadout(source, output)
    assert not (source / 'new').exists()


@pytest.mark.parametrize('fault', ['input', 'new_pending_artifact', 'source'])
def test_saved_esp_readout_rejects_drift_before_any_output(tmp_path, monkeypatch, fault):
    source, output = tmp_path / 'source', tmp_path / 'new'
    source.mkdir()
    raw = source / 'raw.json'
    raw.write_text('{}')
    session = SavedReadout(source, output)
    session.read_json(raw)
    assert not session.exists(source / 'pending.json')
    session.write_text(output / 'protocol_evidence_v1.json', '{"quality_passed":true}')
    if fault == 'input':
        raw.write_text('{"changed":true}')
    elif fault == 'new_pending_artifact':
        (source / 'pending.json').write_text('{}')
    else:
        monkeypatch.setattr(session, '_source_files', lambda: {})
    with pytest.raises(ValueError, match='changed'):
        session.finish()
    assert not output.exists()


def test_saved_esp_readout_streams_sha_and_rejects_changed_resolved_alias(tmp_path, monkeypatch):
    source, output = tmp_path / 'source', tmp_path / 'new'
    source.mkdir()
    left, right, alias = source / 'left.json', source / 'right.json', source / 'alias.json'
    left.write_bytes(b'{}')
    right.write_bytes(b'{"other":true}')
    original_resolve, original_read = Path.resolve, Path.read_bytes
    selected = [left]
    def resolve(path, *args, **options):
        return selected[0] if path == alias else original_resolve(path, *args, **options)
    monkeypatch.setattr(Path, 'resolve', resolve)
    session = SavedReadout(source, output)
    def read(path):
        if path == left:
            raise AssertionError('sha must use the shared streaming implementation')
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    assert session.sha(alias) == hashlib.sha256(b'{}').hexdigest()
    selected[0] = right
    with pytest.raises(ValueError, match='changed'):
        session.finish()
    assert not output.exists()


def test_saved_esp_readout_render_failure_leaves_only_partial_ready_evidence(tmp_path, monkeypatch):
    import importlib
    plot = importlib.import_module('studies.esp_table1_bf16_full_argmax.flow.write.run')
    source, output = tmp_path / 'source', tmp_path / 'new'
    source.mkdir()
    raw = source / 'raw.json'
    raw.write_text('{}')
    session = SavedReadout(source, output)
    session.read_json(raw)
    session.write_text(output / 'protocol_evidence_v1.json', json.dumps({'quality_passed': True}))
    def fail(*args):
        raise OSError('plot write failed')
    monkeypatch.setattr(plot, 'write_plot', fail)
    with pytest.raises(OSError, match='plot write failed'):
        session.finish(figure={})
    assert (output / 'protocol_evidence_v1.json.partial').is_file()
    assert not (output / 'protocol_evidence_v1.json').exists()
    assert not (output / 'readout_inputs.json').exists()
