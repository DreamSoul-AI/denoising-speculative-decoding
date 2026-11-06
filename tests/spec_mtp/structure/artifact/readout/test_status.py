"""Bounded diagnostic parsing preserves historical record contracts."""
import io
import json
from pathlib import Path

import pytest

from tests.conftest import saved_diagnostic as _saved_diagnostic
from spec_mtp.structure.artifact.readout.status import diagnostic_lines, _note_from_log
from spec_mtp.structure.artifact.readout import status as status_readout
from spec_mtp.structure.artifact.readout.logs import format_logs


@pytest.mark.parametrize('reader', ['lines', 'note', 'logs'])
def test_diagnostic_readers_use_bounded_streams_without_whole_file_reads(tmp_path, monkeypatch, reader):
    path = tmp_path / 'runs/r1/assets/logs/events.jsonl'
    path.parent.mkdir(parents=True)
    raw = _saved_diagnostic('first') + _saved_diagnostic('last')
    path.write_text(raw, encoding='utf-8')
    (tmp_path / 'index.json').write_text(json.dumps({'experiments': [{'runs': [{'id': 'r1'}]}]}))
    original_open, original_read_text = Path.open, Path.read_text
    limits = []

    class BoundedStream(io.StringIO):
        def readline(self, size=-1):
            assert size == 65537, 'JSONL records must be read with a finite character bound'
            limits.append(size)
            return super().readline(size)

        def read(self, *args):
            raise AssertionError('whole-file diagnostic reads are forbidden')

        def __iter__(self):
            raise AssertionError('unbounded physical-line iteration is forbidden for JSONL')

    def opened(candidate, *args, **kwargs):
        return BoundedStream(raw) if candidate == path else original_open(candidate, *args, **kwargs)

    def read_text(candidate, *args, **kwargs):
        assert candidate != path, 'diagnostic reads must not call read_text'
        return original_read_text(candidate, *args, **kwargs)

    def no_list_wrapper(*args, **kwargs):
        raise AssertionError('stream consumers must not materialize diagnostic_lines')

    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(Path, 'read_text', read_text)
    if reader == 'lines':
        assert len(diagnostic_lines(path)) == 2
    else:
        monkeypatch.setattr(status_readout, 'diagnostic_lines', no_list_wrapper)
        if reader == 'note':
            assert _note_from_log(path) == '[flow] last'
        else:
            assert '[flow] first' in format_logs(tmp_path) and '[flow] last' in format_logs(tmp_path)
    assert limits


@pytest.mark.parametrize('length,terminated', [(65536, True), (65537, True), (65535, False)])
def test_diagnostic_record_character_limit_includes_terminating_linefeed(tmp_path, length, terminated):
    path = tmp_path / 'events.jsonl'
    raw = _saved_diagnostic('')
    raw = raw.replace('"message": ""', '"message": "' + 'x' * (length - len(raw)) + '"')
    assert len(raw) == length
    if not terminated:
        raw = raw[:-1]
    path.write_text(raw, encoding='utf-8', newline='\n')
    if length == 65536 and terminated:
        assert len(diagnostic_lines(path)) == 1
    else:
        with pytest.raises(ValueError, match='truncated or oversized'):
            diagnostic_lines(path)
    assert path.read_bytes() == raw.encode('utf-8')


@pytest.mark.parametrize('separator', ['\v', '\f', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029'])
def test_json_diagnostics_reject_embedded_splitlines_separators(tmp_path, separator):
    path = tmp_path / 'events.jsonl'
    raw = _saved_diagnostic().replace('saved', 'before' + separator + 'after')
    path.write_text(raw, encoding='utf-8', newline='\n')
    with pytest.raises(ValueError, match='truncated or oversized'):
        diagnostic_lines(path)
    assert path.read_bytes() == raw.encode('utf-8')


@pytest.mark.parametrize('terminator', ['\n', '\r\n', '\r'])
def test_legacy_stream_preserves_splitlines_replacement_and_unbounded_old_records(tmp_path, monkeypatch, terminator):
    path = tmp_path / 'run.log'
    raw = ('old' + '\v\f\x1c\x1d\x1e\x85\u2028\u2029' + 'x' * 70000 + terminator + 'last').encode() + b'\xff'
    path.write_bytes(raw)
    original = Path.read_text

    def read_text(candidate, *args, **kwargs):
        assert candidate != path, 'legacy diagnostics must stream physical lines'
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    assert diagnostic_lines(path) == raw.decode('utf-8', errors='replace').splitlines()
    assert path.read_bytes() == raw
