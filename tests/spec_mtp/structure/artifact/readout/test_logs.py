"""Study event aggregation rejects partial Run reads."""
import json

from tests.conftest import saved_diagnostic as _saved_diagnostic
from spec_mtp.structure.artifact.readout.status import diagnostic_lines, _note_from_log
from spec_mtp.structure.artifact.readout import status as status_readout
from spec_mtp.structure.artifact.readout import logs as logs_readout
from spec_mtp.structure.artifact.readout.logs import format_logs


def test_midstream_io_failure_discards_that_runs_partial_events(tmp_path, monkeypatch):
    paths = {}
    for run_id in ('first', 'broken', 'last'):
        path = tmp_path / f'runs/{run_id}/assets/logs/events.jsonl'
        path.parent.mkdir(parents=True)
        path.write_text(_saved_diagnostic(run_id), encoding='utf-8')
        paths[run_id] = path
    (tmp_path / 'index.json').write_text(json.dumps({
        'experiments': [{'runs': [{'id': run_id} for run_id in paths]}],
    }))
    actual = getattr(status_readout, 'iter_diagnostic_lines', lambda path: iter(diagnostic_lines(path)))

    def interrupted(path):
        yield from actual(path)
        if path == paths['broken']:
            raise OSError('injected after one valid event')

    monkeypatch.setattr(status_readout, 'iter_diagnostic_lines', interrupted, raising=False)
    monkeypatch.setattr(logs_readout, 'iter_diagnostic_lines', interrupted, raising=False)
    assert _note_from_log(paths['broken']) == '[error] diagnostic log unavailable'
    lines = format_logs(tmp_path).splitlines()
    assert len(lines) == 2 and '[flow] first' in lines[0] and '[flow] last' in lines[1]
