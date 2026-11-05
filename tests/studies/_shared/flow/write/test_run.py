"""The shared historical report writer binds real files and never publishes partial work."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

writer = importlib.import_module('studies._shared.flow.write.run')
COMMIT = 'b8b20b5ada7fd860894114cbf224ce749af430cf'
SHA = '0a38ab61aa0deee9ad84bf527323bb797b59b6dd51954e6be6205e4e5d8f9b0d'


@pytest.fixture(scope='session')
def original_bytes():
    from spec_mtp.structure.artifact.provenance import read_historical_bytes
    return read_historical_bytes(Path(__file__).resolve().parents[5], COMMIT,
        'studies/output_space_4b_holdout/docs/execution_v1.json', SHA)


@pytest.fixture
def report_case(tmp_path, monkeypatch, original_bytes):
    from spec_mtp.structure.artifact import provenance
    path, repository = tmp_path / 'inputs/execution.json', tmp_path / 'repository'
    path.parent.mkdir()
    repository.mkdir()
    path.write_bytes(original_bytes.replace(b'\n', b'\r\n'))
    bindings = path.parent / 'bindings.json'
    bindings.write_text(json.dumps({'execution': {'path': str(path), 'commit': COMMIT, 'sha256': SHA}}))
    archived = [original_bytes]
    def read(repo, commit, member, sha):
        assert repo == repository and commit == COMMIT
        assert member == 'studies/output_space_4b_holdout/docs/execution_v1.json'
        if hashlib.sha256(archived[0]).hexdigest() != sha:
            raise ValueError('historical report hash differs')
        return archived[0]
    monkeypatch.setattr(provenance, 'read_historical_bytes', read)
    return SimpleNamespace(path=path, repository=repository, bindings=bindings, archived=archived, output=tmp_path / 'export')


def _write(case):
    return writer.write_paired_report('output_space_4b_holdout', case.bindings, case.repository, case.output)


def test_paired_report_links_actual_input_and_preserves_unknown_references(report_case):
    case = report_case
    before = case.path.read_bytes()
    target = _write(case)
    text = target.read_text(encoding='utf-8')
    assert '[原生证据](<../inputs/execution.json>)' in text
    assert '历史引用，本次未提供对应输入' in text and '](' not in text.replace('[原生证据](<../inputs/execution.json>)', '').replace('![实际配对接受数与调用数](paired_counts_v1.png)', '')
    receipt = json.loads((case.output / 'report_complete.json').read_text())
    assert receipt['historical_inputs'][0]['sha256'] == SHA
    assert receipt['runtime_inputs']['execution']['sha256'] == hashlib.sha256(before).hexdigest() != SHA
    assert receipt['rendering_environment']['backend'] == 'Agg' and receipt['rendering_sources']
    assert (case.output / 'paired_counts_v1.png').read_bytes().startswith(b'\x89PNG')
    assert case.path.read_bytes() == before


@pytest.mark.parametrize('fault', ['missing', 'hash', 'content', 'quality', 'timing', 'source', 'unknown_role'])
def test_paired_report_rejects_unbound_inputs_before_output(report_case, fault):
    case = report_case
    if fault == 'missing': case.path.unlink()
    elif fault == 'hash':
        binding = json.loads(case.bindings.read_text()); binding['execution']['sha256'] = '0' * 64
        case.bindings.write_text(json.dumps(binding))
    elif fault == 'unknown_role':
        binding = json.loads(case.bindings.read_text()); binding['other'] = binding['execution']
        case.bindings.write_text(json.dumps(binding))
    else:
        data = json.loads(case.archived[0])
        if fault == 'content': data['changed'] = True
        if fault == 'quality': data['quality_passed'] = False
        if fault == 'timing': data['decoding_timing_disabled'] = False
        if fault == 'source': data['source_digest'] = 'unknown'
        raw = json.dumps(data).encode(); case.path.write_bytes(raw)
        if fault != 'content':
            case.archived[0] = raw
            binding = json.loads(case.bindings.read_text()); binding['execution']['sha256'] = hashlib.sha256(raw).hexdigest()
            case.bindings.write_text(json.dumps(binding))
    with pytest.raises((ValueError, FileNotFoundError)):
        _write(case)
    assert not case.output.exists()


@pytest.mark.parametrize('placement', ['inputs', 'repository', 'existing'])
def test_paired_report_refuses_output_overlap_and_prior_outputs(report_case, placement):
    case = report_case
    case.output = {'inputs': case.path.parent / 'out', 'repository': case.repository / 'out', 'existing': case.output}[placement]
    if placement == 'existing': case.output.mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        _write(case)


@pytest.mark.parametrize('drift', ['input', 'source'])
def test_paired_report_drift_after_save_leaves_partial_without_completion(report_case, monkeypatch, drift):
    from studies._shared.structure import paired_report
    case = report_case
    fake = case.output.parent / 'fake-render.py'; fake.write_text('# original')
    monkeypatch.setattr(paired_report, '__file__', str(fake))
    native = paired_report.paired_counts_figure
    def figure(data):
        result = native(data); save = result.savefig
        def mutate(*args, **kwargs):
            save(*args, **kwargs); (case.path if drift == 'input' else fake).write_text('changed')
        result.savefig = mutate
        return result
    monkeypatch.setattr(paired_report, 'paired_counts_figure', figure)
    with pytest.raises(ValueError, match='changed'):
        _write(case)
    assert (case.output / 'paired_counts_v1.png').exists() and not (case.output / 'report_complete.json').exists()


@pytest.mark.parametrize('study', ['output_space_4b_holdout', 'output_space_4b_recheck_v2'])
def test_paired_report_study_commands_delegate_their_actual_identity(monkeypatch, study):
    calls = []
    monkeypatch.setattr(writer, 'paired_report_main', lambda selected, argv: calls.append((selected, argv)) or 0)
    module = importlib.import_module(f'studies.{study}.flow.write.run')
    assert module.main(['--help']) == 0 and calls == [(study, ['--help'])]


@pytest.mark.parametrize('missing', ['--bindings', '--repository', '--output-root'])
def test_paired_report_cli_requires_all_explicit_bindings(tmp_path, monkeypatch, missing):
    values = {'--bindings': str(tmp_path / 'bindings.json'), '--repository': str(tmp_path), '--output-root': str(tmp_path / 'out')}
    monkeypatch.setattr(writer, 'write_paired_report', lambda *a: pytest.fail('invalid arguments dispatched'))
    with pytest.raises(SystemExit) as caught:
        writer.paired_report_main('output_space_4b_holdout', [p for k, v in values.items() if k != missing for p in (k, v)])
    assert caught.value.code == 2


@pytest.mark.parametrize('fault', ['write', 'existing'])
def test_completion_publication_failure_never_deletes_prior_receipts_or_keeps_a_new_partial(tmp_path, monkeypatch, fault):
    path = tmp_path / 'complete.json'
    if fault == 'existing':
        path.write_text('old receipt')
        with pytest.raises(FileExistsError):
            writer.publish_new_completion(path, '{"complete":true}')
        assert path.read_text() == 'old receipt'
        return
    native = Path.open
    class FailedStream:
        def __enter__(self):
            self.stream = native(path, 'x', encoding='utf-8'); return self
        def write(self, payload):
            self.stream.write(payload); self.stream.flush()
            raise OSError('injected receipt write failure')
        def __exit__(self, *args):
            self.stream.close()
    monkeypatch.setattr(Path, 'open', lambda selected, *a, **kw: FailedStream() if selected == path else native(selected, *a, **kw))
    with pytest.raises(OSError, match='injected receipt'):
        writer.publish_new_completion(path, '{"complete":true}')
    assert not path.exists()


@pytest.mark.parametrize('fault', ['receipt_input', 'receipt_source', 'clear_input'])
def test_paired_report_receipt_stage_drift_prevents_completion(report_case, monkeypatch, fault):
    from studies._shared.structure import paired_report
    case = report_case
    fake_source = case.output.parent/'fake-render-source.py'
    fake_source.write_text('# original controlled source', encoding='utf-8')
    monkeypatch.setattr(paired_report, '__file__', str(fake_source))
    cleared = []
    def savefig(stream, **kwargs):
        stream.write(b'controlled lifecycle stub; not a rendered Figure')
    def clear():
        cleared.append(True)
        if fault == 'clear_input': case.path.write_text('changed', encoding='utf-8')
    monkeypatch.setattr(paired_report, 'paired_counts_figure',
                        lambda data: SimpleNamespace(savefig=savefig, clear=clear))
    dumps = json.dumps
    def receipt_dumps(value, *args, **kwargs):
        result = dumps(value, *args, **kwargs)
        if isinstance(value, dict) and 'rendering_sources' in value and fault != 'clear_input':
            (case.path if fault == 'receipt_input' else fake_source).write_text('changed', encoding='utf-8')
        return result
    monkeypatch.setattr(json, 'dumps', receipt_dumps)
    with pytest.raises(ValueError, match='changed'):
        _write(case)
    assert cleared == [True] and (case.output/'paired_report.md').is_file()
    assert not (case.output/'report_complete.json').exists()


@pytest.mark.parametrize('fault', ['clear_only', 'save_and_clear', 'verify_and_clear'])
def test_paired_report_cleanup_preserves_primary_and_blocks_completion(report_case, monkeypatch, fault):
    from spec_mtp.structure.artifact import provenance
    from studies._shared.structure import paired_report
    case = report_case
    primary, cleanup = OSError('controlled primary failure'), RuntimeError('controlled Figure cleanup failure')
    def savefig(stream, **kwargs):
        stream.write(b'controlled lifecycle stub; not a rendered Figure')
        if fault == 'save_and_clear': raise primary
    def clear():
        raise cleanup
    monkeypatch.setattr(paired_report, 'paired_counts_figure',
                        lambda data: SimpleNamespace(savefig=savefig, clear=clear))
    native_hash = provenance.file_sha256
    def source_hash(path):
        if fault == 'verify_and_clear' and case.output.exists() and Path(path) == case.path:
            raise primary
        return native_hash(path)
    monkeypatch.setattr(provenance, 'file_sha256', source_hash)
    expected = cleanup if fault == 'clear_only' else primary
    with pytest.raises(type(expected)) as caught:
        _write(case)
    assert caught.value is expected
    if fault != 'clear_only': assert caught.value.figure_cleanup_error is cleanup
    assert not (case.output/'report_complete.json').exists()
