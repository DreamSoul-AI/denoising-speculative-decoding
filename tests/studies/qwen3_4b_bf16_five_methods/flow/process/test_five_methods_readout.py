"""Saved-counter admission rejects changed source bytes and output reuse."""
import json
from pathlib import Path

import pytest

from studies.qwen3_4b_bf16_five_methods.flow.process.run import write_readout

ROOT = Path(__file__).resolve().parents[5]
COMMIT = 'b8b20b5ada7fd860894114cbf224ce749af430cf'
MEMBER = 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json'
SHA = '8e85ac6ea9984047abf0ffd5d6653b28b6aadd74aa876139fa10465dc5bbf274'


def test_saved_counter_readout_preserves_original_identity_and_refuses_overwrite(historical_file, tmp_path):
    source = historical_file(MEMBER)
    before = source.read_bytes()
    output = tmp_path / 'readout'
    target = write_readout(source, ROOT, COMMIT, SHA, output)
    receipt = json.loads(target.read_bytes())
    assert receipt['native_run_revalidated'] is False
    assert receipt['historical_input'] == {'commit': COMMIT, 'path': MEMBER, 'sha256': SHA}
    assert (receipt['cells_checked'], receipt['pairs_checked']) == (11, 20)
    assert receipt['original_report_identity'] == {
        key: json.loads(before)[key] for key in ('analyzer_sha256', 'plotter_sha256')}
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        write_readout(source, ROOT, COMMIT, SHA, output)


@pytest.mark.parametrize('damage', ['git_hash', 'runtime_fields', 'runtime_identity', 'source_output'])
def test_saved_counter_admission_rejects_drift_without_output(historical_file, tmp_path, damage):
    source = historical_file(MEMBER)
    local = tmp_path / 'input' / 'report.json'
    local.parent.mkdir()
    body = json.loads(source.read_bytes())
    if damage == 'runtime_fields':
        body['archival_annotation'] = 'cannot silently permit added fields'
    elif damage == 'runtime_identity':
        body['analyzer_sha256'] = '0' * 64
    local.write_text(json.dumps(body), encoding='utf-8')
    output = ROOT / 'studies' / 'forbidden-five-methods-output' if damage == 'source_output' else tmp_path / 'output'
    with pytest.raises(ValueError):
        write_readout(local, ROOT, COMMIT, '0' * 64 if damage == 'git_hash' else SHA, output)
    assert not output.exists()


def test_readout_rejects_helper_changed_during_verification(historical_file, tmp_path, monkeypatch):
    from studies.qwen3_4b_bf16_five_methods.structure import readout
    fake = tmp_path / 'fake_helper.py'
    fake.write_text('before', encoding='utf-8')
    monkeypatch.setattr(readout, '__file__', str(fake))
    original = readout.verify

    def change(path):
        result = original(path)
        fake.write_text('after', encoding='utf-8')
        return result

    monkeypatch.setattr(readout, 'verify', change)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='source changed'):
        write_readout(historical_file(MEMBER), ROOT, COMMIT, SHA, output)
    assert not output.exists()


@pytest.mark.parametrize('damage', ['missing_hashes', 'missing_file', 'unsafe_id', 'duplicate_id'])
def test_original_file_check_rejects_incomplete_or_unsafe_identity(historical_file, tmp_path, monkeypatch, damage):
    from spec_mtp.structure.artifact import provenance
    from studies.qwen3_4b_bf16_five_methods.structure import readout
    data = json.loads(historical_file(MEMBER).read_bytes())
    values = list(data['cells'].values())
    if damage == 'missing_hashes':
        values[0]['files_sha256'] = {}
    elif damage == 'missing_file':
        values[0]['files_sha256'].pop('result.json')
    elif damage == 'unsafe_id':
        values[0]['run_id'] = '../outside'
    else:
        values[0]['run_id'] = values[1]['run_id']
        # Reject duplicates before any real original files are required.
        values[0]['files_sha256'] = values[1]['files_sha256']
    raw = json.dumps(data).encode()
    local = tmp_path / 'input' / 'report.json'
    local.parent.mkdir()
    local.write_bytes(raw)
    monkeypatch.setattr(provenance, 'read_historical_bytes', lambda *args: raw)
    monkeypatch.setattr(readout, 'verify', lambda path: {})
    owner = tmp_path / 'original'
    owner.mkdir()
    output = tmp_path / 'output'
    with pytest.raises(ValueError):
        write_readout(local, ROOT, COMMIT, SHA, output, original_study=owner)
    assert not output.exists()


def test_original_file_check_binds_actual_files_without_native_revalidation(historical_file, tmp_path, monkeypatch):
    import hashlib
    from spec_mtp.structure.artifact import provenance
    data = json.loads(historical_file(MEMBER).read_bytes())
    owner = tmp_path / 'original'
    actual_files = []
    for cell in data['cells'].values():
        for member in cell['files_sha256']:
            file = owner / 'runs' / cell['run_id'] / member
            file.parent.mkdir(parents=True, exist_ok=True)
            # Synthetic byte fixture tests only the declared file-hash chain.
            payload = (cell['run_id'] + ':' + member).encode()
            file.write_bytes(payload)
            cell['files_sha256'][member] = hashlib.sha256(payload).hexdigest()
            actual_files.append(file)
    raw = json.dumps(data).encode()
    source = tmp_path / 'input' / 'report.json'
    source.parent.mkdir()
    source.write_bytes(raw)
    monkeypatch.setattr(provenance, 'read_historical_bytes', lambda *args: raw)
    receipt = json.loads(write_readout(source, ROOT, COMMIT, SHA, tmp_path / 'output', original_study=owner).read_bytes())
    assert receipt['native_run_revalidated'] is False
    assert all(str(path.resolve()) in receipt['runtime_inputs'] for path in actual_files)
    actual_files[0].write_bytes(b'changed')
    with pytest.raises(ValueError, match='original Run file differs'):
        write_readout(source, ROOT, COMMIT, SHA, tmp_path / 'rejected', original_study=owner)
    assert not (tmp_path / 'rejected').exists()
