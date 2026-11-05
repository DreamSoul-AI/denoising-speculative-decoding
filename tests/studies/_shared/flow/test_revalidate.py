"""Dedicated revalidation checks immutable Study inputs before importing policies."""

import importlib
import json
from types import SimpleNamespace

import pytest

from studies._shared.flow import cli, revalidate


def test_validator_requires_a_loadable_python_module(tmp_path):
    from studies._shared.flow.revalidate import load_validator

    with pytest.raises(ImportError, match='loadable Python module'):
        load_validator(tmp_path / 'validator.unknown')


def _invoke(entry, study, validator):
    if entry == 'run':
        return revalidate.revalidate_run(study, 'saved-run', validator)
    return revalidate.process_study(study, validator)


@pytest.mark.parametrize('entry', ['run', 'process'])
def test_revalidation_rejects_source_checkout_before_validator_or_writes(tmp_path, monkeypatch, entry):
    checkout = tmp_path / 'checkout'
    (checkout / '.git').mkdir(parents=True)
    study = checkout / 'study'
    study.mkdir()
    (study / 'study.yaml').write_text('study: test\n', encoding='utf-8')
    before = {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob('*') if path.is_file()}
    monkeypatch.setattr(revalidate, 'load_validator', lambda path: pytest.fail('validator must not be imported'))
    with pytest.raises(ValueError, match='source checkout'):
        _invoke(entry, study, tmp_path / 'untrusted-policy.py')
    assert before == {path.relative_to(checkout): path.read_bytes() for path in checkout.rglob('*') if path.is_file()}
    assert not (study / 'runs').exists() and not (study / 'shared').exists()


@pytest.mark.parametrize('entry', ['run', 'process'])
def test_revalidation_source_snapshot_drift_precedes_validator(tmp_path, monkeypatch, entry):
    from spec_mtp.structure.artifact.layout import materialize_study

    source = tmp_path / 'source'
    source.mkdir()
    (source / 'study.yaml').write_text('study: test\n', encoding='utf-8')
    study = materialize_study(source, tmp_path / 'runtime')
    (source / 'study.yaml').write_text('study: changed\n', encoding='utf-8')
    before = {path.relative_to(study): path.read_bytes() for path in study.rglob('*') if path.is_file()}
    monkeypatch.setattr(revalidate, 'load_validator', lambda path: pytest.fail('validator must not be imported'))
    with pytest.raises(ValueError, match='changed'):
        _invoke(entry, study, tmp_path / 'untrusted-policy.py')
    assert before == {path.relative_to(study): path.read_bytes() for path in study.rglob('*') if path.is_file()}
    assert not (study / 'runs').exists()


@pytest.mark.parametrize('entry', ['run', 'process'])
@pytest.mark.parametrize('guard', ['shared_sources', 'host_plan'])
def test_revalidation_reuses_shared_preflight_before_loading_policy(tmp_path, monkeypatch, entry, guard):
    from spec_mtp.structure.artifact import provenance
    from studies._shared.structure import sources

    (tmp_path / 'study.yaml').write_text('study: test\nfreeze: true\n', encoding='utf-8')
    before = (tmp_path / 'study.yaml').read_bytes()
    seen = []

    def reject(study, *args):
        seen.append(study)
        raise RuntimeError(guard)

    monkeypatch.setattr(sources, 'manifest', lambda: {})
    monkeypatch.setattr(sources, 'check_frozen', reject if guard == 'shared_sources' else lambda *args: None)
    monkeypatch.setattr(provenance, 'check_frozen', reject if guard == 'host_plan' else lambda *args: None)
    monkeypatch.setattr(revalidate, 'load_validator', lambda path: pytest.fail('validator must not be imported'))
    with pytest.raises(RuntimeError, match=guard):
        _invoke(entry, tmp_path, tmp_path / 'untrusted-policy.py')
    assert seen == [tmp_path.resolve()]
    assert (tmp_path / 'study.yaml').read_bytes() == before
    assert list(tmp_path.iterdir()) == [tmp_path / 'study.yaml']


@pytest.mark.parametrize('failure', [False, True])
def test_revalidation_process_restores_evidence_checker_after_guarded_dispatch(tmp_path, monkeypatch, failure):
    process = importlib.import_module('studies._shared.flow.process.run')
    host = importlib.import_module('spec_mtp.flow.process')
    original = process.verify_evidence
    replacement = lambda *args: None
    calls = []
    monkeypatch.setattr(cli, 'check_study_inputs', lambda study: calls.append(('check', study)))
    monkeypatch.setattr(revalidate, 'load_validator', lambda path: SimpleNamespace(verify_evidence=replacement))

    def aggregate(study):
        assert calls == [('check', tmp_path.resolve())]
        assert process.verify_evidence is replacement
        if failure:
            raise RuntimeError('injected aggregate failure')
        return {'study': str(study)}

    monkeypatch.setattr(host, 'run_study', aggregate)
    if failure:
        with pytest.raises(RuntimeError, match='injected aggregate failure'):
            revalidate.process_study(tmp_path, tmp_path / 'policy.py')
    else:
        assert revalidate.process_study(tmp_path, tmp_path / 'policy.py') == {'study': str(tmp_path.resolve())}
    assert process.verify_evidence is original


def test_revalidation_keeps_failure_and_execute_bytes_and_is_idempotent(tmp_path,monkeypatch, saved_decode_study):
    from pathlib import Path
    from studies._shared.structure import validation
    ctx=saved_decode_study(tmp_path,[('a','failed',False,'d')])
    run=ctx.study_dir/'runs/run_a'
    body=json.loads((run/'result.json').read_text())
    body.update(control={},metrics={},error='SpecMtpGateError: old AR gate')
    (run/'result.json').write_text(json.dumps(body))
    previous=(run/'result.json').read_bytes()
    raw=(run/'assets/decode_observations.json').read_bytes()
    import spec_mtp.structure.algorithm.decode.loop as decoding
    monkeypatch.setattr(decoding.SpecDecoder,'generate',lambda *a:pytest.fail('execute must not rerun'))
    result=revalidate.revalidate_run(ctx.study_dir,'run_a',Path(validation.__file__))
    assert result['status']=='succeeded' and result['spec_mtp']['argmax_all_passed']
    migration=result['spec_mtp']['revalidation']
    assert migration['execute_rerun'] is False
    assert (run/'assets'/migration['previous_result_asset']).read_bytes()==previous
    assert (run/'assets/decode_observations.json').read_bytes()==raw
    saved=(run/'result.json').read_bytes()
    revalidate.revalidate_run(ctx.study_dir,'run_a',Path(validation.__file__))
    assert (run/'result.json').read_bytes()==saved
