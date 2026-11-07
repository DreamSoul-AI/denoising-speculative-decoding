"""CPU saved endpoint IO, source boundaries and the original continuation mapping."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml

reader = importlib.import_module('studies.qwen3_4b_bf16_bias_norm_scan.flow.process.run')
ROOT = Path(__file__).resolve().parents[5]
COMMIT = 'b8b20b5ada7fd860894114cbf224ce749af430cf'


def test_actual_reused_report_reaggregates_without_native_acceptance(historical_file, tmp_path):
    proof = historical_file('studies/qwen3_4b_bf16_bias_norm_scan/docs/reused_readiness_v1.json')
    direction = historical_file('studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json')
    output = tmp_path/'readout'
    body = reader.published(proof, direction, ROOT, COMMIT, hashlib.sha256(proof.read_bytes()).hexdigest(),
        hashlib.sha256(direction.read_bytes()).hexdigest(), output)
    assert body['raw_argmax_checked_tokens'] == 405083
    scope = json.loads((output/'counter_scope.json').read_bytes())
    assert scope['cells_checked'] == 5 and scope['native_run_revalidated'] is False
    assert scope['raw_files_revalidated'] is False
    with pytest.raises(FileExistsError):
        reader.published(proof, direction, ROOT, COMMIT, hashlib.sha256(proof.read_bytes()).hexdigest(),
            hashlib.sha256(direction.read_bytes()).hexdigest(), output)


def test_native_output_cannot_enter_the_separate_continuation_input_root(tmp_path):
    data, study, continuation = (tmp_path/name for name in ('data', 'study', 'continuation'))
    for folder in (data, study, continuation):
        folder.mkdir()
    output = continuation/'new-output'
    with pytest.raises(ValueError, match='separate'):
        reader.native_readout(data, study, output, continuation)
    assert not output.exists()


@pytest.mark.parametrize('fault', ['plot_failure', 'input_drift'])
def test_native_transaction_does_not_publish_complete_after_plot_failure_or_input_drift(tmp_path, fault):
    from studies.qwen3_4b_bf16_paired_revalidation.flow.process.run import PublishedReadout
    inputs = tmp_path/'inputs'; inputs.mkdir()
    source = inputs/'source.json'; source.write_text('{}', encoding='utf-8')
    output = tmp_path/'output'
    session = PublishedReadout(inputs, output, ROOT)
    session.read_json(source)
    target = output/'norm_evidence_v1.json'
    session.write_text(target, '{"complete": true}\n')
    def render(folder):
        raise RuntimeError('injected plot failure')
    if fault == 'input_drift':
        source.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises((RuntimeError, ValueError)):
        session.finish(render=render)
    assert not target.exists() and not (output/'readout_inputs.json').exists()


def test_seven_cells_are_exact_original_tail_with_new_identity():
    old = ROOT / 'studies/qwen3_4b_bf16_bias_norm_scan'
    new = ROOT / 'studies/qwen3_4b_bf16_bias_norm_continuation'
    read = lambda p: yaml.safe_load(p.read_text(encoding='utf-8'))
    before, after = read(old / 'study.yaml'), read(new / 'study.yaml')
    assert after['axes']['algorithm.config'] == before['axes']['algorithm.config'][1:]
    before, after = read(old / 'experiment_config.yaml'), read(new / 'experiment_config.yaml')
    assert after['algorithm']['config']['version'] != before['algorithm']['config']['version']
    after['experiment'] = before['experiment']
    after['algorithm']['config']['version'] = before['algorithm']['config']['version']
    assert after == before



def test_historical_continuation_original_source_refs(historical_file):
    new = historical_file('studies/qwen3_4b_bf16_bias_norm_continuation/docs/source_refs.json').parents[1]
    refs = json.loads((new / 'docs/source_refs.json').read_text(encoding='utf-8'))
    assert refs['original_recovered_run'] == '6ee3f749ee20c450'



@pytest.mark.parametrize('change', ['none', 'prompt', 'axis', 'executed_old'])
def test_continuation_mapping_preserves_contract_and_old_history(monkeypatch, tmp_path, change):
    from spec_mtp.structure.control.merge import deep_merge
    from studies._shared.structure import sources
    from spec_mtp.structure.artifact import provenance
    reader = importlib.import_module('studies.qwen3_4b_bf16_bias_norm_scan.flow.process.run')
    monkeypatch.setattr(sources, 'check_frozen', lambda *a: None)
    monkeypatch.setattr(sources, 'manifest', lambda: {})
    monkeypatch.setattr(provenance, 'check_frozen', lambda *a: None)
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); new.mkdir()
    decl = {'axes': {'algorithm.config': [{'direction': i} for i in range(8)]}}
    new_decl = {'axes': {'algorithm.config': decl['axes']['algorithm.config'][1:]}}
    before = {'experiment': 'old', 'algorithm': {'config': {'version': 'v1', 'eval': {'max_new_tokens': 256}}}}
    after = {'experiment': 'new', 'algorithm': {'config': {'version': 'v2', 'eval': {'max_new_tokens': 256}}}}
    if change == 'prompt':
        after['algorithm']['config']['eval']['max_new_tokens'] = 128
    if change == 'axis':
        new_decl['axes']['algorithm.config'] = list(reversed(new_decl['axes']['algorithm.config']))
    for folder, value in [(old, before), (new, after)]:
        (folder / 'experiment_config.yaml').write_text(yaml.safe_dump(value))
    (new / 'study.yaml').write_text(yaml.safe_dump(new_decl))
    entries = [{'runs': [{'id': str(i)}]} for i in range(8)]
    newer = [{'factors': {'algorithm.config': deep_merge(after['algorithm']['config'], a)}, 'runs': [{'id': 'n'+str(i)}]} for i,a in enumerate(new_decl['axes']['algorithm.config'])]
    (new / 'index.json').write_text(json.dumps({'experiments': newer}))
    first = old / 'runs/0'; first.mkdir(parents=True)
    (first / 'result.json').write_text(json.dumps({'status': 'succeeded', 'spec_mtp': {'revalidation': {'execute_rerun': False}}}))
    if change == 'executed_old':
        folder = old / 'runs/1/assets'; folder.mkdir(parents=True)
        (folder / 'decode_observations.json').write_text('{}')
    if change != 'none':
        with pytest.raises(AssertionError):
            reader.continuation_entries(old, decl, entries, new)
    else:
        selected = reader.continuation_entries(old, decl, entries, new)
        assert len(selected) == 8 and selected[0] == (old, entries[0])
        assert all(folder == new for folder, entry in selected[1:])
