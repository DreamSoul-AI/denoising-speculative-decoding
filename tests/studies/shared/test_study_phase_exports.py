"""Native dispatcher smoke checks before freezing the next queues."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]

@pytest.mark.parametrize('name', ['qwen3_4b_bf16_paired_revalidation', 'qwen3_4b_bf16_bias_norm_continuation'])
@pytest.mark.parametrize('phase', ['collect', 'summarize', 'write', 'process'])
def test_native_phase_dispatches_exported_callable(monkeypatch, name, phase):
    from rpipe.flow.study import run_study_phase
    native = importlib.import_module('studies._shared.flow.' + phase)
    calls = []
    monkeypatch.setattr(native, 'run', lambda ctx: calls.append(ctx))
    folder = ROOT / 'studies' / name
    decl = yaml.safe_load((folder / 'study.yaml').read_text(encoding='utf-8'))
    ctx = SimpleNamespace()
    assert run_study_phase(folder, phase, ctx, decl)
    assert calls == [ctx]

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
    import importlib.util
    from rpipe.structure.control.merge import deep_merge
    from studies._shared.structure import sources
    from rpipe.structure.artifact import provenance
    path = ROOT / 'studies/qwen3_4b_bf16_bias_norm_scan/analyze_results.py'
    spec = importlib.util.spec_from_file_location('norm_mapping_test', path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
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
