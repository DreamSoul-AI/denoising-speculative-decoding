"""Native magnitude controls keep the shared mean, fixed direction and argmax gate."""

from tests.conftest import norm_control_module as module


import importlib

from pathlib import Path


import pytest


import yaml


ROOT = Path(__file__).resolve().parents[2]

def test_eight_preregistered_cells_and_immutable_inputs(monkeypatch):
    m = module(monkeypatch)
    decl = yaml.safe_load((m.STUDY / 'study.yaml').read_text(encoding='utf-8'))
    rows = decl['axes']['algorithm.config']
    assert [(r['direction'], r['spec']['ema']['step_scale']) for r in rows] == [
        (mode, scale) for mode in ('template', 'random0', 'random1', 'random2') for scale in (.5, 2.)]
    assert all(r['spec']['num_masks'] == 2 and r['spec']['block_complexity'] == 30 for r in rows)

def test_unmade_readout_cannot_claim_completed_scan(monkeypatch, tmp_path):
    from studies._shared.structure.sources import SpecMtpSourceChangedError
    m = module(monkeypatch)
    monkeypatch.syspath_prepend(str(m.STUDY))
    readout = importlib.import_module('studies.qwen3_4b_bf16_bias_norm_scan.flow.process.run')
    fake_root = tmp_path / 'checkout'
    unmade = fake_root / 'studies/unmade'
    unmade.mkdir(parents=True)
    (unmade / 'study.yaml').write_bytes((m.STUDY / 'study.yaml').read_bytes())
    monkeypatch.chdir(fake_root)
    output = tmp_path / 'output'
    with pytest.raises((SpecMtpSourceChangedError, FileNotFoundError)):
        readout.native_readout(fake_root, unmade, output)
    assert not output.exists()

def test_first_block_pairing_excludes_changed_prefill_and_missing_blocks(monkeypatch):
    readout = importlib.import_module('studies.qwen3_4b_bf16_bias_norm_scan.structure.readout')
    prompts = [{'question_id': i, 'input_ids': [1, 2]} for i in range(4)]
    def evidence(token, accepted):
        trace = [] if accepted is None else [{'accepted_nodes': list(range(accepted))}]
        return {'input_ids': [1, 2], 'output_ids': [1, 2, token],
            'stats': {'prefill_calls': 1, 'decode_calls': len(trace), 'trace': trace,
                      'attempt_by_depth': [0, 1, 0], 'accept_by_depth': [0, 0, 0]}}
    first = readout.depth_rows([evidence(3, n) for n in (0, 1, 1, None)], prompts)
    second = readout.depth_rows([evidence(3, 1), evidence(3, 0), evidence(4, 2), evidence(3, 2)], prompts)
    assert readout.first_block_compare(first, second) == {'total': 4, 'same_prefix_with_block': 2,
        'different_prefix': 1, 'missing_block': 1, 'accepted': {'win': 1, 'tie': 0, 'loss': 1}}
    second[0]['question_id'] = 99
    with pytest.raises(AssertionError):
        readout.first_block_compare(first, second)

def norm_reader(monkeypatch):
    m = module(monkeypatch)
    monkeypatch.syspath_prepend(str(m.STUDY))
    reader = importlib.import_module('studies.qwen3_4b_bf16_bias_norm_scan.flow.process.run')
    return reader

@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'order', 'wrong_scale', 'wrong_spec', 'wrong_eval', 'partial_factor', 'extra_run'])
def test_norm_matrix_must_match_all_eight_frozen_cases(monkeypatch, damage):
    import copy
    m = norm_reader(monkeypatch)
    decl = yaml.safe_load((m.STUDY / 'study.yaml').read_text(encoding='utf-8'))
    entries = [{'factors': {'algorithm.config': copy.deepcopy(c)}, 'runs': [{'id': str(i)}]}
               for i, c in enumerate(m.matrix_settings(decl))]
    m.check_matrix(decl, entries)
    if damage == 'missing':
        entries.pop()
    elif damage == 'duplicate':
        entries[-1] = copy.deepcopy(entries[0])
    elif damage == 'order':
        entries.reverse()
    elif damage == 'wrong_scale':
        entries[0]['factors']['algorithm.config']['spec']['ema']['step_scale'] = 1.
    elif damage == 'wrong_spec':
        entries[0]['factors']['algorithm.config']['spec']['ema']['beta'] = .9
    elif damage == 'wrong_eval':
        entries[0]['factors']['algorithm.config']['eval']['max_new_tokens'] = 128
    elif damage == 'partial_factor':
        entries[0]['factors']['algorithm.config'] = copy.deepcopy(decl['axes']['algorithm.config'][0])
    else:
        entries[0]['runs'].append({'id': 'extra'})
    with pytest.raises(AssertionError):
        m.check_matrix(decl, entries)

def test_reused_only_readout_never_accepts_new_runs_or_plots(monkeypatch, tmp_path):
    from studies._shared.structure import sources
    m = norm_reader(monkeypatch)
    # This test isolates reused-only aggregation; separate tests verify missing provenance refusal.
    from studies._shared.structure import validation
    monkeypatch.setattr(validation, "check_control_refs", lambda *args, **kwargs: None)
    rows = [dict(question_id=i, category='a', accepted=1, model_calls=3, new_tokens=5,
                 committed=4, decode_calls=2) for i in range(480)]
    depth = [dict(question_id=i, initial_prefix_sha256='same', first_block_accepted=1) for i in range(480)]
    names = ('zero/0', 'template/1', 'random0/1', 'random1/1', 'random2/1')
    cells = {name: dict(run_id=name.replace('/', '_'), rows=rows, depth_rows=depth,
                       totals={'pooled_tau': 2., 'argmax_checked_tokens': 2400},
                       raw_argmax_checked_tokens=2400, raw_files_rechecked=6, files_sha256={}) for name in names}
    reference = [{'output_ids': [1]} for _ in range(480)]
    formal = {'cells': {'AR': {'run_id': 'AR', 'files_sha256': {'assets/decode_observations.json': 'fixture'}}}}
    monkeypatch.setattr(m, 'load_reused', lambda root, reader=None: (formal, None, None, None, reference, cells))
    def forbidden(*args, **kwargs):
        pytest.fail('Reused-only mode cannot accept a new Run, check a new freeze or plot')
    monkeypatch.setattr(m._pair, 'load_cell', forbidden)
    monkeypatch.setattr(sources, 'check_frozen', forbidden)
    monkeypatch.setattr(m, 'plot', forbidden)
    # Synthetic source files exercise aggregation only; no historical claim.
    fixture = tmp_path / 'fixture'
    study = fixture / 'studies/norm'
    (study / 'docs').mkdir(parents=True)
    for name in ('study.yaml', 'experiment_config.yaml'):
        (study / name).write_bytes((m.STUDY / name).read_bytes())
    for relative in ('studies/norm/docs/source_refs.json',
            'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json',
            'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json'):
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(m, 'ROOT', fixture)
    monkeypatch.setattr(m, 'STUDY', study)
    monkeypatch.setattr(m, 'BASE', fixture / 'studies/qwen3_4b_bf16_five_methods')
    proof = m.verify_reused(fixture, tmp_path / 'output')
    assert proof['complete'] is False and proof['new_cells_inspected'] is False and proof['reused_only']
    assert len(proof['cells']) == 5 and len(proof['expected_new_cases']) == 8
    assert proof['raw_argmax_checked_tokens'] == 12000
    assert (tmp_path / 'output/reused_readiness_v1.json').is_file()
    assert not (tmp_path / 'output/norm_evidence_v1.json').exists()
    assert not (tmp_path / 'output/figures').exists()

def test_reused_only_missing_private_evidence_fails_without_new_outputs(monkeypatch, tmp_path):
    m = norm_reader(monkeypatch)
    with pytest.raises(FileNotFoundError):
        m.verify_reused(tmp_path / 'absent', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()

def test_published_reuse_checkpoint_matches_existing_endpoints(monkeypatch, historical_file):
    m = norm_reader(monkeypatch)
    proof = m.read(historical_file('studies/qwen3_4b_bf16_bias_norm_scan/docs/reused_readiness_v1.json'))
    original = m.read(historical_file('studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json'))
    assert not proof['complete'] and proof['reused_only'] and not proof['new_cells_inspected']
    assert len(proof['cells']) == 5 and len(proof['expected_new_cases']) == 8
    for path, expected in proof['formal_sources'].items():
        assert '\\' not in path and path.startswith('studies/')
        assert m.sha(historical_file(path)) == expected
    for name, cell in proof['cells'].items():
        source = original['cells'][name.split('/')[0]]
        assert cell['totals'] == source['totals']
        assert cell['files_sha256'] == source['files_sha256']
        assert cell['raw_argmax_checked_tokens'] == source['totals']['argmax_checked_tokens']
        assert cell['raw_files_rechecked'] == len(source['files_sha256']) == 6
    assert sum(c['raw_argmax_checked_tokens'] for c in proof['cells'].values()) == proof['raw_argmax_checked_tokens'] == 405083
    for pair in proof['paired_comparisons']:
        first = pair['first_block_same_prefix']
        assert first['total'] == first['same_prefix_with_block'] + first['different_prefix'] + first['missing_block'] == 480
        assert sum(first['accepted'].values()) == first['same_prefix_with_block'] == 480

def test_historical_norm_scan_immutable_source_refs(monkeypatch, historical_file):
    path = historical_file('studies/qwen3_4b_bf16_bias_norm_scan/docs/source_refs.json')
    import json
    refs = json.loads(path.read_text(encoding='utf-8'))
    for name in refs['files']:
        historical_file(name)
    from studies._shared.structure.validation import check_control_refs
    check_control_refs(path.parents[1], path.parents[3], kind='norm', git_repository=ROOT)
