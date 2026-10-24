"""Native magnitude controls keep the shared mean, fixed direction and argmax gate."""

from spec_mtp.masks import build_mask_provider
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import yaml

from spec_mtp.masks import EMAVelocityMaskProvider, ESPMaskProvider

ROOT = Path(__file__).resolve().parents[2]


def module(monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    path = ROOT / 'studies/qwen3_4b_bf16_bias_norm_scan/norm_control.py'
    spec = importlib.util.spec_from_file_location('sd_norm_control', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('mode', ['template', 'random0', 'random1', 'random2'])
def test_native_scale_retains_mean_and_seed_direction(monkeypatch, mode):
    m = module(monkeypatch)
    embeds = torch.randn(10, 16, generator=torch.Generator().manual_seed(7)).bfloat16()
    directions = []
    for scale in (0., .5, 1., 2.):
        provider = EMAVelocityMaskProvider(2, embeds, beta=.9999, level='order0', lam=.1, step_scale=scale)
        info = {}
        m.observe_initial_bias(provider, mode, info).init_from_prompt(embeds[:8])
        assert info['rounded_slot_displacement_norms'] == (provider.masks().float() - provider._level.float()).norm(dim=-1).tolist()
        zero = ESPMaskProvider(2, embeds, lam=.1)
        zero.init_from_prompt(embeds[:8])
        directions.append(provider._velocity.clone())
        for previous, new in zip(embeds[7:9], embeds[8:10]):
            assert torch.equal(provider._level, zero.masks()[0])
            if scale == 0:
                assert torch.equal(provider.masks(), zero.masks())
            provider.on_token_committed(new, previous)
            zero.on_token_committed(new, previous)
            assert torch.equal(provider._velocity, directions[-1])
        assert info['step_scale'] == scale and info['full_prompt_mean_preserved']
    assert all(torch.equal(v, directions[0]) for v in directions)


def test_eight_preregistered_cells_and_immutable_inputs(monkeypatch):
    m = module(monkeypatch)
    decl = yaml.safe_load((m.STUDY / 'study.yaml').read_text(encoding='utf-8'))
    rows = decl['axes']['algorithm.config']
    assert [(r['direction'], r['spec']['ema']['step_scale']) for r in rows] == [
        (mode, scale) for mode in ('template', 'random0', 'random1', 'random2') for scale in (.5, 2.)]
    assert all(r['spec']['num_masks'] == 2 and r['spec']['block_complexity'] == 30 for r in rows)


@pytest.mark.parametrize('scale', [.5, 2.])
def test_native_tiny_bf16_argmax_audit_for_scaled_bias(monkeypatch, scale):
    import spec_mtp.decode as decoding
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM
    m = module(monkeypatch)
    torch.manual_seed(17)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 2,
        'max_position_embeddings': 128}, torch_dtype='bfloat16', attn_implementation='eager')
    spec = _convert({'method': 'ema_velocity', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
        'tree': {'mode': 'dynamic', 'pruning': True},
        'ema': {'level': 'order0', 'beta': .9999, 'lam': .1, 'step_scale': scale}})
    original = build_mask_provider
    monkeypatch.setattr(decoding, 'time', SimpleNamespace(perf_counter=lambda: 0.))
    policy = m.observe_initial_bias(original(spec, model.embedding_table), 'random1', {})
    output, stats = decoding.SpecDecoder(model, spec, _convert({'temperature': 0., 'max_new_tokens': 12}), mask_policy=policy).generate(torch.tensor([[1, 2, 3, 4, 5]]))
    assert output.shape[1] == 17
    assert stats['verification_argmax_audit']['passed'] and stats['verification_argmax_audit']['checked_committed_tokens'] == 12


def test_unmade_readout_cannot_claim_completed_scan(monkeypatch, tmp_path):
    from studies._shared.structure.sources import SpecMtpSourceChangedError
    m = module(monkeypatch)
    monkeypatch.syspath_prepend(str(m.STUDY))
    spec = importlib.util.spec_from_file_location('sd_norm_readout', m.STUDY / 'analyze_results.py')
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)
    fake_root = tmp_path / 'checkout'
    unmade = fake_root / 'studies/unmade'
    unmade.mkdir(parents=True)
    (unmade / 'study.yaml').write_bytes((m.STUDY / 'study.yaml').read_bytes())
    monkeypatch.chdir(fake_root)
    output = tmp_path / 'output'
    with pytest.raises((SpecMtpSourceChangedError, FileNotFoundError)):
        readout.main(study_dir=unmade, output_dir=output)
    assert not output.exists()


def test_first_block_pairing_excludes_changed_prefill_and_missing_blocks(monkeypatch):
    m = module(monkeypatch)
    spec = importlib.util.spec_from_file_location('sd_norm_depth_readout', m.STUDY / 'analyze_results.py')
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)
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
    spec = importlib.util.spec_from_file_location('sd_norm_cpu_acceptance', m.STUDY / 'analyze_results.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
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
    import norm_control
    monkeypatch.setattr(norm_control, 'check_refs', lambda: None)
    rows = [dict(question_id=i, category='a', accepted=1, model_calls=3, new_tokens=5,
                 committed=4, decode_calls=2) for i in range(480)]
    depth = [dict(question_id=i, initial_prefix_sha256='same', first_block_accepted=1) for i in range(480)]
    names = ('zero/0', 'template/1', 'random0/1', 'random1/1', 'random2/1')
    cells = {name: dict(run_id=name.replace('/', '_'), rows=rows, depth_rows=depth,
                       totals={'pooled_tau': 2., 'argmax_checked_tokens': 2400},
                       raw_argmax_checked_tokens=2400, raw_files_rechecked=6, files_sha256={}) for name in names}
    reference = [{'output_ids': [1]} for _ in range(480)]
    formal = {'cells': {'AR': {'run_id': 'AR', 'files_sha256': {'assets/decode_observations.json': 'fixture'}}}}
    monkeypatch.setattr(m, 'load_reused', lambda root: (formal, None, None, None, reference, cells))
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
            'studies/base/docs/paired_evidence_v1.json',
            'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json'):
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(m, 'ROOT', fixture)
    monkeypatch.setattr(m, 'STUDY', study)
    monkeypatch.setattr(m, 'BASE', fixture / 'studies/base')
    proof = m.verify_reused(tmp_path / 'raw', tmp_path / 'output')
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


def test_measured_4b_admission_keeps_reserve_and_gpu_exclusion(monkeypatch):
    module(monkeypatch)
    path = ROOT / 'studies/qwen3_4b_bf16_bias_norm_scan/norm_admission.py'
    spec = importlib.util.spec_from_file_location('sd_norm_admission_test', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cap = {'complete': True, 'peak_process_tree_rss_bytes': 8 << 30,
           'decode': {'argmax_checked_tokens': 256, 'dtype': 'torch.bfloat16', 'cuda_peak_reserved_bytes': 10 << 30}}
    base = dict(host_available_bytes=9 << 30, current_worker_rss_bytes=1 << 30,
                foreign_gpu_processes=[], free_vram_mib=11264)
    assert helper.apply_capacity(base, cap, True)['passed']
    assert not helper.apply_capacity(dict(base, host_available_bytes=(9 << 30)-1), cap, True)['passed']
    assert not helper.apply_capacity(dict(base, foreign_gpu_processes=[{'pid': 1}]), cap, True)['passed']
    assert not helper.apply_capacity(dict(base, free_vram_mib=11263), cap, True)['passed']
    assert not helper.apply_capacity(dict(base, host_available_bytes=(2 << 30)-1), cap, False)['passed']
    with pytest.raises(AssertionError):
        helper.apply_capacity(base, dict(cap, complete=False), True)


def test_historical_norm_scan_immutable_source_refs(monkeypatch, historical_file):
    m = module(monkeypatch)
    path = historical_file('studies/qwen3_4b_bf16_bias_norm_scan/docs/source_refs.json')
    import json
    refs = json.loads(path.read_text(encoding='utf-8'))
    for name in refs['files']:
        historical_file(name)
    # Current validator reads the original frozen bytes; Git still queries the real repository.
    check_output = m.subprocess.check_output
    monkeypatch.setattr(m, 'subprocess', SimpleNamespace(
        check_output=lambda args, **kwargs: check_output(args, **{**kwargs, 'cwd': ROOT})))
    monkeypatch.setattr(m, 'ROOT', path.parents[3])
    monkeypatch.setattr(m, 'STUDY', path.parents[1])
    m.check_refs()
