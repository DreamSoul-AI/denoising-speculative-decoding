
from spec_mtp.masks import build_mask_provider
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_root_is_generated_index_and_bad_totals_are_rejected():
    module = tool('analyze_4b_geometry')
    stats = {'trace': [{'root_position': 5, 'cache_before': 5, 'accepted_nodes': [1], 'committed_tokens': [2, 3]},
                       {'root_position': 7, 'cache_before': 7, 'accepted_nodes': [], 'committed_tokens': [4]}],
             'decode_calls': 2, 'accepted': 1, 'committed': 3, 'accept_by_depth': [0, 1, 0, 0]}
    assert module.trace_metrics({'stats': stats}, 5) == [(0, 1, 2), (2, 0, 1)]
    with pytest.raises(AssertionError):
        module.trace_metrics({'stats': dict(stats, accepted=2)}, 5)
    with pytest.raises(AssertionError):
        module.trace_metrics({'stats': stats}, 4)
    assert module.distribution([]) == {'n': 0}
    assert module.distribution([2])['median'] == 2


def test_output_velocity_does_not_directly_change_first_slot_at_depth_zero_or_one():
    module = tool('capture_4b_geometry')
    proxies = torch.tensor([[2., 1.], [3., 2.]])
    anchor, velocity = torch.tensor([1., 0.]), torch.tensor([.5, -.5])
    raw, _ = module.raw_output_slots(anchor, velocity, proxies, 1, .99)
    assert torch.equal(raw[:2, 0], proxies)
    assert not torch.equal(raw[:2, 1], proxies)
    assert torch.equal(anchor, torch.tensor([1., 0.])) and torch.equal(velocity, torch.tensor([.5, -.5]))
    zero, _ = module.raw_output_slots(anchor, torch.zeros_like(velocity), proxies, 0, .99)
    assert torch.equal(zero[:, 0], zero[:, 1])


def test_target_embedding_alignment_and_tail_are_not_shifted():
    module = tool('capture_4b_geometry')
    embedding = torch.eye(4)
    masks = torch.stack([embedding[2], embedding[3]])[None]
    frame = {'root_index': 0, 'depths': [0], 'masks': masks, 'raw_predictions': masks,
             'real_hidden': embedding[2][None], 'velocity_ratios': [0.]}
    rows, skipped = module.point_rows([frame], torch.tensor([1, 2, 3]), embedding, 'input_first', 121, 2)
    assert skipped == 0 and [r['target_index'] for r in rows] == [1, 2]
    assert all(r['mask_target_cos'] == 1 for r in rows)
    assert [r['shared_prefix'] for r in rows] == [True, False]
    assert rows[0]['real_hidden_target_cos'] == 1
    _, skipped = module.point_rows([frame], torch.tensor([1, 2]), embedding, 'input_first', 121, 2)
    assert skipped == 1


@pytest.mark.parametrize('free,command,passed', [(19000, 'python release-ci.py', True),
                                                (18000, 'python release-ci.py', False),
                                                (19000, 'python model_training.py', False)])
def test_admission_defers_on_capacity_or_unknown_work(monkeypatch, free, command, passed):
    import json
    module = tool('capture_4b_geometry')
    def output(args, **kwargs):
        return f'GPU-test, {free}, 2000' if args[0] == 'nvidia-smi' else json.dumps(
            [{'ProcessId': -1, 'CommandLine': command}])
    monkeypatch.setattr(module.subprocess, 'check_output', output)
    assert module.admission()['passed'] is passed


@pytest.mark.parametrize('method', ['input_first', 'output_zero', 'output_first'])
def test_read_only_observer_preserves_tiny_bf16_output_and_forward_budget(method):
    from spec_mtp.model import HFCausalLM
    from spec_mtp.decode import SpecDecoder
    from studies._shared.structure.config import _convert
    module = tool('capture_4b_geometry')
    torch.manual_seed(5)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'num_layers': 1,
                                   'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 64})
    model.hf.lm_head.weight = model.hf.model.embed_tokens.weight
    model.hf.bfloat16()
    spec = {'method': 'ema_velocity' if method == 'input_first' else 'output_hidden', 'num_masks': 2,
            'block_complexity': 12, 'impl': 'naive', 'tree': {'mode': 'dynamic', 'pruning': True}}
    if method == 'input_first':
        spec['ema'] = {'level': 'order0', 'beta': .9999, 'lam': .1}
    else:
        spec['output'] = {'order': int(method == 'output_first'), 'beta': .9, 'beta_prime': .99}
    cfg = {'temperature': 0., 'max_new_tokens': 7, 'stop_on_eos': False}
    prompt = torch.tensor([[1, 2, 3]])
    expected, expected_stats = SpecDecoder(model, _convert(spec), _convert(cfg), mask_policy=build_mask_provider(_convert(spec), model.embedding_table)).generate(prompt)
    actual, stats, frames, calls = module.observe(model, prompt, spec, cfg, [], 123)
    assert torch.equal(expected[0], actual)
    assert expected_stats['accepted'] == stats['accepted'] and calls == 1 + len(frames)
    assert stats['verification_argmax_audit']['checked_committed_tokens'] == 7
    rows, _ = module.point_rows(frames, actual[3:], model.embedding_table, method, 121, 7)
    assert rows and all(row['target_index'] >= row['root_index'] + row['depth'] + 1 for row in rows)
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.hf.model.norm._forward_hooks


def test_cpu_downloader_requires_verified_content(monkeypatch):
    import json
    from types import SimpleNamespace
    module = tool('capture_4b_geometry')
    command = 'D:\\GitHub\\DreamSoul\\quantization\\.tmp\\paroquant-paper-env\\python.exe -u .tmp/paroquant-reference/download_layers.py'
    monkeypatch.setattr(module.subprocess, 'check_output', lambda args, **kw:
                        'GPU-test, 19000, 2000' if args[0] == 'nvidia-smi' else
                        json.dumps([{'ProcessId': -1, 'CommandLine': command}]))
    monkeypatch.setattr(module.Path, 'is_file', lambda _: True)
    monkeypatch.setattr(module.Path, 'read_bytes', lambda _: b'inspected downloader')
    monkeypatch.setattr(module.hashlib, 'sha256', lambda _: SimpleNamespace(hexdigest=lambda:
                        '10689bac808d0e5fa1958d306508f459d3c4637dc92a51bf5b6c1bb915545026'))
    assert module.admission()['passed'] and module.admission()['verified_cpu_preparation_count'] == 1
    monkeypatch.setattr(module.hashlib, 'sha256', lambda _: SimpleNamespace(hexdigest=lambda: 'changed'))
    assert not module.admission()['passed']


def test_verified_cpu_ci_descendants_not_unrelated_jobs(monkeypatch):
    import json
    module = tool('capture_4b_geometry')
    processes = [{'ProcessId': -1, 'ParentProcessId': 0, 'CommandLine': 'python release-ci.py'},
                 {'ProcessId': -3, 'ParentProcessId': -2, 'CommandLine': None},
                 {'ProcessId': -2, 'ParentProcessId': -1, 'CommandLine': None}]
    monkeypatch.setattr(module.subprocess, 'check_output', lambda args, **kw:
                        'GPU-test, 19000, 2000' if args[0] == 'nvidia-smi' else json.dumps(processes))
    assert module.admission()['passed'] and module.admission()['recognized_cpu_ci_count'] == 3
    processes.append({'ProcessId': -4, 'ParentProcessId': 0, 'CommandLine': 'python train.py'})
    assert not module.admission()['passed']


def test_geometry_summary_pairs_only_identical_shared_targets():
    module = tool('analyze_4b_geometry')
    points = []
    for method, value in [('input_first', .2), ('output_zero', .1), ('output_first', .11)]:
        for root in (0, 1):
            points.append(dict(case=method, question_id=121, root_index=root, depth=0, slot=0,
                               shared_prefix=root == 0, mask_target_cos=value, target_norm=1.,
                               raw_norm=1., mask_norm=1.,
                               mask_target_norm_ratio=1., velocity_anchor_ratio=.1))
    capture = {'quality_passed': True, 'status': 'complete', 'cases': [], 'points': points}
    result = module.summarize_geometry(capture)
    paired = result['matched_shared_prefix']['output_first-output_zero/slot0']['mask_target_cos']
    assert paired['n'] == 1 and paired['mean'] == pytest.approx(.01)
    assert result['groups']['all/input_first/depthNone/slot0']['mask_target_cos']['n'] == 2
    assert result['groups']['all/input_first/depthNone/slot1']['mask_target_cos'] == {'n': 0}
    for row in list(points):
        if row['root_index'] == 0:
            points.append(dict(row, slot=1, raw_norm=2., mask_norm=1.))
    effects = module.summarize_geometry(capture)['slot_effects']['shared_prefix/output_first']
    assert effects['raw_norm_slot1_over_slot0']['mean'] == 2
    assert effects['mask_norm_slot1_over_slot0']['mean'] == 1
    points.append(points[0])
    with pytest.raises(AssertionError, match='Duplicate'):
        module.summarize_geometry(capture)


def test_saved_audit_rejects_missing_or_changed_native_files(tmp_path):
    from hashlib import sha256
    from spec_mtp.audit import verify_files
    declared = {'runs/ar/assets/raw.json': sha256(b'frozen raw').hexdigest()}
    with pytest.raises(FileNotFoundError):
        verify_files(tmp_path, declared)
    raw = tmp_path / 'runs/ar/assets/raw.json'
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'corrupt native artifact')
    with pytest.raises(ValueError, match='changed'):
        verify_files(tmp_path, declared)
    assert raw.read_bytes() == b'corrupt native artifact'
    with pytest.raises(ValueError, match='escapes'):
        verify_files(tmp_path, {'../outside': 'not read'})


def test_published_saved_audit_preserves_full_coverage_and_old_accounting(historical_file):
    import json
    audit = json.loads(historical_file('docs/research/4b-paired-raw-audit-2026-10-16.json').read_text(encoding='utf-8'))
    saved = audit['saved_audit']
    assert saved['raw_files_checked'] == 66 and saved['raw_argmax_checked_tokens'] == 809118
    assert saved['shared_AR_tokens'] == 81880 and len(saved['pairs']) == 20
    assert audit['prior_accounting']['all_10_cells_equal'] and audit['prior_accounting']['all_6_tau_decompositions_equal']
    for pair in saved['pairs']:
        assert pair['first_block']['same_prefix_with_block'] == 480
        if pair['first'].endswith('/output_zero') and pair['second'].endswith('/output_first'):
            assert pair['first_block']['accepted'] == dict(win=0, tie=480, loss=0)
    assert audit['no_model_forward'] and audit['no_timing'] and audit['new_gpu_runs'] == 0
