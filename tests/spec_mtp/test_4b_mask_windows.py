
from spec_mtp.masks import build_mask_provider
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def tool(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_radial_and_tangent_velocity_are_distinguished(monkeypatch):
    m = tool('analyze_4b_velocity', monkeypatch)
    a = torch.tensor([1., 0.])
    radial = m.velocity_geometry(a, torch.tensor([1., 0.]))
    assert radial['tangent_over_velocity'] == 0 and radial['normalized_movement_retention'] == 0
    tangent = m.velocity_geometry(a, torch.tensor([0., .01]))
    assert tangent['tangent_over_velocity'] == 1 and tangent['normalized_movement_retention'] > .99
    with pytest.raises(AssertionError):
        m.velocity_geometry(a, torch.zeros(2))


def test_transitions_use_previous_owner_depth_and_never_cross_prompts(monkeypatch):
    m = tool('analyze_4b_velocity', monkeypatch)
    trace = [{'root_position': 5, 'cache_before': 5, 'accepted_nodes': [1, 2], 'committed_tokens': [2, 3, 4], 'stopped': False},
             {'root_position': 8, 'cache_before': 8, 'accepted_nodes': [], 'committed_tokens': [5], 'stopped': False}]
    e = {'input_ids': [0] * 5, 'stats': {'trace': trace, 'decode_calls': 2, 'accepted': 2,
                                       'committed': 4, 'accept_by_depth': [0, 1, 1, 0]}}
    r = m.transition_counts([e, e])
    assert r['2']['transitions'] == 2 and r['2']['next_accept_d1_rate'] == 0
    assert r['0']['transitions'] == r['1']['transitions'] == 0


def test_local_factorial_keeps_native_first_and_shifts_only_declared_formula(monkeypatch):
    from spec_mtp.masks import OutputHiddenMaskProvider
    m = tool('diagnose_4b_mask_windows', monkeypatch)
    e = torch.tensor([[1., 0.], [0., 1.]])
    shadow = OutputHiddenMaskProvider(2, e, order=1)
    shadow.init_hidden(torch.tensor([[1., 0.], [1., 1.]]))
    proxies = torch.tensor([[2., 1.], [3., 1.]])
    before = shadow.anchor.clone(), shadow.velocity.clone()
    zero = torch.ones(3, 2, 2, dtype=torch.bfloat16)
    masks = m.intervention_masks(shadow, proxies, zero)
    assert torch.equal(masks['zero'], zero)
    assert torch.equal(masks['first'], shadow.masks_per_owner(proxies).bfloat16())
    assert torch.equal(masks['shifted'][:, 0], masks['first'][:, 1])
    assert not torch.equal(masks['radial'][:, 1], masks['first'][:, 1])
    assert torch.equal(shadow.anchor, before[0]) and torch.equal(shadow.velocity, before[1])


@pytest.mark.parametrize('template', [False, True])
def test_fixed_window_uses_native_control_and_predicts_after_bonus(monkeypatch, template):
    from spec_mtp.model import HFCausalLM
    from spec_mtp.decode import SpecDecoder
    from studies._shared.structure.config import _convert
    m = tool('diagnose_4b_mask_windows', monkeypatch)
    torch.manual_seed(42)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'num_layers': 1,
                                  'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 128})
    model.hf.lm_head.weight = model.hf.model.embed_tokens.weight
    model.hf.bfloat16()
    spec = {'method': 'output_hidden', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
            'tree': {'mode': 'dynamic', 'pruning': True}, 'output': {'order': 0, 'beta': .9, 'beta_prime': .99}}
    cfg = {'temperature': 0., 'max_new_tokens': 32, 'stop_on_eos': False}
    ids = torch.tensor([[1, 2, 3]])
    reference, _ = SpecDecoder(model, _convert(spec), _convert(cfg), mask_policy=build_mask_provider(_convert(spec), model.embedding_table)).generate(ids)
    packet = m.capture_window(model, ids, spec, cfg, [], 123, minimum=2)
    input_spec = {'method': 'ema_velocity', 'num_masks': 2, 'ema': {'level': 'order0', 'beta': .9999, 'lam': .1}}
    packet['masks']['input_first'] = m.input_control(model, ids, reference[0, 3:3 + packet['root_index'] + 1].tolist(), input_spec)
    variants = m.TEMPLATE_VARIANTS if template else m.VARIANTS
    if template:
        for variant, mode in (('input_template', 'template'), ('input_mean', 'zero')):
            packet['masks'][variant] = m.input_control(model, ids,
                reference[0, 3:3 + packet['root_index'] + 1].tolist(), input_spec, velocity_mode=mode)
    assert not model._forward_hooks and not model._forward_pre_hooks
    rows, private = m.probe_window(model, packet, spec, reference[0, 3:].tolist(), variants)
    assert [r['variant'] for r in rows] == list(variants) and len(private) == len(variants)
    assert all(r['current_owner_argmax_unchanged'] and r['argmax_passed'] for r in rows)
    assert all(r['next_root_index'] == packet['root_index'] + len(packet['chosen']) for r in rows)
    assert rows[0]['next_verified_chain_equals_reference']


def test_template_and_zero_controls_keep_same_updated_mean(monkeypatch):
    from types import SimpleNamespace
    from spec_mtp.masks import EMAVelocityMaskProvider, ESPMaskProvider
    m = tool('diagnose_4b_mask_windows', monkeypatch)
    table = torch.tensor([[1., 0.], [0., 1.], [2., 3.], [3., -2.], [1., 2.]], dtype=torch.bfloat16)
    model = SimpleNamespace(embedding_table=table, dtype=table.dtype, embed=lambda ids: table[ids])
    ids = torch.tensor([[0, 1, 2]])
    cfg = {'method': 'ema_velocity', 'num_masks': 2, 'ema': {'level': 'order0', 'beta': .5, 'lam': .1}}
    provider = EMAVelocityMaskProvider(2, table, beta=.5, level='order0', lam=.1)
    provider.init_from_prompt(table[ids[0]])
    provider.on_token_committed(table[3], table[2])
    provider.on_token_committed(table[4], table[3])
    native = m.input_control(model, ids, [3, 4], cfg)
    frozen = m.input_control(model, ids, [3, 4], cfg, velocity_mode='template')
    mean = m.input_control(model, ids, [3, 4], cfg, velocity_mode='zero')
    steps = torch.arange(1, 3, dtype=table.dtype)[:, None]
    assert torch.equal(native[0], provider.masks())
    assert torch.equal(frozen[0], provider._level + steps * (table[1] - table[0]))
    assert torch.equal(mean[0], provider._level.expand(2, -1))
    assert not torch.equal(mean[0, 0], table[ids[0]].mean(0))
    assert not torch.equal(frozen, native) and not torch.equal(frozen, mean)
    esp = ESPMaskProvider(2, table, lam=.1, update=True)
    esp.init_from_prompt(table[ids[0]])
    esp.on_token_committed(table[3], table[2])
    esp.on_token_committed(table[4], table[3])
    assert torch.equal(mean[0], esp.masks())
    info = {}
    random = m.input_control(model, ids, [3, 4], cfg, velocity_mode='random', random_seed=0, state_report=info)
    assert abs(info['template_norm_ratio'] - 1) < .01
    assert torch.equal(random, m.input_control(model, ids, [3, 4], cfg, velocity_mode='random', random_seed=0))
    assert not torch.equal(random, m.input_control(model, ids, [3, 4], cfg, velocity_mode='random', random_seed=1))
    content_info = {}
    content = m.input_control(model, ids, [3, 4], cfg, velocity_mode='content', content_start=1, state_report=content_info)
    assert content_info['level_norm'] == float(esp.masks()[0].float().norm())
    v = .5 * (table[2] - table[1]) + .5 * (table[3] - table[2])
    v = .5 * v + .5 * (table[4] - table[3])
    assert torch.equal(content[0], esp.masks() + steps * v)


def test_reverse_swap_replays_actual_input_path_and_preserves_native_control(monkeypatch):
    from spec_mtp.model import HFCausalLM
    from spec_mtp.decode import SpecDecoder
    from studies._shared.structure.config import _convert
    m = tool('diagnose_4b_mask_windows', monkeypatch)
    torch.manual_seed(42)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'num_layers': 1,
                                  'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 128})
    model.hf.lm_head.weight = model.hf.model.embed_tokens.weight
    model.hf.bfloat16()
    spec = {'method': 'ema_velocity', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
            'tree': {'mode': 'dynamic', 'pruning': True}, 'ema': {'level': 'order0', 'beta': .9999, 'lam': .1}}
    cfg = {'temperature': 0., 'max_new_tokens': 32, 'stop_on_eos': False}
    ids = torch.tensor([[1, 2, 3]])
    reference, _ = SpecDecoder(model, _convert(spec), _convert(cfg), mask_policy=build_mask_provider(_convert(spec), model.embedding_table)).generate(ids)
    packet = m.capture_input_window(model, ids, spec, cfg, [], 123, minimum=1)
    packet['masks']['input_first'] = m.input_control(model, ids,
        reference[0, 3:3 + packet['root_index'] + 1].tolist(), spec)
    packet['next_control_masks'] = m.input_control(model, ids, packet['prefix_generated'], spec)
    rows, _ = m.probe_window(model, packet, spec, reference[0, 3:].tolist(), m.RECIPROCAL_VARIANTS, control='input_first')
    assert len(rows) == 3 and all(r['argmax_passed'] and r['current_owner_argmax_unchanged'] for r in rows)
    assert next(r for r in rows if r['variant'] == 'input_first')['next_verified_chain_equals_reference']
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_window_admission_uses_only_gpu_residency_and_capacity(monkeypatch):
    m = tool('diagnose_4b_mask_windows', monkeypatch)
    free = 19000
    processes = f'1, C:\\Windows\\dwm.exe\n{m.os.getpid()}, D:\\python.exe\n'
    def query(args, **kwargs):
        assert args[0] == 'nvidia-smi' and '--id=0' in args
        return f'{free}, 2000' if '--query-gpu=memory.free,memory.used' in args else processes
    monkeypatch.setattr(m.subprocess, 'check_output', query)
    assert m.admission()['passed']
    processes += '999999, D:\\unknown\\Python3.13.exe\n'
    assert not m.admission()['passed'] and m.admission()['foreign_gpu_python_pids'] == [999999]
    processes = ''
    free = 14000
    assert not m.admission()['passed'] and m.admission(initial=False)['passed']
    free = 19000
    assert m.admission()['passed']


def test_coverage_separates_budget_and_correct_parent_from_real_acceptance(monkeypatch):
    m = tool('analyze_4b_window_results', monkeypatch)
    logits = torch.tensor([[5., 4.9, 1., 0.], [0., 0., 6., 0.]])
    tree, r = m.coverage(logits, root=3, truth=[1, 2], budget=3)
    assert r['reference_d1_in_tree'] and not r['reference_d1_is_expanded_top1']
    assert not r['reference_d2_in_actual_tree'] and r['oracle_parent_d2_definite_coverage_only']
    assert r['all_budget_to_d1_definite_coverage_only']
    _, repeated = m.coverage(logits, root=1, truth=[1, 2], budget=3)
    assert repeated['repeat_pruning_bans_d1'] and not repeated['all_budget_to_d1_possible_coverage_only']
    assert m.rank_interval(torch.tensor([2., 2., 0.]), 0, 2) == (1, 2)


def test_active_selection_uses_first_predecessor_and_rejects_no_opportunity(monkeypatch):
    m = tool('diagnose_4b_mask_windows', monkeypatch)
    trace = [{'root_position': 5, 'accepted_nodes': []}, {'root_position': 6, 'accepted_nodes': [1, 2]},
             {'root_position': 9, 'accepted_nodes': [1]}, {'root_position': 11, 'accepted_nodes': [1]}]
    assert m.active_predecessor(trace, 5) == 1
    with pytest.raises(ValueError, match='No predefined'):
        m.active_predecessor(trace[:2], 5)


def test_constant_template_check_is_only_algebraic_not_an_acceptance_claim(monkeypatch):
    m = tool('analyze_4b_velocity', monkeypatch)
    level, velocity = torch.tensor([1., 1.]), torch.tensor([1., 0.])
    masks = torch.stack([level + velocity, level + 2 * velocity])
    result = m.template_metrics(level, velocity, velocity, masks)
    assert result['velocity_initial_relative_error'] == result['constant_slot1_relative_error'] == 0
    assert result['velocity_initial_cos'] == 1
    changed = m.template_metrics(level, velocity, -velocity, masks)
    assert changed['velocity_initial_cos'] == -1 and changed['constant_slot0_relative_error'] > 0
