"""CPU-only contracts for the separately authorized paired repeat."""

from spec_mtp.masks import build_mask_provider
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import copy

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'studies/qwen3_4b_bf16_paired_revalidation'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(STUDY))
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    return load('paired_repeat_test', STUDY / 'paired_repeat.py'), load('paired_repeat_readout_test', STUDY / 'analyze_results.py')


def test_all_eleven_effective_configs_match_native_expansion(modules):
    from studies._shared.structure.validation import declared_factors
    protocol, reader = modules
    configs = protocol.matrix_settings()
    decl, base = (yaml.safe_load((STUDY / name).read_text(encoding='utf-8')) for name in ('study.yaml', 'experiment_config.yaml'))
    factors = declared_factors(decl, base)
    assert [f['algorithm.config'] for f in factors] == configs
    entries = [dict(factors=f, runs=[{'id': str(i)}]) for i, f in enumerate(factors)]
    assert tuple(reader.check_index(entries, expected=configs)) == reader.original.CASES
    assert configs[1]['spec']['mask'] == dict(init='mean_prompt', lam=0., update=False)
    assert configs[2]['spec']['mask'] == dict(init='mean_prompt', lam=.1, update=True)
    assert len(configs) == 11 and configs[0]['task'] == 'ar_baseline'
    for setting in configs:
        assert setting['eval']['temperature'] == 0 and setting['eval']['max_new_tokens'] == 256
        assert not setting['eval']['warmup']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'order', 'lambda', 'eval', 'extra_run', 'duplicate_id'])
def test_changed_or_incomplete_repeat_index_is_rejected(modules, damage):
    protocol, reader = modules
    expected = protocol.matrix_settings()
    entries = [dict(factors={'algorithm.config': copy.deepcopy(c)}, runs=[{'id': str(i)}]) for i, c in enumerate(expected)]
    if damage == 'missing': entries.pop()
    elif damage == 'duplicate': entries[-1] = copy.deepcopy(entries[0])
    elif damage == 'order': entries.reverse()
    elif damage == 'lambda': entries[1]['factors']['algorithm.config']['spec']['mask']['lam'] = .1
    elif damage == 'eval': entries[1]['factors']['algorithm.config']['eval']['temperature'] = .8
    elif damage == 'extra_run': entries[1]['runs'].append({'id': 'extra'})
    else: entries[1]['runs'][0]['id'] = entries[0]['runs'][0]['id']
    with pytest.raises(AssertionError): reader.check_index(entries, expected=expected)


def test_resource_guard_precedes_model_and_both_decode_paths(modules, monkeypatch, tmp_path):
    m, _ = modules
    events = []
    monkeypatch.setattr(m, 'check_refs', lambda: events.append('refs'))
    monkeypatch.setattr(m, 'wait_for_admission', lambda assets, initial=False: events.append(('admit', initial)) or {'passed': True})
    monkeypatch.setattr(m, 'build_cuda', lambda *a, **kw: events.append('load') or SimpleNamespace(meta={}))
    assert m.admitted_build(None, tmp_path).meta['paired_repeat_admission']['passed']
    assert events == ['refs', ('admit', True), 'load']
    events.clear()
    monkeypatch.setattr(m.decoding.SpecDecoder, 'generate', lambda *a: events.append('spec') or 'spec')
    monkeypatch.setattr(m.ar_baseline, 'ar_generate', lambda *a, **kw: events.append('ar') or 'ar')
    def base_run(self, *args):
        assert m.decoding.SpecDecoder.generate(None, None) == 'spec'
        assert m.ar_baseline.ar_generate(None, None, {}) == 'ar'
        return 'complete'
    monkeypatch.setattr(m.PairedUntimedEval, 'run', base_run)
    instance = object.__new__(m.PairedRepeatEval)
    assert instance.run(None, None, SimpleNamespace(assets_dir=tmp_path), None) == 'complete'
    assert events == ['refs', ('admit', False), 'spec', ('admit', False), 'ar', 'refs']


def test_unmade_repeat_writes_pending_only(modules, monkeypatch, tmp_path):
    _, reader = modules
    unmade = tmp_path / 'unmade'
    (unmade / 'docs').mkdir(parents=True)
    monkeypatch.setattr(reader, 'STUDY', unmade)
    # Missing original provenance refuses readout rather than manufacturing a new pending identity.
    with pytest.raises(FileNotFoundError, match='source_refs'):
        reader.readout(tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    assert not (tmp_path / 'out/paired_evidence_v1.json').exists()


def test_native_argmax_gate_ignores_ar_diagnostic_but_rejects_changed_token():
    import torch
    from spec_mtp.model import HFCausalLM
    from spec_mtp.decode import SpecDecoder
    from studies._shared.structure.config import _convert
    from studies._shared.structure.validation import _check_argmax
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size': 24, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 1,
        'max_position_embeddings': 64}, torch_dtype='bfloat16', attn_implementation='eager')
    spec = _convert({'method': 'esp', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
                     'mask': {'init': 'mean_prompt', 'lam': 0., 'update': False},
                     'tree': {'mode': 'dynamic', 'pruning': True}})
    output, stats = SpecDecoder(model, spec, _convert({'temperature': 0., 'max_new_tokens': 6}), mask_policy=build_mask_provider(spec, model.embedding_table)).generate(torch.tensor([[1, 2, 3]]))
    item = {'input_ids': [1, 2, 3], 'output_ids': output[0].tolist(), 'ar_ids': [1, 2, 3, 999], 'stats': stats}
    assert _check_argmax(item) == 6
    item['output_ids'][-1] = (item['output_ids'][-1] + 1) % 24
    with pytest.raises(RuntimeError): _check_argmax(item)
