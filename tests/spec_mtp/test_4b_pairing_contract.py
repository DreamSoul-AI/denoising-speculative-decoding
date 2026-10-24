import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
import spec_mtp.decode as decoding
from rpipe.structure.make.expand import expand_patches, load_study_yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'studies/qwen3_4b_bf16_five_methods'


def load_adapter():
    spec = importlib.util.spec_from_file_location('paired_4b_adapter', STUDY/'untimed_pairing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predeclared_11_cells_share_bf16_data_and_two_mask_trees():
    config = yaml.safe_load((STUDY/'experiment_config.yaml').read_text())
    study = load_study_yaml(STUDY)
    patches = expand_patches(study)
    assert len(patches) == 11 and patches[0]['algorithm']['config']['task'] == 'ar_baseline'
    assert config['model']['config']['name_or_path'] == 'Qwen/Qwen3-4B'
    assert config['model']['config']['torch_dtype'] == 'bfloat16'
    assert config['algorithm']['config']['eval']['run_ar_baseline'] is False
    specs = [p['algorithm']['config']['spec'] for p in patches[1:]]
    assert [s['block_complexity'] for s in specs] == [30]*5+[60]*5
    assert all(s['num_masks'] == 2 and s['impl'] == 'naive' and
               s['tree'] == {'mode': 'dynamic', 'pruning': True} for s in specs)
    assert specs[:5] == [dict(s, block_complexity=30) for s in specs[5:]]
    assert specs[0]['mask']['update'] is False and specs[1]['mask']['lam'] == .1
    assert specs[2]['ema']['beta'] == .9999 and specs[2]['ema']['step_scale'] == 1.
    assert [s['output']['order'] for s in specs[3:5]] == [0,1]


def test_historical_11_cell_original_inputs(historical_file):
    prompts = json.loads(historical_file('studies/qwen3_4b_bf16_five_methods/docs/prompts_manifest.json').read_text())
    prior = json.loads(historical_file('studies/esp_table1_bf16_full_argmax/docs/prompts_manifest.json').read_text())
    assert len(prompts['rendered_prompts']) == 480
    assert prompts['rendered_prompts'] == prior['rendered_prompts']
    assert len(json.loads(historical_file('studies/qwen3_4b_bf16_five_methods/docs/modelscope_snapshot.json').read_text())['files']) == 14


@pytest.mark.parametrize('failure', [None, 'runtime', 'dtype', 'prompt', 'weights', 'tied'])
def test_pairing_adapter_checks_before_decode_and_restores_globals(tmp_path, monkeypatch, failure):
    adapter = load_adapter()
    docs = tmp_path/'docs'; docs.mkdir()
    manifest = {'data_sha256': 'data', 'eos_token_ids': [2], 'rendered_prompts': [
        {'category': 'a', 'input_ids': [1,2]}]}
    snapshot = {'config': {'model_type': 'qwen3'}, 'files': [{'path': 'config.json', 'sha256': 'weights'}]}
    (docs/'prompts_manifest.json').write_text(json.dumps(manifest))
    (docs/'modelscope_snapshot.json').write_text(json.dumps(snapshot))
    monkeypatch.setattr(adapter, '__file__', str(tmp_path/'untimed_pairing.py'))
    saved = {'dtype': 'torch.bfloat16', 'backend': 'eager', 'eos_ids': [2],
        'prompts': [{'category': 'a', 'input_ids': [1,2]}],
        'files': {'/cache/config.json': 'weights', '/data/question.jsonl': 'data'},
        'settings': {'eval': {'temperature': 0., 'max_new_tokens': 256, 'warmup': False,
                              'stop_on_eos': True, 'run_ar_baseline': False}}}
    if failure == 'dtype': saved['dtype'] = 'torch.float32'
    if failure == 'prompt': saved['prompts'][0]['input_ids'] = [3]
    if failure == 'weights': saved['files']['/cache/config.json'] = 'wrong'
    weight = torch.ones(1, 2, dtype=torch.bfloat16)
    module = SimpleNamespace(dtype=torch.bfloat16, device=torch.device('cuda'), embedding_table=weight,
        hf=SimpleNamespace(config=SimpleNamespace(model_type='qwen3'),
            get_output_embeddings=lambda: SimpleNamespace(weight=weight.clone() if failure == 'tied' else weight)))
    clock = decoding.time
    capture = lambda *args, **kwargs: saved
    monkeypatch.setattr(adapter.evidence, 'capture', capture)
    decoded = []
    def parent(*args):
        assert decoding.time.perf_counter() == 0 and decoding.time is not clock
        assert adapter.evidence.capture() is saved
        decoded.append(True)
        if failure == 'runtime': raise RuntimeError('injected decode failure')
        return {'ok': True}
    monkeypatch.setattr(adapter.SpecMtpEvalAlgorithm, 'run', parent)
    algo = adapter.PairedUntimedEval.__new__(adapter.PairedUntimedEval)
    algo.config = SimpleNamespace(setting=lambda *args: {'warmup': False})
    args = (None, SimpleNamespace(module=module), SimpleNamespace(assets_dir=tmp_path), None)
    if failure:
        with pytest.raises(RuntimeError if failure == 'runtime' else AssertionError):
            algo.run(*args)
    else:
        assert algo.run(*args) == {'ok': True}
        assert json.loads((tmp_path/'pairing_contract.json').read_text())['manifest_checked']
    assert bool(decoded) == (failure in (None, 'runtime'))
    assert decoding.time is clock and adapter.evidence.capture is capture


@pytest.mark.parametrize('failure', [False, True])
def test_direct_cuda_loader_preserves_bf16_and_restores(monkeypatch, failure):
    from transformers import AutoModelForCausalLM
    adapter = load_adapter()
    seen = {}
    def original(*args, **kwargs):
        seen.update(kwargs)
        if failure: raise RuntimeError('injected load failure')
        return 'model'
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', original)
    def factory(*args):
        return AutoModelForCausalLM.from_pretrained('cache', dtype=torch.bfloat16, local_files_only=True)
    monkeypatch.setattr(adapter, '_build_hf', factory)
    if failure:
        with pytest.raises(RuntimeError, match='load failure'):
            adapter.build_cuda(None, None)
    else:
        assert adapter.build_cuda(None, None) == 'model'
    assert seen == {'dtype': torch.bfloat16, 'local_files_only': True, 'device_map': 'cuda'}
    assert AutoModelForCausalLM.from_pretrained is original
