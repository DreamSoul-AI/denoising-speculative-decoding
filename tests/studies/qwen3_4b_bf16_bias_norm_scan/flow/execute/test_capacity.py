"""Capacity entry admission and bounded engineering decode; no 4B weights."""
import importlib

import pytest

from studies.qwen3_4b_bf16_bias_norm_scan.flow.execute import capacity


def test_capacity_import_and_help_never_construct_a_model(monkeypatch):
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    monkeypatch.setattr(HFCausalLM, '__init__', lambda *a, **kw: pytest.fail('import/help loaded a model'))
    importlib.reload(capacity)
    with pytest.raises(SystemExit) as caught:
        capacity.main(['--help'])
    assert caught.value.code == 0


@pytest.mark.parametrize('call', ['capacity.run(None)', 'capacity.measure_decode(None, None, None)'])
def test_capacity_rejects_optimized_python_before_reading_inputs_or_loading_models(tmp_path, call):
    import subprocess
    import sys
    script = ('from studies.qwen3_4b_bf16_bias_norm_scan.flow.execute import capacity\n'
              'try:\n    ' + call + '\n'
              "except RuntimeError as error:\n    assert 'assertion checks' in str(error)\n    print(str(error))\n"
              "else:\n    raise SystemExit('optimized capacity unexpectedly admitted')\n")
    result = subprocess.run([sys.executable, '-O', '-B', '-c', script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'capacity requires assertion checks' in result.stdout


@pytest.mark.parametrize('missing', ['model-root', 'model-sources', 'prompts', 'prompts-sha256', 'study',
                                   'study-sha256', 'plan', 'plan-sha256', 'device', 'output-root', 'version'])
def test_capacity_cli_requires_every_local_identity_and_new_output(monkeypatch, missing):
    monkeypatch.setattr(capacity, 'run', lambda *a, **kw: pytest.fail('incomplete CLI dispatched'))
    values = {name: ('cuda:0' if name == 'device' else 'fixture') for name in ['model-root', 'model-sources',
        'prompts', 'prompts-sha256', 'study', 'study-sha256', 'plan', 'plan-sha256', 'device', 'output-root', 'version']}
    with pytest.raises(SystemExit) as caught:
        capacity.main([item for name, value in values.items() if name != missing for item in ('--' + name, value)])
    assert caught.value.code == 2


@pytest.mark.parametrize('fault', ['none', 'close', 'input_drift', 'decode_and_close_without_add_note'])
def test_capacity_run_publishes_only_after_close_and_final_input_validation(tmp_path, monkeypatch, fault):
    import hashlib
    import json
    from types import SimpleNamespace
    import torch
    import yaml
    from transformers import AutoModelForCausalLM
    from spec_mtp.structure.model import hf_causal
    from studies._shared.structure import gpu_admission, sources
    from spec_mtp.structure.system import runtime
    inputs, weights = tmp_path / 'inputs', tmp_path / 'model'
    inputs.mkdir(); weights.mkdir()
    (weights / 'config.json').write_bytes(b'{}')
    manifest = inputs / 'model_sources.json'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({'config.json': sha(weights / 'config.json')}))
    prompts, study, plan = [inputs / name for name in ('prompts.json', 'study.yaml', 'plan.md')]
    prompts.write_text(json.dumps({'rendered_prompts': [{'question_id': 295, 'input_ids': [1] * 1500}]}))
    setting = {'direction': 'template', 'spec': {'ema': {'step_scale': 2}}}
    study.write_text(yaml.safe_dump({'axes': {'algorithm.config': [{}, setting]}}))
    plan.write_text('explicit engineering plan')
    args = SimpleNamespace(model_root=weights, model_sources=manifest, prompts=prompts,
        study=study, plan=plan, prompts_sha256=sha(prompts), study_sha256=sha(study), plan_sha256=sha(plan),
        device='cuda:0', output_root=tmp_path / 'output', version='round16-io-engineering')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'IO fixture; no CUDA computation')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda *a: None)
    monkeypatch.setattr(runtime, 'apply_runtime', lambda *a: None)
    monkeypatch.setattr(sources, 'package_files', lambda: {'explicit-host': 'fixed'})
    monkeypatch.setattr(gpu_admission, 'observe', lambda **kw: {'foreign_gpu_processes': [],
        'host_available_bytes': 64 << 30, 'current_worker_rss_bytes': 100})
    events, loading = [], []
    class NoNoteError(RuntimeError):
        def __getattribute__(self, name):
            if name == 'add_note':
                raise AttributeError(name)
            return super().__getattribute__(name)
    def loader(*a, **kw):
        loading.append(kw)
        return SimpleNamespace(to=lambda *a, **kw: None)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', loader)
    def close():
        events.append('close')
        assert not (args.output_root / 'capacity_complete.json').exists()
        if fault in ('close', 'decode_and_close_without_add_note'):
            raise RuntimeError('owned close failed')
    def build(*a, **kw):
        return SimpleNamespace(hf=AutoModelForCausalLM.from_pretrained(*a, **kw), device=torch.device('cuda:0'),
            tied_embeddings=True, close=close)
    monkeypatch.setattr(hf_causal, 'HFCausalLM', build)
    def measure(*a, **kw):
        events.append('measure')
        if fault == 'decode_and_close_without_add_note':
            raise NoNoteError('primary decode failure')
        if fault == 'input_drift':
            prompts.write_text('changed original input')
        return {'complete': True, 'capacity_only': True, 'argmax_checked_tokens': 256}
    monkeypatch.setattr(capacity, 'measure_decode', measure)
    if fault == 'none':
        result = capacity.run(args)
        complete = json.loads((args.output_root / 'capacity_complete.json').read_text())
        assert result['argmax_checked_tokens'] == 256 and complete['complete']
        assert not complete['host_process_tree_peak_measured'] and not complete['historical_calibration_revalidated']
    else:
        with pytest.raises((RuntimeError, ValueError), match='owned close failed|inputs or sources changed|primary decode failure'):
            capacity.run(args)
        assert not (args.output_root / 'capacity_complete.json').exists()
    assert events == ['measure', 'close']
    assert loading[0]['device_map'] == 'cuda:0' and loading[0]['local_files_only']
    assert AutoModelForCausalLM.from_pretrained is loader


def test_capacity_tiny_cuda_uses_the_controlled_policy_and_complete_argmax_chain(record_property):
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    assert torch.cuda.is_available(), 'capacity correctness requires actual CUDA; no CPU fallback'
    device = torch.device('cuda:0')
    torch.manual_seed(123)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
        'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 128,
        'tie_word_embeddings': True}, attn_implementation='eager')
    try:
        model.hf.to(device=device, dtype=torch.bfloat16)
        setting = {'spec': {'num_masks': 2, 'impl': 'naive', 'tree': {'mode': 'dynamic', 'pruning': True},
            'block_complexity': 6, 'method': 'ema_velocity', 'ema': {'level': 'order0', 'beta': .9999,
            'lam': .1, 'step_scale': 2., 'normalize': False, 'extrapolate_ema': False}}}
        torch.cuda.reset_peak_memory_stats(device)
        result = capacity.measure_decode(model, {'question_id': 295, 'input_ids': [1, 2, 3]}, setting, max_new_tokens=4)
        assert result['argmax_checked_tokens'] == result['generated_tokens'] == 4
        assert result['bias']['mode'] == 'template' and result['bias']['step_scale'] == 2
        assert result['capacity_only'] and result['no_timing'] and not result['formal_tau_reported']
        assert result['cuda_peak_allocated_bytes'] > 0 and result['dtype'] == 'torch.bfloat16'
        record_property('cuda_device', torch.cuda.get_device_name(device))
        record_property('validation_scope', 'tiny CUDA engineering decode; not original 4B capacity calibration')
    finally:
        model.close()
