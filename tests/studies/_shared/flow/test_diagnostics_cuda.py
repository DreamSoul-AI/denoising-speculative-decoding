"""Bounded real CUDA cell and host cProfile engineering checks; no preset matrix."""

import hashlib
import json

import pytest
import torch

from studies._shared.flow import diagnostics as command
from studies._shared.structure import decode_fingerprint as fp


@pytest.mark.parametrize('temperature', [0., .7])
def test_single_fingerprint_cell_runs_on_cuda_with_fixed_seed(record_property, temperature):
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from studies._shared.structure.config import apply_variant

    assert torch.cuda.is_available(), 'fingerprint engineering check requires real CUDA'
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
        'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 64})
    try:
        model.hf.to('cuda:0')
        prompt = torch.tensor([[1, 2, 3, 4]], device='cuda:0')
        config = {'max_new_tokens': 3, 'temperature': temperature, 'stop_on_eos': False}
        spec = apply_variant({}, 'esp_static')
        first = fp.fingerprint_cell(model, prompt, spec, config, 100)
        second = fp.fingerprint_cell(model, prompt, spec, config, 100)
        assert first == second and first['new_tokens'] == 3
        if temperature == 0:
            assert first['equals_full_ar'] is True
        else:
            assert type(first['equals_full_ar']) is bool
        assert len(first['attempt_by_depth']) == len(first['accept_by_depth']) == 3
        record_property('cuda_device', torch.cuda.get_device_name(0))
    finally:
        model.close()


def test_profile_small_cuda_preserves_host_self_time_attribution_and_exact_match(tmp_path, monkeypatch, record_property):
    assert torch.cuda.is_available(), 'profile engineering check requires real CUDA'
    source = {'git': {'head': 'a' * 40, 'package_dirty': False},
              'checkout_git': {'head': 'a' * 40, 'package_dirty': False}, 'files': {'fixture.py': 'b' * 64}}
    monkeypatch.setattr(command, '_source', lambda: source)
    config, plan, output = tmp_path / 'config.json', tmp_path / 'plan.md', tmp_path / 'run'
    config.write_text(json.dumps({**command.PROFILE_DEFAULTS, 'variants': ['esp_static'], 'hidden': 32,
        'layers': 1, 'vocab': 64, 'prompt_len': 8, 'prompts': 1, 'new_tokens': 3}))
    plan.write_bytes(b'# Bounded CUDA engineering fixture\n')
    code = command.main(['profile', '--config', str(config), '--plan', str(plan),
        '--plan-sha256', hashlib.sha256(plan.read_bytes()).hexdigest(), '--device', 'cuda:0', '--output', str(output)])
    report = json.loads((output / 'profile.json').read_text())
    assert code == 0 and report['results'][0]['exact_match_rate'] == 1.
    assert abs(sum(report['results'][0]['self_time_share'].values()) - 1.) < .01
    assert report['identity']['device'] == 'cuda:0' and 'not CUDA kernel time' in report['measurement']
    assert 'unsynchronized host-clock' in report['rate_clock'] and 'not end-to-end CUDA throughput' in report['rate_clock']
    assert 'tok/s, host clock' in (output / 'profile.md').read_text()
    assert (output / 'protocol_complete.json').is_file()
    record_property('cuda_device', torch.cuda.get_device_name(0))
