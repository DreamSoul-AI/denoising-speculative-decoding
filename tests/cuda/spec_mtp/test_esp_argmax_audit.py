from spec_mtp.structure.algorithm.mask import build_mask_provider

import json


import pytest

import torch

import spec_mtp.structure.algorithm.decode.loop as decoding

from studies._shared.structure.config import _convert

from spec_mtp.structure.model.hf_causal import HFCausalLM

from studies._shared.structure import esp_controls as audit

def decoder():
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    print("ESP audit device=cuda:0 name="+torch.cuda.get_device_name(0), flush=True)
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size':32, 'hidden_size':16, 'intermediate_size':32,
        'num_layers':1, 'num_heads':2, 'num_kv_heads':1, 'max_position_embeddings':64}).to(device="cuda:0", dtype=torch.bfloat16)
    config = _convert({'method':'esp', 'num_masks':1, 'block_complexity':10, 'impl':'naive',
        'tree':{'mode':'dynamic','pruning':True}, 'mask':{'init':'mean_prompt','lam':.1,'update':True}})
    return decoding.SpecDecoder(model, config, {'temperature':0, 'max_new_tokens':7, 'stop_on_eos':False}, mask_policy=build_mask_provider(config, model.embedding_table))

@pytest.mark.parametrize('prefill_only', [False, True])
def test_observer_preserves_bf16_outputs_calls_and_restores(tmp_path, prefill_only):
    dec = decoder()
    if prefill_only:
        dec.max_new_tokens = 1
    prompt = torch.tensor([[1,2,3]], device="cuda:0")
    expected, base = dec.generate(prompt)
    generate, select = decoding.SpecDecoder.generate, decoding.select_token
    with audit.observe_argmax(tmp_path):
        actual, stats = dec.generate(prompt)
        # Independent AR remains outside the observer.
        decoding.ar_generate(dec.model, prompt, {'temperature':0,'max_new_tokens':1})
    assert torch.equal(actual, expected)
    assert stats['decode_calls'] == base['decode_calls']
    assert stats['trace'] == base['trace']
    assert decoding.SpecDecoder.generate is generate and decoding.select_token is select
    assert not dec.model._forward_hooks
    recorded = json.loads((tmp_path/'argmax_audit.jsonl').read_text())
    assert recorded == stats['verification_argmax_audit']
    assert recorded['passed'] and recorded['checked_committed_tokens'] == stats['new_tokens']
    with pytest.raises(AssertionError, match='old audit'):
        with audit.observe_argmax(tmp_path):
            pass

@pytest.mark.parametrize('cleanup_failure', [False, True])
def test_wrong_selection_is_rejected_and_restored(tmp_path, monkeypatch, cleanup_failure):
    dec = decoder()
    if cleanup_failure:
        register = dec.model.register_forward_hook
        def broken_cleanup(*args):
            handle = register(*args)
            remove = handle.remove
            def cleanup():
                remove()
                raise RuntimeError('secondary hook cleanup failed')
            handle.remove = cleanup
            return handle
        monkeypatch.setattr(dec.model, 'register_forward_hook', broken_cleanup)
    wrong = lambda row, temperature, generator: (int(row.argmax().item()) + 1) % row.numel()
    monkeypatch.setattr(decoding, 'select_token', wrong)
    generate = decoding.SpecDecoder.generate
    with pytest.raises(AssertionError, match='forward argmax') as raised:
        with audit.observe_argmax(tmp_path):
            dec.generate(torch.tensor([[1,2,3]], device="cuda:0"))
    assert decoding.select_token is wrong and decoding.SpecDecoder.generate is generate
    assert not dec.model._forward_hooks and not (tmp_path/'argmax_audit.jsonl').read_text()
    errors = getattr(raised.value, 'cleanup_errors', ())
    assert len(errors) == int(cleanup_failure)
    if errors: assert str(errors[0]) == 'secondary hook cleanup failed'
