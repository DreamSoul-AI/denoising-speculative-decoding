"""Untimed policy restores its clock and retains the complete verifier audit."""
from types import SimpleNamespace
import time

import pytest

import spec_mtp.structure.algorithm.decode.loop as decoding
from studies._shared.structure import algorithm


@pytest.mark.parametrize('failure', [False, True])
def test_study_wrapper_restores_clock_on_success_and_failure(monkeypatch, failure):
    original = decoding.time
    def parent(*args):
        assert decoding.time.perf_counter() == 0 and decoding.time is not time
        if failure:
            raise RuntimeError('injected failure')
        return {'ok': True}
    monkeypatch.setattr(algorithm.SpecMtpEvalAlgorithm, 'run', parent)
    adapter = object.__new__(algorithm.UntimedEval)
    adapter.config = SimpleNamespace(setting=lambda *args: {'warmup': False})
    if failure:
        with pytest.raises(RuntimeError, match='injected failure'):
            adapter.run(None, None, None, None)
    else:
        assert adapter.run(None, None, None, None) == {'ok': True}
    assert decoding.time is original is time


@pytest.mark.parametrize('warmup', [None, True, 0])
def test_untimed_eval_refuses_warmup_before_native_delegate(monkeypatch, warmup):
    monkeypatch.setattr(algorithm.SpecMtpEvalAlgorithm, 'run', lambda *args: pytest.fail('invalid timing contract'))
    adapter = object.__new__(algorithm.UntimedEval)
    adapter.config = SimpleNamespace(setting=lambda *args: {'warmup': warmup})
    with pytest.raises(AssertionError):
        adapter.run(None, None, None, None)
    assert decoding.time is time


@pytest.mark.parametrize('order', [None, 0, 1])
def test_untimed_decoder_preserves_complete_argmax_with_ar_as_diagnostic_on_cuda(monkeypatch, order):
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    from studies._shared.structure.config import _convert
    from studies._shared.structure.validation import _check_argmax

    assert torch.cuda.is_available(), 'CUDA is required; no CPU substitute'
    device = torch.device('cuda:0')
    print(f'untimed tiny correctness device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    torch.manual_seed(17)
    model = HFCausalLM(tiny_config={'vocab_size': 24, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 1,
        'max_position_embeddings': 64, 'tie_word_embeddings': True}, torch_dtype='float32', attn_implementation='eager').to(device)
    body = {'num_masks': 2, 'impl': 'efficient', 'tree': {'mode': 'static', 'branches': [1, 1], 'pruning': False}}
    if order is None:
        body.update(method='esp', mask={'init': 'mean_prompt', 'update': False, 'lam': 0})
    else:
        body.update(method='output_hidden', output={'order': order, 'beta': .9, 'beta_prime': .99})
    spec = _convert(body)
    evaluation = _convert({'max_new_tokens': 6, 'temperature': 0., 'stop_on_eos': False})
    prompt = torch.tensor([[1, 2, 3]], device=device)
    expected, timed = decoding.SpecDecoder(model, spec, evaluation,
        mask_policy=build_mask_provider(spec, model.embedding_table)).generate(prompt)
    def parent(*args):
        output, stats = decoding.SpecDecoder(model, spec, evaluation,
            mask_policy=build_mask_provider(spec, model.embedding_table)).generate(prompt)
        _, ar = decoding.ar_generate(model, prompt, evaluation)
        assert ar['ar_time'] == 0
        return output, stats
    monkeypatch.setattr(algorithm.SpecMtpEvalAlgorithm, 'run', parent)
    adapter = object.__new__(algorithm.UntimedEval)
    adapter.config = SimpleNamespace(setting=lambda *args: {'warmup': False})
    output, stats = adapter.run(None, None, None, None)
    assert torch.equal(output, expected)
    clocks = {'prefill_time', 'decode_time', 'draft_seconds', 'verify_seconds'}
    assert {key: value for key, value in stats.items() if key not in clocks} == {
        key: value for key, value in timed.items() if key not in clocks}
    assert stats['prefill_time'] == stats['decode_time'] == stats['draft_seconds'] == stats['verify_seconds'] == 0
    item = {'input_ids': prompt[0].tolist(), 'output_ids': output[0].tolist(),
            'ar_ids': [999], 'stats': stats}
    assert _check_argmax(item) == 6
    assert decoding.time is time
