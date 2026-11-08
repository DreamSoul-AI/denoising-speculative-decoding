"""Fast IO/spies and explicit CUDA argmax contracts for the independent paired repeat."""

from spec_mtp.structure.algorithm.mask import build_mask_provider





import pytest


def test_native_argmax_gate_ignores_ar_diagnostic_but_rejects_changed_token():
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
    from studies._shared.structure.config import _convert
    from studies._shared.structure.validation import _check_argmax
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    print("paired correctness device=cuda:0 name="+torch.cuda.get_device_name(0), flush=True)
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size': 24, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 1,
        'max_position_embeddings': 64}, torch_dtype='bfloat16', attn_implementation='eager').to('cuda:0')
    spec = _convert({'method': 'esp', 'num_masks': 2, 'block_complexity': 30, 'impl': 'naive',
                     'mask': {'init': 'mean_prompt', 'lam': 0., 'update': False},
                     'tree': {'mode': 'dynamic', 'pruning': True}})
    output, stats = SpecDecoder(model, spec, _convert({'temperature': 0., 'max_new_tokens': 6}), mask_policy=build_mask_provider(spec, model.embedding_table)).generate(torch.tensor([[1, 2, 3]], device="cuda:0"))
    item = {'input_ids': [1, 2, 3], 'output_ids': output[0].tolist(), 'ar_ids': [1, 2, 3, 999], 'stats': stats}
    assert _check_argmax(item) == 6
    item['output_ids'][-1] = (item['output_ids'][-1] + 1) % 24
    with pytest.raises(RuntimeError): _check_argmax(item)
