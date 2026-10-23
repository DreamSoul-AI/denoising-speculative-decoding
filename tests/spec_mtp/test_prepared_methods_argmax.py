"""Exercise every frozen paired Spec configuration through the real BF16 argmax gate on CPU."""

from spec_mtp.masks import build_mask_provider
import copy
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
AXES = yaml.safe_load((ROOT / 'studies/qwen3_4b_bf16_paired_revalidation/study.yaml').read_text(encoding='utf-8'))['axes']['algorithm.config'][1:]

@pytest.mark.parametrize('setting', AXES, ids=[f"BC{c['spec']['block_complexity']}-{i % 5}" for i,c in enumerate(AXES)])
def test_every_prepared_spec_method_has_full_argmax_coverage_and_rejects_corruption(setting):
    from spec_mtp.model import HFCausalLM
    from spec_mtp.decode import SpecDecoder
    from studies._shared.structure.config import _convert
    from studies._shared.structure.validation import _check_argmax
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size': 24, 'hidden_size': 16, 'intermediate_size': 32,
        'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1, 'tie_word_embeddings': True,
        'max_position_embeddings': 128}, torch_dtype='bfloat16', attn_implementation='eager').to(torch.bfloat16)
    assert model.dtype == torch.bfloat16
    assert model.hf.get_output_embeddings().weight.data_ptr() == model.embedding_table.data_ptr()
    output, stats = SpecDecoder(model, _convert(copy.deepcopy(setting['spec'])),
        _convert({'temperature': 0., 'max_new_tokens': 6}), mask_policy=build_mask_provider(_convert(copy.deepcopy(setting['spec'])), model.embedding_table)).generate(torch.tensor([[1, 2, 3]]))
    item = {'input_ids': [1, 2, 3], 'output_ids': output[0].tolist(), 'ar_ids': [1, 2, 3, 999], 'stats': stats}
    assert stats['verification_argmax_audit']['passed']
    assert stats['verification_argmax_audit']['checked_committed_tokens'] == stats['new_tokens'] == 6
    assert _check_argmax(item) == 6
    item['output_ids'][-1] = (item['output_ids'][-1] + 1) % 24
    with pytest.raises(RuntimeError):
        _check_argmax(item)
