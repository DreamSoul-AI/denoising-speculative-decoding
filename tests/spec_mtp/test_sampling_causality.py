"""Stochastic verification is coupled to AR; candidate ranking and RNG are separate."""

from spec_mtp.masks import build_mask_provider

import torch
import pytest

from test_window_observer import ROOT, model as model, spec
from studies._shared.structure.config import _convert


def test_conditional_draws_are_shared_not_greedy(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_sampling_paths import conditional_acceptance
    logits = torch.tensor([.1, .2, .3])
    candidates = {'self': 0, 'state': 0, 'proxy': 2, 'both': 2}
    result = conditional_acceptance(logits, candidates, .8, range(20, 52))
    repeat = conditional_acceptance(logits, candidates, .8, range(20, 52))
    assert result == repeat and result['RNG_consumed']
    assert len(set(result['verifier_token_ids'])) > 1
    cells = {c['label']: c for c in result['cells']}
    assert cells['self']['first_slot_hits'] == cells['state']['first_slot_hits']
    assert cells['proxy']['first_slot_hits'] == cells['both']['first_slot_hits']
    assert cells['proxy']['first_slot_hits'] == sum(x == 2 for x in result['verifier_token_ids'])


@pytest.mark.parametrize('temperature', [.8, 1.])
def test_output_hidden_sample_matching_equals_fresh_AR(model, temperature):
    from spec_mtp.decode import SpecDecoder, ar_generate
    prompt = torch.tensor([[5, 17, 6, 11, 20]])
    cfg = _convert({'temperature': temperature, 'max_new_tokens': 24, 'stop_on_eos': False})
    for seed in (123, 124):
        reference, _ = ar_generate(model, prompt, cfg, generator=torch.Generator().manual_seed(seed))
        for order in (0, 1):
            actual, _ = SpecDecoder(model, spec(order), cfg, generator=torch.Generator().manual_seed(seed), mask_policy=build_mask_provider(spec(order), model.embedding_table)).generate(prompt)
            assert torch.equal(reference, actual)


def test_positive_temperature_draft_is_top1_while_verifier_samples():
    from spec_mtp.decode import select_token, tree_probs
    from spec_mtp.tree import build_static_tree
    logits = torch.tensor([[.1, .2, .3], [.3, .2, .1]])
    for temperature in (.8, 1.):
        tree = build_static_tree(tree_probs(logits, temperature), [1, 1], root_token=1, pruning=False)
        assert tree.tokens == [2, 0]
        draws = [select_token(logits[0], temperature, torch.Generator().manual_seed(seed)) for seed in range(20, 52)]
        assert len(set(draws)) > 1


def test_owner1_seeds_have_correct_rows_and_positions():
    from spec_mtp.tree import build_static_tree
    from spec_mtp.tree import build_block_attention, mask_indices_of
    probs = torch.tensor([[.1, .7, .2], [.6, .3, .1]])
    tree = build_static_tree(probs, [1, 1], root_token=0, pruning=False)
    rows = mask_indices_of(tree, 2, 1)
    assert rows == [5, 6] and mask_indices_of(tree, 2, 0) == [3, 4]
    _, positions = build_block_attention(tree, 2, 717, 717, torch.float32, torch.device('cpu'), -1e9)
    assert positions[0, rows].tolist() == [719, 720]
    assert positions[0, mask_indices_of(tree, 2, 0)].tolist() == [718, 719]
