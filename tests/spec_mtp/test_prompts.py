"""Prompt windows and the synthetic set. No SpecBench download."""

from types import SimpleNamespace

import pytest

from spec_mtp.data import read_local_prompts as load_records
from studies._shared.structure.data import _build
from spec_mtp.data import window_by_category


def test_window_is_per_category_in_name_order():
    questions = [
        {'category': 'b', 'question_id': 1},
        {'category': 'b', 'question_id': 2},
        {'category': 'a', 'question_id': 3},
        {'category': 'a', 'question_id': 4},
        {'category': 'a', 'question_id': 5},
    ]
    window = window_by_category(questions, skip=1, take=1)
    assert [item['question_id'] for item in window] == [4, 2]


def test_synthetic_prompts_are_seeded():
    cfg = {'mode': 'synthetic', 'num_prompts': 2, 'prompt_len': 4, 'vocab_size': 32, 'seed': 7}
    first = load_records(cfg)
    second = load_records(cfg)
    assert len(first) == 2
    assert first[0]['input_ids'].shape == (1, 4)
    assert (first[0]['input_ids'] == second[0]['input_ids']).all()


def test_registry_builds_synthetic_prompts_without_a_tokenizer():
    config = SimpleNamespace(name='local_prompts', config={
        'mode': 'synthetic', 'num_prompts': 3, 'prompt_len': 4, 'vocab_size': 16, 'seed': 1,
    })
    data = _build(config, None)
    assert data.source == 'spec_mtp'
    assert data.meta['prompt_count'] == 3


def test_select_prompts_caps_and_keeps_eval_eos():
    from studies._shared.structure.inputs import select_prompts

    prompts = [{'id': i} for i in range(5)]
    sliced, eos = select_prompts(prompts, [1, 2], {'num_eval_prompts': 2})
    assert sliced == prompts[:2] and eos == [1, 2]
    sliced, eos = select_prompts(prompts, [1], {'num_eval_prompts': 99, 'eos_token_ids': [9]})
    assert sliced == prompts and eos == [9]
    assert select_prompts(prompts, [], {'num_eval_prompts': 0}) == ([], [])


def test_local_prompts_text_mode_and_bad_mode():
    from spec_mtp.data import read_local_prompts as load_records

    rows = load_records({'mode': 'text', 'num_prompts': 3})
    assert len(rows) == 3 and all('text' in r for r in rows)
    try:
        load_records({'mode': 'bytes'})
        assert False, 'expected ValueError'
    except ValueError as error:
        assert 'Unsupported local_prompts mode' in str(error)


@pytest.mark.parametrize('limit', [-1, True, 1.5, '2'])
@pytest.mark.parametrize('entry', ['selector', 'alignment'])
def test_eval_prompt_limit_rejects_negative_or_coerced_values(limit, entry):
    import torch
    from studies._shared.structure.inputs import select_prompts
    from studies._shared.structure import alignment_probe
    from test_decode import FakeSession
    model = FakeSession()
    model.config = SimpleNamespace(num_hidden_layers=1)
    prompts = [{'input_ids':torch.tensor([[1, 2, 3]])} for _ in range(3)]
    cfg = {'num_eval_prompts':limit, 'max_new_tokens':0}
    with pytest.raises(ValueError, match='num_eval_prompts'):
        if entry == 'selector':
            select_prompts(prompts, [], cfg)
        else:
            alignment_probe.evaluate(model, prompts, [], {'num_masks':1}, cfg, aggregate=False)
    assert len(prompts) == 3 and model.calls == 0
    if entry == 'alignment':
        with pytest.raises(ValueError, match='num_eval_prompts'):
            alignment_probe.evaluate(None, prompts, [], {}, cfg, aggregate=False)


@pytest.mark.parametrize('limit', [None, 0, 2, 99])
def test_alignment_prompt_cap_is_idempotent_and_keeps_eos(limit, monkeypatch):
    import torch
    from studies._shared.structure.inputs import select_prompts
    from studies._shared.structure import alignment_probe
    from test_decode import FakeSession
    model = FakeSession()
    model.config = SimpleNamespace(num_hidden_layers=1)
    prompts = [{'input_ids':torch.tensor([[i, 2, 3]])} for i in (1, 2, 3)]
    cfg = {'max_new_tokens':0}
    if limit is not None:
        cfg['num_eval_prompts'] = limit
    selected, eos = select_prompts(prompts, [7], cfg)
    reached = []
    original = alignment_probe.ar_generate
    def observe(model, ids, *args, **kwargs):
        reached.append((ids.tolist(), kwargs['eos_ids']))
        return original(model, ids, *args, **kwargs)
    monkeypatch.setattr(alignment_probe, 'ar_generate', observe)
    alignment_probe.evaluate(model, selected, eos, {'num_masks':1}, cfg, aggregate=False)
    expected = 3 if limit is None else min(limit, 3)
    assert [ids for ids, _ in reached] == [p['input_ids'].tolist() for p in prompts[:expected]]
    assert all(ids == [7] for _, ids in reached) and eos == [7]
    assert model.calls == 0 and len(prompts) == 3
