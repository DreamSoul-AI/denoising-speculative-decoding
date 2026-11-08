"""Actual local JSONL admission: identities, limits and ordering."""
import json
from types import SimpleNamespace
from studies._shared.structure.data import _build
from spec_mtp.structure.data.prompts import read_local_prompts as load_records, window_by_category

import pytest

from spec_mtp.structure.data.prompts import read_prompts, read_local_prompts


@pytest.mark.parametrize('damage', ['duplicate', 'boolean_id', 'empty_turns', 'not_object'])
def test_malformed_prompt_records_fail(tmp_path, damage):
    row = {'question_id': 1, 'category': 'math', 'turns': ['1+1?']}
    rows = [row]
    if damage == 'duplicate': rows *= 2
    if damage == 'boolean_id': row['question_id'] = True
    if damage == 'empty_turns': row['turns'] = []
    if damage == 'not_object': rows = [[]]
    path = tmp_path / '提问.jsonl'
    path.write_text(''.join(json.dumps(item) + '\n' for item in rows), encoding='utf-8')
    with pytest.raises(ValueError):
        read_prompts(path)


@pytest.mark.parametrize('key,value', [('num_prompts', True), ('num_prompts', -1), ('prompt_len', 0), ('vocab_size', False)])
def test_synthetic_shape_rejects_coercion_and_unbounded_values(key, value):
    with pytest.raises(ValueError):
        read_local_prompts({key: value})


@pytest.mark.parametrize('categories', [['keep'], ['other']])
def test_record_limit_applies_before_category_filter(tmp_path, categories):
    path = tmp_path / 'filtered.jsonl'
    with path.open('w', encoding='utf-8') as handle:
        for qid in range(100001):
            handle.write(json.dumps({'question_id':qid, 'category':'keep', 'turns':['text']}) + '\n')
    with pytest.raises(ValueError, match='bounded size'):
        read_prompts(path, categories=categories)


@pytest.mark.parametrize('categories', [['keep'], ['other']])
def test_exact_record_limit_keeps_filter_and_trailing_blanks(tmp_path, categories):
    path = tmp_path / 'at-limit.jsonl'
    with path.open('w', encoding='utf-8') as handle:
        for qid in range(100000):
            handle.write(json.dumps({'question_id':qid, 'category':'keep', 'turns':['text']}) + '\n')
        handle.write('\n   \n')
    records = read_prompts(path, categories=categories)
    assert len(records) == (100000 if categories == ['keep'] else 0)
    if records:
        assert records[0]['question_id'] == 0 and records[-1]['question_id'] == 99999


@pytest.mark.parametrize('limit', [0, 1, 2, 100000])
def test_dolly_prompt_limit_preserves_zero_and_filtered_order(limit):
    from types import SimpleNamespace
    import torch
    from spec_mtp.structure.data.prompts import encode_prompts

    seen = []
    def tokenize(messages, **kwargs):
        text = messages[0]['content']
        seen.append(text)
        return torch.tensor([[1] * len(text)])
    records = [{'category': 'creative_writing', 'text': text} for text in ('long', 'a', 'bb', 'c')]
    prompts = encode_prompts(records, SimpleNamespace(apply_chat_template=tokenize),
        {'name': 'dolly', 'num_prompts': limit, 'max_prompt_len': 2})
    assert [p['input_ids'].shape[1] for p in prompts] == [1, 2, 1][:limit]
    assert [p['question_id'] for p in prompts] == list(range(min(limit, 3)))
    assert records == [{'category': 'creative_writing', 'text': text} for text in ('long', 'a', 'bb', 'c')]
    if limit == 0:
        assert seen == []


@pytest.mark.parametrize('limit', [-1, True, False, 1.5, '2', 100001])
def test_dolly_prompt_limit_rejects_invalid_counts_before_tokenizing(limit):
    from types import SimpleNamespace
    from spec_mtp.structure.data.prompts import encode_prompts

    def tokenize(*args, **kwargs):
        pytest.fail('invalid prompt count reached tokenizer')
    with pytest.raises(ValueError, match='num_prompts'):
        encode_prompts([{'category': 'creative_writing', 'text': 'a'}],
                      SimpleNamespace(apply_chat_template=tokenize),
                      {'name': 'dolly', 'num_prompts': limit})


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
    from spec_mtp.structure.data.prompts import read_local_prompts as load_records

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
    model = SimpleNamespace(device=torch.device('cpu'), embedding_table=torch.eye(8),
                            vocab_size=8, calls=0, config=SimpleNamespace(num_hidden_layers=1))
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
    model = SimpleNamespace(device=torch.device('cpu'), embedding_table=torch.eye(8),
                            vocab_size=8, calls=0, config=SimpleNamespace(num_hidden_layers=1))
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
