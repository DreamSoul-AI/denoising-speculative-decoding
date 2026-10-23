"""Actual local JSONL admission: identities, limits and ordering."""
import json

import pytest

from spec_mtp.data import read_prompts, read_local_prompts


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
    from spec_mtp.data import encode_prompts

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
    if limit == 0:
        assert seen == []


@pytest.mark.parametrize('limit', [-1, True, False, 1.5, '2', 100001])
def test_dolly_prompt_limit_rejects_invalid_counts_before_tokenizing(limit):
    from types import SimpleNamespace
    from spec_mtp.data import encode_prompts

    def tokenize(*args, **kwargs):
        pytest.fail('invalid prompt count reached tokenizer')
    with pytest.raises(ValueError, match='num_prompts'):
        encode_prompts([{'category': 'creative_writing', 'text': 'a'}],
                      SimpleNamespace(apply_chat_template=tokenize),
                      {'name': 'dolly', 'num_prompts': limit})
