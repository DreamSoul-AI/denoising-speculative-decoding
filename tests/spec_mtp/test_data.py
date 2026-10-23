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
