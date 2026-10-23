"""Protocol arithmetic keeps missing prompts and negative pairs."""
import pytest

from spec_mtp.analysis import compare, summarize


def row(qid='q', accepted=2, **updates):
    return dict(qid=qid, accepted=accepted, committed=3, decode_calls=2, new_tokens=4,
                input_sha256='input', protocol='v1', evidence_complete=True, **updates)


def test_pooled_tau_and_missing_prompts_are_explicit():
    result = summarize([row()], {'tau': 'decode_committed/decode_calls', 'qids': ['q', 'missing']})
    assert result['tau'] == 1.5 and result['missing'] == ['missing'] and not result['complete']


def test_pairing_preserves_losses_and_missing_rows():
    result = compare([row(accepted=1), row('absent')], [row(accepted=3)], {})
    assert result['losses'] == 1 and result['incomparable'] == 1 and len(result['pairs']) == 2


def test_output_length_mismatch_is_not_a_speed_or_acceptance_comparison():
    other = row()
    other['new_tokens'] = 5
    result = compare([row()], [other], {})
    assert result['incomparable'] == 1


def test_incomplete_evidence_is_preserved_as_incomparable():
    missing = row(accepted=20)
    missing['evidence_complete'] = False
    result = compare([missing], [row()], {})
    assert result['wins'] == 0 and result['incomparable'] == 1 and missing['accepted'] == 20


@pytest.mark.parametrize('bad', [True, float('nan'), -1, '1'])
def test_untrusted_counts_are_rejected(bad):
    item = row()
    item['accepted'] = bad
    with pytest.raises(ValueError):
        summarize([item], {'tau': 'decode_committed/decode_calls'})


@pytest.mark.parametrize('keys', [[], ['factor', 'input_sha256']])
@pytest.mark.parametrize('field', ['input_sha256', 'protocol', 'new_tokens'])
def test_custom_pairing_cannot_remove_required_identity_checks(keys, field):
    left, right = row(factor='f'), row(factor='f')
    right[field] = 5 if field == 'new_tokens' else 'changed'
    result = compare([left], [right], {'keys': keys})
    assert result['incomparable'] == 1 and result['wins'] == result['losses'] == 0
    assert result['pairs'][0]['reasons'] == [f'{field} mismatch']


@pytest.mark.parametrize('same', [True, False])
def test_custom_pairing_adds_factor_once_without_changing_valid_pairs(same):
    result = compare([row(factor='f')], [row(factor='f' if same else 'other')],
                     {'keys':['factor', 'factor', 'protocol']})
    assert result['pairs'][0]['reasons'] == ([] if same else ['factor mismatch'])
    assert result['ties'] == int(same) and result['incomparable'] == int(not same)
