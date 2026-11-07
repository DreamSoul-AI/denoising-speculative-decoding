"""Pure scalar full-AR formulas and roles remain distinct from verifier-argmax quality."""
import pytest

from studies.esp_table1_bf16_full.structure import readout as full
from studies.esp_table1_bf16_full_argmax.structure import readout as argmax


def test_full_reuses_count_formulas_without_inheriting_argmax_cell_roles():
    rows = [{'category': 'a', 'committed': 4, 'decode_calls': 4, 'accepted': 0, 'exact': False},
            {'category': 'a', 'committed': 4, 'decode_calls': 2, 'accepted': 2, 'exact': True},
            {'category': 'b', 'committed': 9, 'decode_calls': 3, 'accepted': 6, 'exact': False}]
    summary = argmax.count_summary(rows)
    assert summary['pooled_tau'] == pytest.approx(17 / 9)
    assert summary['prompt_mean_tau'] == 2
    assert summary['category_mean_pooled_tau'] == pytest.approx((8 / 6 + 3) / 2)
    assert summary['main_calls'] == 12 and summary['exact_count'] == 1
    assert argmax.ordered_cells({}, declarations=full.CELLS)[0]['role'] == 'auxiliary'
    assert argmax.ordered_cells({})[0]['role'] == 'primary'


def test_full_saved_ar_mismatch_is_not_hidden_by_an_argmax_pass():
    prompt = {'question_id': 81, 'category': 'a', 'input_ids': [1, 2]}
    row = {'category': 'a', 'committed': 1, 'decode_calls': 1, 'accepted': 0,
           'spec_time': 0, 'ar_time': 0, 'exact_match': 0}
    evidence = {'input_ids': [1, 2], 'output_ids': [1, 2, 3, 4], 'ar_ids': [1, 2, 3, 9],
                'stats': {'block_complexity': 30, 'committed': 1, 'decode_calls': 1, 'accepted': 0,
                          'verification_argmax_audit': {'passed': True}}}
    rows, audits = full.native_rows({'rows': [row], 'evidence': [evidence]}, [prompt], {'block_complexity': 30})
    assert rows[0]['exact'] is False and audits[0]['first_new_difference_index'] == 1
    assert argmax.count_summary(rows)['exact_count'] == 0
