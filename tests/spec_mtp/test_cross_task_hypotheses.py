"""H1/H3 verdicts require complete fixed units and legal AR-first-token ancestry."""

import copy
import importlib.util
import pytest
from test_window_observer import ROOT

specification = importlib.util.spec_from_file_location('cross_task_checker',
    ROOT / 'studies/output_space_cross_task/check_hypotheses.py')
checker = importlib.util.module_from_spec(specification)
specification.loader.exec_module(checker)


def legal_transfer(native, diagnostic):
    case = diagnostic['prompts'][0]['cases'][0]
    case['replay']['trace'][0].update(accepted_nodes=[1, 2], cache_after=4, committed_tokens=[10, 11, 12])
    case['windows'] = [{'root_position': 1, 'recipient_baseline_visited': True,
        'baseline_first_slot': {'top1_id': 11, 'target_id': 11},
        'slot_rankings': {'first': {'top1_id': 11, 'target_id': 11},
                         'second': {'top1_id': 12, 'target_id': 12},
                         'second_AR_path_valid': True, 'seed_logits_finite': True}}]
    return case


def test_complete_null_falsifies_only_bounded_H1():
    native, diagnostic = checker.fixture()
    result = checker.verdict(native, diagnostic, [])
    assert result['H1'] == 'falsified_bounded_occurrence'
    assert result['H2'].startswith('not_tested') and not result['legal_depth2_witnesses']


def test_count_equal_semantic_difference_is_observed():
    native, diagnostic = checker.fixture()
    native['runs'][3]['raw']['evidence'][0]['stats']['trace'][0]['stopped'] = True
    update_references(native, diagnostic)
    assert checker.verdict(native, diagnostic, [])['H1'] == 'observed'


@pytest.mark.parametrize('boundary', ['wrong_first', 'EOS', 'missing_second_target', 'legal'])
def test_depth2_witness_excludes_invalid_first_EOS_and_missing_target(boundary):
    native, diagnostic = checker.fixture()
    case = legal_transfer(native, diagnostic)
    eos = []
    if boundary == 'wrong_first':
        case['windows'][0]['slot_rankings']['first']['top1_id'] = 99
    elif boundary == 'EOS':
        eos = [11]
    elif boundary == 'missing_second_target':
        for run in native['runs']:
            run['raw']['evidence'][0]['output_ids'] = [0, 10, 11]
    result = checker.verdict(native, diagnostic, eos)
    assert bool(result['legal_depth2_witnesses']) == (boundary == 'legal')


def test_infeasible_is_not_a_null_transfer():
    native, diagnostic = checker.fixture()
    diagnostic['prompts'][0].update(intervention_root=None, cases=[])
    result = checker.verdict(native, diagnostic, [])
    assert result['H1'] == 'inconclusive_infeasible_windows'
    assert result['infeasible_question_ids'] == [241]


@pytest.mark.parametrize('failure', ['quality', 'missing_unit', 'missing_case', 'duplicate_case', 'failed_placebo'])
def test_incomplete_or_failed_gate_cannot_produce_a_verdict(failure):
    native, diagnostic = checker.fixture()
    if failure == 'quality':
        diagnostic['quality_passed'] = False
    elif failure == 'missing_unit':
        diagnostic['prompts'].pop()
    elif failure == 'missing_case':
        diagnostic['prompts'][0]['cases'].pop()
    elif failure == 'duplicate_case':
        diagnostic['prompts'][0]['cases'][1] = copy.deepcopy(diagnostic['prompts'][0]['cases'][0])
    else:
        diagnostic['prompts'][0]['cases'][2]['replay']['placebo_full_semantic_equal'] = False
    with pytest.raises(AssertionError):
        checker.verdict(native, diagnostic, [])


def test_natural_depth2_witness():
    native, diagnostic = checker.fixture()
    native['runs'][3]['raw']['evidence'][0]['stats']['trace'][0].update(
        accepted_nodes=[1, 2], cache_after=4, committed_tokens=[10, 11, 12])
    rank = {'first': {'top1_id': 11, 'target_id': 11}, 'second_AR_path_valid': True}
    diagnostic['prompts'][0]['natural_baseline_margin_pairs'] = [
        {'root_position': 1, 'zero_slots': copy.deepcopy(rank), 'one_slots': copy.deepcopy(rank)}]
    update_references(native, diagnostic)
    result = checker.verdict(native, diagnostic, [])
    assert result['legal_depth2_witnesses'][0]['label'] == 'natural_output0_vs_output1'


def update_references(native, diagnostic):
    refs = {r['case']: r['raw']['evidence'][0]['stats']['trace'] for r in native['runs']}
    prompt = diagnostic['prompts'][0]
    for baseline in prompt['baselines']:
        baseline['trace'] = copy.deepcopy(refs[baseline['case']])
    for case in prompt['cases']:
        if case['recipient'] == case['donor']:
            case['replay']['trace'] = copy.deepcopy(refs[case['recipient']])


@pytest.mark.parametrize('target', ['baseline', 'placebo'])
def test_true_quality_flags_do_not_override_trace_mismatch(target):
    native, diagnostic = checker.fixture()
    if target == 'baseline':
        diagnostic['prompts'][0]['baselines'][0]['trace'][0]['stopped'] = True
    else:
        diagnostic['prompts'][0]['cases'][2]['replay']['trace'][0]['stopped'] = True
    with pytest.raises(AssertionError):
        checker.verdict(native, diagnostic, [])
