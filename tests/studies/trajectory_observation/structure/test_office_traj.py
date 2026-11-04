"""Slice and observation switch for the Qwen3-8B trajectory ablation."""

import json

import pytest
import torch

from spec_mtp.structure.algorithm.mask import build_mask_provider
from studies.trajectory_observation.structure.office_qwen8b import ESP
from studies.trajectory_observation.structure.office_tree import SMOKE
from studies.trajectory_observation.structure.office_traj import (
    CATEGORY_XCAT, CATEGORY_XCAT2, CATEGORY_XCAT3, CATEGORY_XCAT4, CATEGORY_XCAT5, CATEGORY_XCAT6,
    CATEGORY_XCAT7, EVAL_QIDS, EVAL_QIDS_XCAT, EVAL_QIDS_XCAT2, EVAL_QIDS_XCAT3, EVAL_QIDS_XCAT4,
    EVAL_QIDS_XCAT5, EVAL_QIDS_XCAT6, EVAL_QIDS_XCAT7, EVAL_SHA256, EVAL_SHA256_XCAT, EVAL_SHA256_XCAT2,
    EVAL_SHA256_XCAT3, EVAL_SHA256_XCAT4, EVAL_SHA256_XCAT5, EVAL_SHA256_XCAT6, EVAL_SHA256_XCAT7,
    GAIN_FULL, GAIN_TRAJ, GSCAN_PREFIX, GSCAN_STAMP, MASK, PROTOCOL, PROTOCOL_GSCAN, PROTOCOL_XCAT,
    PROTOCOL_XCAT2, PROTOCOL_XCAT3, PROTOCOL_XCAT4,
    PROTOCOL_XCAT5, PROTOCOL_XCAT6, PROTOCOL_XCAT7, SKIPPED_QIDS, SKIPPED_QIDS_XCAT, SKIPPED_QIDS_XCAT2,
    SKIPPED_QIDS_XCAT3, SKIPPED_QIDS_XCAT4, SKIPPED_QIDS_XCAT5, SKIPPED_QIDS_XCAT6, SKIPPED_QIDS_XCAT7,
    TAKE, TAKE_XCAT, TAKE_XCAT2, TAKE_XCAT3, TAKE_XCAT4, TAKE_XCAT5, TAKE_XCAT6, TAKE_XCAT7, TREE,
    GROUPS, MISSING_FIELDS, ablation_summary, acceptance_label, arm_spec, arm_specs, choose_conclusion,
    classify_pair, decode_measured_prompt, describe_paired, finite_wall, jaccard, jaccard_band,
    load_local_tokenizer, measure_generate, pair_relation, prompts_brainstorming, prompts_classification,
    prompts_closed_qa, prompts_from_647, prompts_general_qa, prompts_information_extraction, prompts_open_qa,
    prompts_summarization, report_group, wall_group, wall_report, round_config,
    gscan_arm_spec, gscan_formal, gscan_limit, gscan_smoke, gscan_specs, gscan_summary,
)
from studies.trajectory_observation.structure.office_traj_length import (
    LENGTHS, LENGTH_PREFIX, PROTOCOL_LENGTH, length_decision, row_buckets,
)


def _creative(text, context=''):
    return {'category': 'creative_writing', 'instruction': text, 'context': context}


def test_prompts_from_647_skip_texts_already_seen():
    rows = [_creative(f'write {index}') for index in range(647)]
    rows.append(_creative('write 408'))
    rows.append(_creative('fresh'))
    rows.append(_creative('write 198'))
    rows.append(_creative('later'))
    chosen, skipped = prompts_from_647(rows, take=2)
    assert skipped == (647, 649)
    assert [item['qid'] for item in chosen] == [648, 650]


def test_short_pool_from_647_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_from_647([_creative(f'write {index}') for index in range(660)], take=TAKE)


def _row(category, text, context=''):
    return {'category': category, 'instruction': text, 'context': context}


def test_open_qa_skips_creative_duplicate_template_and_specbench():
    specbench = ('Compose an engaging travel blog post about a recent trip to Hawaii, '
                 'highlighting cultural experiences and must-see attractions.')
    template = 'Name the color of a common example'
    rows = [
        _row('creative_writing', 'used once'),
        _row('open_qa', 'used once'),
        _row('open_qa', 'fresh', 'ctx'),
        _row('open_qa', 'ctx\n\nfresh'),
        _row('open_qa', template),
        _row('open_qa', specbench),
        _row('open_qa', 'later'),
    ]
    chosen, skipped = prompts_open_qa(rows, take=2)
    assert skipped == (0, 2, 3, 4)
    assert chosen == [
        {'qid': 1, 'category': 'open_qa', 'text': 'ctx\n\nfresh'},
        {'qid': 5, 'category': 'open_qa', 'text': 'later'},
    ]


def test_short_open_qa_pool_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_open_qa([_row('open_qa', 'only one')], take=TAKE_XCAT)


def test_general_qa_skips_open_qa_creative_template_and_specbench():
    specbench = ('Compose an engaging travel blog post about a recent trip to Hawaii, '
                 'highlighting cultural experiences and must-see attractions.')
    template = 'Name the color of a common example'
    rows = [_row('open_qa', f'open {index}') for index in range(128)]
    rows.extend([
        _row('creative_writing', 'used once'),
        _row('general_qa', 'open 0'),
        _row('general_qa', 'used once'),
        _row('general_qa', 'fresh', 'ctx'),
        _row('general_qa', template),
        _row('general_qa', specbench),
        _row('general_qa', 'later'),
    ])
    chosen, skipped = prompts_general_qa(rows, take=2)
    assert skipped == (0, 1, 3, 4)
    assert chosen == [
        {'qid': 2, 'category': 'general_qa', 'text': 'ctx\n\nfresh'},
        {'qid': 5, 'category': 'general_qa', 'text': 'later'},
    ]


def test_short_general_qa_pool_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_general_qa([_row('general_qa', 'only one')], take=TAKE_XCAT2)


def test_classification_skips_prior_slices_template_and_specbench():
    specbench = ('Compose an engaging travel blog post about a recent trip to Hawaii, '
                 'highlighting cultural experiences and must-see attractions.')
    template = 'Name the color of a common example'
    rows = [_row('open_qa', f'open {index}') for index in range(128)]
    rows.extend(_row('general_qa', f'general {index}') for index in range(128))
    rows.extend([
        _row('creative_writing', 'used once'),
        _row('classification', 'open 0'),
        _row('classification', 'general 0'),
        _row('classification', 'used once'),
        _row('classification', 'fresh', 'ctx'),
        _row('classification', template),
        _row('classification', specbench),
        _row('classification', 'later'),
    ])
    chosen, skipped = prompts_classification(rows, take=2)
    assert skipped == (0, 1, 2, 4, 5)
    assert chosen == [
        {'qid': 3, 'category': 'classification', 'text': 'ctx\n\nfresh'},
        {'qid': 6, 'category': 'classification', 'text': 'later'},
    ]


def test_short_classification_pool_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_classification([_row('classification', 'only one')], take=TAKE_XCAT3)


def test_closed_qa_skips_prior_slices_template_and_specbench():
    specbench = ('Compose an engaging travel blog post about a recent trip to Hawaii, '
                 'highlighting cultural experiences and must-see attractions.')
    template = 'Name the color of a common example'
    rows = [_row('open_qa', f'open {index}') for index in range(128)]
    rows.extend(_row('general_qa', f'general {index}') for index in range(128))
    rows.extend(_row('classification', f'class {index}') for index in range(128))
    rows.extend([
        _row('creative_writing', 'used once'),
        _row('closed_qa', 'open 0'),
        _row('closed_qa', 'general 0'),
        _row('closed_qa', 'class 0'),
        _row('closed_qa', 'used once'),
        _row('closed_qa', 'fresh', 'ctx'),
        _row('closed_qa', template),
        _row('closed_qa', specbench),
        _row('closed_qa', 'later'),
    ])
    chosen, skipped = prompts_closed_qa(rows, take=2)
    assert skipped == (0, 1, 2, 3, 5, 6)
    assert chosen == [
        {'qid': 4, 'category': 'closed_qa', 'text': 'ctx\n\nfresh'},
        {'qid': 7, 'category': 'closed_qa', 'text': 'later'},
    ]


def test_short_closed_qa_pool_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_closed_qa([_row('closed_qa', 'only one')], take=TAKE_XCAT4)


def test_brainstorming_skips_prior_slices_template_and_specbench():
    specbench = ('Compose an engaging travel blog post about a recent trip to Hawaii, '
                 'highlighting cultural experiences and must-see attractions.')
    template = 'Name the color of a common example'
    rows = [_row('open_qa', f'open {index}') for index in range(128)]
    rows.extend(_row('general_qa', f'general {index}') for index in range(128))
    rows.extend(_row('classification', f'class {index}') for index in range(128))
    rows.extend(_row('closed_qa', f'closed {index}') for index in range(128))
    rows.extend([
        _row('creative_writing', 'used once'),
        _row('brainstorming', 'open 0'),
        _row('brainstorming', 'general 0'),
        _row('brainstorming', 'class 0'),
        _row('brainstorming', 'closed 0'),
        _row('brainstorming', 'used once'),
        _row('brainstorming', 'fresh', 'ctx'),
        _row('brainstorming', template),
        _row('brainstorming', specbench),
        _row('brainstorming', 'later'),
    ])
    chosen, skipped = prompts_brainstorming(rows, take=2)
    assert skipped == (0, 1, 2, 3, 4, 6, 7)
    assert chosen == [
        {'qid': 5, 'category': 'brainstorming', 'text': 'ctx\n\nfresh'},
        {'qid': 8, 'category': 'brainstorming', 'text': 'later'},
    ]


def test_short_brainstorming_pool_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_brainstorming([_row('brainstorming', 'only one')], take=TAKE_XCAT5)


def test_information_extraction_skips_prior_slices_template_and_specbench():
    specbench = ('Compose an engaging travel blog post about a recent trip to Hawaii, '
                 'highlighting cultural experiences and must-see attractions.')
    template = 'Name the color of a common example'
    rows = [_row('open_qa', f'open {index}') for index in range(128)]
    rows.extend(_row('general_qa', f'general {index}') for index in range(128))
    rows.extend(_row('classification', f'class {index}') for index in range(128))
    rows.extend(_row('closed_qa', f'closed {index}') for index in range(128))
    rows.extend(_row('brainstorming', f'brain {index}') for index in range(128))
    rows.extend([
        _row('creative_writing', 'used once'),
        _row('information_extraction', 'open 0'),
        _row('information_extraction', 'general 0'),
        _row('information_extraction', 'class 0'),
        _row('information_extraction', 'closed 0'),
        _row('information_extraction', 'brain 0'),
        _row('information_extraction', 'used once'),
        _row('information_extraction', 'fresh', 'ctx'),
        _row('information_extraction', template),
        _row('information_extraction', specbench),
        _row('information_extraction', 'later'),
    ])
    chosen, skipped = prompts_information_extraction(rows, take=2)
    assert skipped == (0, 1, 2, 3, 4, 5, 7, 8)
    assert chosen == [
        {'qid': 6, 'category': 'information_extraction', 'text': 'ctx\n\nfresh'},
        {'qid': 9, 'category': 'information_extraction', 'text': 'later'},
    ]


def test_short_information_extraction_pool_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_information_extraction([_row('information_extraction', 'only one')], take=TAKE_XCAT6)


def test_summarization_skips_prior_slices_template_and_specbench():
    specbench = ('Compose an engaging travel blog post about a recent trip to Hawaii, '
                 'highlighting cultural experiences and must-see attractions.')
    template = 'Name the color of a common example'
    rows = [_row('open_qa', f'open {index}') for index in range(128)]
    rows.extend(_row('general_qa', f'general {index}') for index in range(128))
    rows.extend(_row('classification', f'class {index}') for index in range(128))
    rows.extend(_row('closed_qa', f'closed {index}') for index in range(128))
    rows.extend(_row('brainstorming', f'brain {index}') for index in range(128))
    rows.extend(_row('information_extraction', f'extract {index}') for index in range(128))
    rows.extend([
        _row('creative_writing', 'used once'),
        _row('summarization', 'open 0'),
        _row('summarization', 'general 0'),
        _row('summarization', 'class 0'),
        _row('summarization', 'closed 0'),
        _row('summarization', 'brain 0'),
        _row('summarization', 'extract 0'),
        _row('summarization', 'used once'),
        _row('summarization', 'fresh', 'ctx'),
        _row('summarization', template),
        _row('summarization', specbench),
        _row('summarization', 'later'),
    ])
    chosen, skipped = prompts_summarization(rows, take=2)
    assert skipped == (0, 1, 2, 3, 4, 5, 6, 8, 9)
    assert chosen == [
        {'qid': 7, 'category': 'summarization', 'text': 'ctx\n\nfresh'},
        {'qid': 10, 'category': 'summarization', 'text': 'later'},
    ]


def test_short_summarization_pool_is_rejected():
    with pytest.raises(RuntimeError):
        prompts_summarization([_row('summarization', 'only one')], take=TAKE_XCAT7)


def test_registered_summarization_slice_has_no_skips():
    assert PROTOCOL_XCAT7 == 'office-8b-traj-xcat7-v1'
    assert CATEGORY_XCAT7 == 'summarization'
    assert TAKE_XCAT7 == 128
    assert SKIPPED_QIDS_XCAT7 == ()
    assert EVAL_QIDS_XCAT7 == tuple(range(128))
    assert len(EVAL_QIDS_XCAT7) == TAKE_XCAT7
    assert EVAL_SHA256_XCAT7 == 'f35b56bf668124a609c47dc51f97d40cabd75fdac5e6a3eb0fd133a25665020a'


def test_registered_information_extraction_slice_has_no_skips():
    assert PROTOCOL_XCAT6 == 'office-8b-traj-xcat6-v1'
    assert CATEGORY_XCAT6 == 'information_extraction'
    assert TAKE_XCAT6 == 128
    assert SKIPPED_QIDS_XCAT6 == ()
    assert EVAL_QIDS_XCAT6 == tuple(range(128))
    assert len(EVAL_QIDS_XCAT6) == TAKE_XCAT6
    assert EVAL_SHA256_XCAT6 == 'b7094d9ecf62a73e299d742016a3fdfcf4608ef5a7a32f7adc7703b86706d7c3'


def test_registered_brainstorming_slice_has_no_skips():
    assert PROTOCOL_XCAT5 == 'office-8b-traj-xcat5-v1'
    assert CATEGORY_XCAT5 == 'brainstorming'
    assert TAKE_XCAT5 == 128
    assert SKIPPED_QIDS_XCAT5 == ()
    assert EVAL_QIDS_XCAT5 == tuple(range(128))
    assert len(EVAL_QIDS_XCAT5) == TAKE_XCAT5
    assert EVAL_SHA256_XCAT5 == '3dcf4c093d586cd329f7393213009962308365098a9790b009cafc32c7f37869'


def test_registered_closed_qa_slice_has_no_skips():
    assert PROTOCOL_XCAT4 == 'office-8b-traj-xcat4-v1'
    assert CATEGORY_XCAT4 == 'closed_qa'
    assert TAKE_XCAT4 == 128
    assert SKIPPED_QIDS_XCAT4 == ()
    assert EVAL_QIDS_XCAT4 == tuple(range(128))
    assert len(EVAL_QIDS_XCAT4) == TAKE_XCAT4
    assert EVAL_SHA256_XCAT4 == '55ad2a82a6853587c7f28a641ba5acbdcc66d374c1b2cc3924c1563ce3fafdf9'


def test_registered_classification_slice_has_no_skips():
    assert PROTOCOL_XCAT3 == 'office-8b-traj-xcat3-v1'
    assert CATEGORY_XCAT3 == 'classification'
    assert TAKE_XCAT3 == 128
    assert SKIPPED_QIDS_XCAT3 == ()
    assert EVAL_QIDS_XCAT3 == tuple(range(128))
    assert len(EVAL_QIDS_XCAT3) == TAKE_XCAT3
    assert EVAL_SHA256_XCAT3 == '01c21dbf4a774b2395bed3a8606d3fe20d92a0cb7d2562ff8e8d965b9f2bd35e'


def test_registered_general_qa_slice_skips_only_the_open_qa_duplicate():
    assert PROTOCOL_XCAT2 == 'office-8b-traj-xcat2-v1'
    assert CATEGORY_XCAT2 == 'general_qa'
    assert TAKE_XCAT2 == 128
    assert SKIPPED_QIDS_XCAT2 == (91,)
    assert EVAL_QIDS_XCAT2 == tuple(range(91)) + tuple(range(92, 129))
    assert len(EVAL_QIDS_XCAT2) == TAKE_XCAT2
    assert EVAL_SHA256_XCAT2 == 'a6a306ea868767da6cf8b6006bb4d91c5360387ae8b16ba1a20142cdd35ec677'


def test_registered_open_qa_head_is_128_with_no_skips():
    assert PROTOCOL_XCAT == 'office-8b-traj-xcat-v1'
    assert CATEGORY_XCAT == 'open_qa'
    assert TAKE_XCAT == 128
    assert SKIPPED_QIDS_XCAT == ()
    assert EVAL_QIDS_XCAT == tuple(range(128))
    assert EVAL_SHA256_XCAT == 'f6e6b1c71a20f1170692e22a2c2bd79fb31eaeb810b861553f7c76a41df25231'


def test_registered_tail_is_sixty_prompts_after_two_skips():
    assert PROTOCOL == 'office-8b-traj-ablate-v1'
    assert TAKE == 60
    assert SKIPPED_QIDS == (658, 678)
    assert EVAL_QIDS == tuple(qid for qid in range(647, 709) if qid not in SKIPPED_QIDS)
    assert len(EVAL_QIDS) == TAKE
    assert EVAL_QIDS[0] == 647 and EVAL_QIDS[-1] == 708
    assert EVAL_SHA256 == 'd23ad590c45307fa7c1981b272adaec243bfaca5a2e390eda394bf2fe42a4cca'


def test_arms_share_the_formal_tree_and_only_the_observation_gain_changes():
    specs = arm_specs()
    assert specs['full']['tree'] == specs['traj']['tree'] == ESP['tree']
    assert specs['full']['mask'] == specs['traj']['mask'] == ESP['mask']
    assert specs['full']['num_masks'] == specs['traj']['num_masks'] == 1
    assert specs['full']['observation']['gain'] == GAIN_FULL == 0.5
    assert specs['traj']['observation']['gain'] == GAIN_TRAJ == 0.0
    assert specs['full']['observation']['condition'] == specs['traj']['observation']['condition'] == 'none'
    assert specs['full']['observation']['form'] == specs['traj']['observation']['form'] == 'mean'
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    full = build_mask_provider(specs['full'], table)
    traj = build_mask_provider(specs['traj'], table)
    assert full.protocol == traj.protocol == 'trajectory-observation-blend-v1'
    assert full.gain == 0.5 and traj.gain == 0.0
    with pytest.raises(RuntimeError):
        arm_spec(1.0)


def _arm(qid, accepted):
    return {'qid': qid, 'accepted': accepted, 'committed': accepted + 8, 'decode_calls': 8,
            'tau': (accepted + 8) / 8, 'argmax_passed': True, 'ar_equal': True,
            'max_nodes': 4, 'block_complexity': 10}


def test_summary_defines_traj_minus_full_and_the_opposite_full_gain():
    full = [_arm(qid, 2) for qid in range(TAKE)]
    traj = [_arm(qid, 1 if qid < 40 else 2) for qid in range(TAKE)]
    summary = ablation_summary({'full': full, 'traj': traj})
    comparison = summary['comparison']
    assert summary['role'] == 'exploration' and summary['confirmatory'] is False
    assert summary['prompts'] == TAKE
    assert comparison['wins_losses_ties'] == [0, 40, 20]
    assert comparison['direction'] == 'negative'
    assert comparison['accepted_difference']['mean'] == -40 / TAKE
    assert comparison['full_gain_difference']['mean'] == 40 / TAKE
    assert comparison['accepted_difference']['ci95'][1] < 0
    assert comparison['full_gain_difference']['ci95'][0] > 0
    assert comparison['max_nodes'] == 4
    assert comparison['block_complexity'] == {'full': [10], 'traj': [10]}
    assert summary['wall'] == {'complete': False}
    assert 'wall_direction' not in summary['wall']
    assert 'wall_protocol' not in summary


def _pair(accepted_full, accepted_traj, wall_full=None, wall_traj=None):
    full = [_arm(qid, accepted_full) for qid in range(2)]
    traj = [_arm(qid, accepted_traj) for qid in range(2)]
    if wall_full is not None:
        for row in full:
            row['wall_seconds'] = wall_full
    if wall_traj is not None:
        for row in traj:
            row['wall_seconds'] = wall_traj
    return full, traj


def test_wall_report_keeps_a_missing_clock_missing():
    full, traj = _pair(1, 4, wall_full=2.0, wall_traj=1.0)
    traj[1]['wall_seconds'] = float('nan')
    report = wall_report(full, traj)
    assert report == {'complete': False}
    assert finite_wall(None) is None
    assert finite_wall(float('inf')) is None
    assert finite_wall(-0.1) is None
    assert finite_wall(True) is None
    assert finite_wall(0.0) == 0.0
    summary = ablation_summary(
        {'full': full, 'traj': traj}, protocol=PROTOCOL_XCAT7, digest=EVAL_SHA256_XCAT7,
        skipped=SKIPPED_QIDS_XCAT7, take=2)
    assert summary['wall_protocol'] == 'office-8b-traj-wall-v1'
    assert summary['wall'] == {'complete': False}
    assert summary['comparison']['direction'] == 'positive'
    assert summary['comparison']['accepted_difference']['mean'] == 3


def test_wall_direction_uses_the_shorter_count_and_not_the_acceptance_sign():
    full, traj = _pair(1, 4, wall_full=1.0, wall_traj=2.0)
    summary = ablation_summary({'full': full, 'traj': traj}, take=2)
    wall = summary['wall']
    assert summary['comparison']['direction'] == 'positive'
    assert wall['complete'] is True
    assert wall['wins_losses_ties'] == [0, 2, 0]
    assert wall['wall_direction'] == 'longer'
    assert wall['delta'] == 'wall_seconds_traj - wall_seconds_full'
    assert wall['wall_difference']['mean'] == 1.0
    full, traj = _pair(1, 4, wall_full=2.0, wall_traj=1.0)
    assert ablation_summary({'full': full, 'traj': traj}, take=2)['wall']['wall_direction'] == 'shorter'
    full, traj = _pair(1, 1, wall_full=1.5, wall_traj=1.5)
    flat = ablation_summary({'full': full, 'traj': traj}, take=2)['wall']
    assert flat['wall_direction'] == 'flat'
    assert flat['p'] is None
    mixed_full = [_arm(0, 1), _arm(1, 1)]
    mixed_traj = [_arm(0, 1), _arm(1, 1)]
    mixed_full[0]['wall_seconds'] = 1.0
    mixed_full[1]['wall_seconds'] = 1.0
    mixed_traj[0]['wall_seconds'] = 0.9
    mixed_traj[1]['wall_seconds'] = 2.0
    mixed = wall_report(mixed_full, mixed_traj)
    assert mixed['wall_direction'] == 'mixed'
    assert mixed['wins_losses_ties'] == [1, 1, 0]


def test_measure_generate_syncs_around_one_call_and_drops_a_bad_clock(monkeypatch):
    from studies.trajectory_observation.structure import office_traj

    events = []

    def sync(device):
        events.append(('sync', device))

    def generate(ids):
        events.append(('generate', ids))
        return 'out', {'accepted': 3, 'prefill_time': 0.2, 'decode_time': 0.3}

    ticks = iter([10.0, 10.5])
    monkeypatch.setattr(office_traj.time, 'perf_counter', lambda: next(ticks))
    monkeypatch.setattr('spec_mtp.model.sync_device', sync)
    output, stats, wall = measure_generate(generate, 'ids', 'cpu')
    assert events == [('sync', 'cpu'), ('generate', 'ids'), ('sync', 'cpu')]
    assert output == 'out' and stats['accepted'] == 3 and wall == 0.5

    events.clear()

    def fail_first(device):
        events.append('sync')
        raise RuntimeError('sync failed')

    monkeypatch.setattr('spec_mtp.model.sync_device', fail_first)
    output, _stats, wall = measure_generate(generate, 'ids', 'cpu')
    assert output == 'out' and wall is None
    assert events == ['sync', ('generate', 'ids')]

    calls = {'n': 0}

    def fail_second(device):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError('sync failed')

    monkeypatch.setattr('spec_mtp.model.sync_device', fail_second)
    output, _stats, wall = measure_generate(generate, 'ids', 'cuda')
    assert output == 'out' and wall is None

    monkeypatch.setattr('spec_mtp.model.sync_device', lambda device: None)
    backward = iter([5.0, 4.0])
    monkeypatch.setattr(office_traj.time, 'perf_counter', lambda: next(backward))
    _output, _stats, wall = measure_generate(generate, 'ids', 'cpu')
    assert wall is None
    broken = iter([1.0, float('nan')])
    monkeypatch.setattr(office_traj.time, 'perf_counter', lambda: next(broken))
    _output, _stats, wall = measure_generate(generate, 'ids', 'cpu')
    assert wall is None


def test_measured_prompt_times_full_then_traj_and_leaves_ar_outside(monkeypatch):
    import spec_mtp.structure.algorithm.decode.loop as decode_mod
    from studies.trajectory_observation.structure import office_traj

    events = []

    class _Tree:
        num_nodes = 4

    class _Model:
        device = 'cpu'
        embedding_table = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    class _Batch:
        input_ids = torch.ones(1, 2, dtype=torch.long)

    class _Tokenizer:
        def __call__(self, text, return_tensors='pt'):
            events.append(('tokenize', text))
            return _Batch()

    class _Decoder:
        def __init__(self, model, spec, limit, mask_policy=None):
            self.gain = spec['observation']['gain']

        def generate(self, ids):
            decode_mod.build_tree(None, None, 0)
            events.append(('generate', self.gain))
            stats = {
                'verification_argmax_audit': {'passed': True},
                'block_complexity': 10,
                'decode_calls': 2,
                'accepted': 4,
                'committed': 8,
                'prefill_time': 0.1,
                'decode_time': float('nan'),
            }
            return torch.tensor([[1, 2, 3]]), stats

    def ar_generate(model, ids, limit):
        events.append('ar')
        return torch.tensor([[1, 2, 3]]), {'ar_time': 9.0}

    def sync(device):
        events.append(('sync', device))

    ticks = iter([10.0, 10.5, 20.0, 21.0])
    monkeypatch.setattr(decode_mod, 'build_tree', lambda *args, **kwargs: _Tree())
    monkeypatch.setattr(decode_mod, 'SpecDecoder', _Decoder)
    monkeypatch.setattr(decode_mod, 'ar_generate', ar_generate)
    monkeypatch.setattr('spec_mtp.model.sync_device', sync)
    monkeypatch.setattr(office_traj.time, 'perf_counter', lambda: next(ticks))
    prompt = {'qid': 3, 'category': 'summarization', 'text': 'sum'}
    decoded = decode_measured_prompt(_Model(), _Tokenizer(), prompt, tuple(arm_specs().items()), SMOKE)
    assert [event for event in events if event == 'ar' or (isinstance(event, tuple) and event[0] in {'sync', 'generate'})] == [
        'ar',
        ('sync', 'cpu'), ('generate', 0.5), ('sync', 'cpu'),
        ('sync', 'cpu'), ('generate', 0.0), ('sync', 'cpu'),
    ]
    assert decoded['full']['wall_seconds'] == 0.5
    assert decoded['traj']['wall_seconds'] == 1.0
    assert decoded['full']['prefill_time'] == 0.1
    assert 'decode_time' not in decoded['full']
    assert decoded['full']['accepted'] == 4
    assert 'ar_time' not in decoded['full']
    assert decoded['full']['ar_equal'] is True
    assert decoded['full']['argmax_passed'] is True
    assert decoded['full']['prompt_length'] == 2
    assert decoded['full']['first_divergence_index'] is None
    assert decoded['full']['common_prefix_length'] == 3
    assert 'side_tokens' not in decoded['full']
    assert 'output_ids' not in decoded['full']

def test_measured_prompt_records_a_shared_fork_without_touching_argmax(monkeypatch):
    import spec_mtp.structure.algorithm.decode.loop as decode_mod

    class _Tree:
        num_nodes = 4

    class _Model:
        device = 'cpu'
        embedding_table = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    class _Batch:
        input_ids = torch.ones(1, 4, dtype=torch.long)

    class _Tokenizer:
        def __call__(self, text, return_tensors='pt'):
            return _Batch()

    class _Decoder:
        def __init__(self, model, spec, limit, mask_policy=None):
            self.gain = spec['observation']['gain']

        def generate(self, ids):
            decode_mod.build_tree(None, None, 0)
            stats = {
                'verification_argmax_audit': {'passed': True},
                'block_complexity': 10,
                'decode_calls': 2,
                'accepted': 4,
                'committed': 8,
            }
            tokens = [1, 2, 9, 4] if self.gain == 0.5 else [1, 2, 9, 8]
            return torch.tensor([tokens]), stats

    def ar_generate(model, ids, limit):
        return torch.tensor([[1, 2, 3, 4]]), {'ar_time': 9.0}

    monkeypatch.setattr(decode_mod, 'build_tree', lambda *args, **kwargs: _Tree())
    monkeypatch.setattr(decode_mod, 'SpecDecoder', _Decoder)
    monkeypatch.setattr(decode_mod, 'ar_generate', ar_generate)
    monkeypatch.setattr('spec_mtp.model.sync_device', lambda device: None)
    prompt = {'qid': 3, 'category': 'summarization', 'text': 'sum'}
    decoded = decode_measured_prompt(
        _Model(), _Tokenizer(), prompt, tuple(arm_specs().items()), SMOKE, keep_sequences=True)
    assert decoded['full']['argmax_passed'] is True
    assert decoded['full']['ar_equal'] is False
    assert decoded['traj']['ar_equal'] is False
    assert decoded['full']['first_divergence_index'] == 2
    assert decoded['full']['common_prefix_length'] == 2
    assert decoded['full']['side_tokens'] == {'reference': 3, 'candidate': 9}
    assert decoded['traj']['side_tokens'] == {'reference': 3, 'candidate': 9}
    assert decoded['full']['output_ids'] == [1, 2, 9, 4]
    assert decoded['full']['reference_ids'] == [1, 2, 3, 4]
    assert decoded['full']['output_sha256'] != decoded['traj']['output_sha256']


def test_groups_keep_the_chapter_table_and_the_archived_xcat7_names():
    by_key = {group['key']: group for group in GROUPS}
    assert [group['key'] for group in GROUPS] == [
        'ablate', 'xcat', 'xcat2', 'xcat3', 'xcat4', 'xcat5', 'xcat6', 'xcat7']
    assert [group['archived_losses'] for group in GROUPS] == [5, 12, 17, 23, 22, 17, 10, 17]
    assert [group['sha256'] for group in GROUPS] == [
        EVAL_SHA256, EVAL_SHA256_XCAT, EVAL_SHA256_XCAT2, EVAL_SHA256_XCAT3, EVAL_SHA256_XCAT4,
        EVAL_SHA256_XCAT5, EVAL_SHA256_XCAT6, EVAL_SHA256_XCAT7]
    assert by_key['xcat7']['prefix'] == 'sd-office-8b-traj-xcat7'
    assert by_key['xcat7']['stamp'] == '20261104-045236'
    assert by_key['ablate']['stamp'] == '20261103-0905'
    assert by_key['xcat6']['prefix'] == 'sd-office-8b-traj-xcat6'


def _group():
    return {
        'protocol': 'office-8b-traj-ablate-v1', 'category': 'creative_writing', 'take': 4,
        'sha256': 'abc', 'archived_losses': 2, 'prefix': 'sd-office-8b-traj-ablate',
        'stamp': '20261103-0905',
    }


def _prompts():
    return [
        {'qid': 10, 'text': 'aa'},
        {'qid': 11, 'text': 'bbbb'},
        {'qid': 12, 'text': 'c'},
        {'qid': 13, 'text': 'dddddd'},
    ]


def _rows(accepted, ar_equal):
    return [
        {'qid': prompt['qid'], 'accepted': count, 'ar_equal': flag}
        for prompt, count, flag in zip(_prompts(), accepted, ar_equal)
    ]


class _Tokenizer:
    def __call__(self, text):
        return {'input_ids': list(range(len(text.split())))}


def test_subset_is_only_full_wins_and_reports_the_four_descriptions():
    report = describe_paired(
        _group(), _prompts(),
        _rows([5, 1, 2, 4], [False, True, True, False]),
        _rows([1, 3, 2, 3], [True, False, True, True]),
        'abc', _Tokenizer())
    assert report['status'] == 'described'
    assert report['subset'] == [{'qid': 10, 'delta': 4}, {'qid': 13, 'delta': 1}]
    assert report['counts'] == {'subset': 2, 'traj_higher': 1, 'tie': 1}
    assert report['lengths']['subset']['chars'] == {'count': 2, 'mean': 4.0, 'median': 4.0}
    assert report['lengths']['traj_higher']['chars'] == {'count': 1, 'mean': 4.0, 'median': 4}
    assert report['lengths']['tie']['chars'] == {'count': 1, 'mean': 1.0, 'median': 1}
    assert report['lengths']['subset']['tokens'] == {'count': 2, 'mean': 1.0, 'median': 1.0}
    assert report['ar_equal_false'] == {
        'subset': {'full': 2, 'traj': 0, 'either': 2},
        'rest': {'full': 0, 'traj': 1, 'either': 1},
    }
    assert report['delta'] == {
        'count': 2, 'min': 1, 'median': 2.5, 'max': 4, 'mean': 2.5, 'sum': 5}


def test_equal_and_traj_higher_stay_out_of_the_subset():
    group = _group()
    group['archived_losses'] = 1
    report = describe_paired(
        group, _prompts(),
        _rows([1, 1, 2, 4], [True, True, True, True]),
        _rows([1, 3, 2, 3], [True, True, True, True]),
        'abc')
    assert [item['qid'] for item in report['subset']] == [13]
    assert report['lengths']['subset']['tokens'] is None


def test_count_mismatch_stops_without_a_finished_subset():
    group = _group()
    group['archived_losses'] = 5
    report = describe_paired(
        group, _prompts(),
        _rows([5, 1, 2, 4], [True, True, True, True]),
        _rows([1, 3, 2, 3], [True, True, True, True]),
        'abc')
    assert report['status'] == 'stopped'
    assert report['observed_subset_count'] == 2
    assert 'subset' not in report


def test_qid_mismatch_stops_the_group():
    rows = _rows([5, 1, 2, 9], [True, True, True, True])
    rows[0]['qid'] = 99
    report = describe_paired(_group(), _prompts(), rows, _rows([1, 3, 2, 3], [True, True, True, True]), 'abc')
    assert report['status'] == 'stopped'
    assert report['reason'] == 'archived qids do not match the frozen slice'


def test_missing_jsonl_is_unknown(tmp_path):
    report = report_group(_group(), tmp_path, _prompts(), 'abc')
    assert report['status'] == 'unknown'
    assert report['reason'] == 'archived jsonl missing'
    assert 'subset' not in report


def test_present_jsonl_uses_the_archived_names(tmp_path):
    prompts = _prompts()
    for arm, accepted in (('full', [1, 1, 2, 4]), ('traj', [1, 3, 2, 3])):
        path = tmp_path / f'sd-office-8b-traj-ablate-{arm}-20261103-0905.jsonl'
        path.write_text(''.join(
            json.dumps({'qid': prompt['qid'], 'accepted': count, 'ar_equal': True}) + '\n'
            for prompt, count in zip(prompts, accepted)), encoding='utf-8')
    group = _group()
    group['archived_losses'] = 1
    report = report_group(group, tmp_path, prompts, 'abc')
    assert report['status'] == 'described'
    assert report['subset'] == [{'qid': 13, 'delta': 1}]


def test_tokenizer_loader_stays_local(tmp_path):
    seen = {}

    def loader(path, local_files_only):
        seen['path'] = path
        seen['local_files_only'] = local_files_only
        return 'tok'

    assert load_local_tokenizer(tmp_path, loader) == 'tok'
    assert seen == {'path': str(tmp_path), 'local_files_only': True}
    assert load_local_tokenizer(tmp_path / 'missing', loader) is None

    def broken(path, local_files_only):
        raise OSError('offline')

    assert load_local_tokenizer(tmp_path, broken) is None


def test_pair_relation_uses_the_chapter_thresholds():
    assert pair_relation(128, 0, 0, 0, 0) == '全一致'
    assert pair_relation(128, 40, 10, 30, 10) == '分离'
    assert pair_relation(128, 20, 18, 8, 15) == '重叠'
    assert pair_relation(128, 20, 10, 10, 10) is None
    assert pair_relation(60, 10, 2, 8, 2) == '分离'


def test_jaccard_bands_and_the_empty_sets():
    assert jaccard([], []) == 1.0
    assert jaccard_band(1.0) == '稳定'
    assert jaccard_band(jaccard([1, 2], [3])) == '不稳定'
    assert jaccard_band(0.6) is None


def test_acceptance_label_ignores_short_groups_and_the_middle_gap():
    assert acceptance_label(1.0, 1.2, 7, 20) is None
    assert acceptance_label(1.0, 1.2, 8, 8) == '不随分组变动'
    assert acceptance_label(1.0, -0.2, 8, 8) == '随分组变动'
    assert acceptance_label(2.0, 0.5, 8, 8) == '随分组变动'
    assert acceptance_label(1.0, 0.4, 8, 8) is None
    assert acceptance_label(None, 1.0, 8, 0) is None


def test_conclusion_order_stops_at_the_first_match():
    quiet = {'status': 'described', 'relation': '重叠', 'acceptance_label': '不随分组变动'}
    unstable = {'status': 'compared', 'full': {'band': '不稳定'}, 'traj': {'band': '不稳定'}}
    assert choose_conclusion([quiet] * 8, unstable) == '保持诊断'
    split = {'status': 'described', 'relation': '分离', 'acceptance_label': None}
    stable = {'status': 'compared', 'full': {'band': '稳定'}, 'traj': {'band': '不稳定'}}
    assert choose_conclusion([split] * 4 + [quiet] * 4, stable) == '正确性风险需处理'
    middle = {'status': 'compared', 'full': {'band': None}, 'traj': {'band': None}}
    assert choose_conclusion([quiet] * 8, middle) == '可隔离的口径问题'
    assert choose_conclusion([quiet] * 5 + [{'status': 'unknown'}] * 3, unstable) == '未知'
    assert choose_conclusion([quiet] * 7 + [{'status': 'invalid'}], unstable) == '无效'
    assert choose_conclusion([quiet] * 8, {'status': 'unknown'}) == '未知'


def test_classify_pair_keeps_divergence_missing_and_stops_on_argmax():
    assert MISSING_FIELDS == (
        'first_divergence_index', 'common_prefix_length', 'side_tokens')
    wall = wall_group()
    assert wall['stamp'] == '20261104-0758'
    assert wall['sha256'] == EVAL_SHA256_XCAT7
    group = _group()
    group['key'] = 'ablate'
    prompts = _prompts()
    full = _rows([5, 1, 2, 4], [False, True, True, False])
    traj = _rows([1, 3, 2, 3], [True, False, True, True])
    for row in full + traj:
        row['argmax_passed'] = True
        row['tau'] = 1.5
        row['output_sha256'] = 'deadbeef'
    report = classify_pair(group, prompts, full, traj, 'abc')
    assert report['status'] == 'described'
    assert report['missing_fields'] == list(MISSING_FIELDS)
    assert 'first_divergence_index' not in report
    assert report['relation'] == '分离'
    assert report['inconsistent']['full']['count'] == 2
    full[0]['argmax_passed'] = False
    stopped = classify_pair(group, prompts, full, traj, 'abc')
    assert stopped['status'] == 'invalid'
    assert 'relation' not in stopped
    full[0]['qid'] = 99
    full[0]['argmax_passed'] = True
    unknown = classify_pair(group, prompts, full, traj, 'abc')
    assert unknown['status'] == 'unknown'


def _fork_row(qid, equal, index, candidate, prompt_length=4, digest='arm'):
    row = {
        'qid': qid,
        'ar_equal': equal,
        'argmax_passed': True,
        'first_divergence_index': None if equal else index,
        'common_prefix_length': prompt_length + 8 if equal else index,
        'prompt_length': prompt_length,
        'output_sha256': digest if equal else f'{digest}-{qid}-{index}-{candidate}',
    }
    if not equal:
        row['side_tokens'] = {'reference': 3, 'candidate': candidate}
    return row


def test_first_fork_labels_follow_the_registered_rates():
    from studies.trajectory_observation.structure.office_traj_fork import describe_first_fork

    shared_full = [_fork_row(qid, False, 10, 9) for qid in range(5)]
    shared_traj = [_fork_row(qid, False, 10, 9, digest='same') for qid in range(5)]
    for left, right in zip(shared_full, shared_traj):
        right['output_sha256'] = left['output_sha256']
    shared = describe_first_fork(shared_full, shared_traj)
    assert shared['label'] == '共享参照'
    assert shared['h1_direction'] == 'positive'
    assert shared['shared_fork_rate'] == 1.0
    assert shared['arms_hash_equal_rate'] == 1.0
    assert shared['argmax_all_passed'] is True
    assert shared['continuation_offset_median'] == {'full': 6, 'traj': 6}
    assert 'fp32' not in shared

    split_traj = [_fork_row(qid, False, 12, 8, digest='other') for qid in range(5)]
    split = describe_first_fork(shared_full, split_traj)
    assert split['label'] == '分叉分开'
    assert split['h1_direction'] == 'negative'
    assert split['shared_fork_rate'] == 0.0

    inside = [_fork_row(qid, False, 1, 9, prompt_length=4) for qid in range(5)]
    blocked = describe_first_fork(inside, [dict(row) for row in inside])
    assert blocked['fork_inside_prompt'] == 10
    assert blocked['label'] is None
    assert blocked['shared_fork_rate'] == 1.0

    quiet = [_fork_row(qid, True, None, None) for qid in range(4)]
    assert describe_first_fork(quiet, [dict(row) for row in quiet])['relation'] == '全一致'

    failed = [dict(row, argmax_passed=False) for row in shared_full]
    assert describe_first_fork(failed, shared_traj)['status'] == 'invalid'
    assert 'shared_fork_rate' not in describe_first_fork(failed, shared_traj)


def test_fp32_gap_is_separate_from_the_bf16_fork_label():
    from studies.trajectory_observation.structure.office_traj_fork import describe_first_fork, fp32_fields

    full = []
    traj = []
    for qid in range(5):
        left = _fork_row(qid, False, 10, 9)
        right = dict(left)
        fields = fp32_fields([1, 2, 9], [1, 2, 9], [1, 2, 9])
        left.update(fields)
        right.update(dict(fields))
        full.append(left)
        traj.append(right)
    report = describe_first_fork(full, traj)
    assert report['label'] == '共享参照'
    assert report['fp32']['label'] == 'bf16参照离开fp32'
    assert report['fp32']['h3_direction'] == 'positive'
    assert report['fp32']['full_gap'] == 1.0
    assert report['argmax_all_passed'] is True


def test_fp32_gate_does_not_load_when_the_resident_copy_does_not_fit(tmp_path):
    from studies.trajectory_observation.structure.office_traj_fork import (
        FORK_PREFIX, FORK_STAMP, finish_first_fork, fp32_allocation_fits,
    )

    assert fp32_allocation_fits(8_000_000_000, 21000)['fits'] is False
    assert fp32_allocation_fits(1000, 21000)['fits'] is True
    calls = []
    full = _fork_row(1, False, 6, 4)
    full['output_ids'] = [1, 2, 9]
    full['reference_ids'] = [1, 2, 3]
    text = json.dumps(full, ensure_ascii=False) + '\n'
    (tmp_path / f'{FORK_PREFIX}-full-{FORK_STAMP}.jsonl').write_text(text, encoding='utf-8')
    (tmp_path / f'{FORK_PREFIX}-traj-{FORK_STAMP}.jsonl').write_text(text, encoding='utf-8')
    payload = finish_first_fork(tmp_path, 8_000_000_000, 21000, loader=lambda: calls.append('load'))
    assert calls == []
    assert payload['status'] == 'not_loaded'
    assert payload['reason'] == 'fp32 resident weights exceed free GPU memory'
    assert payload['fork']['label'] == '共享参照'
    saved = json.loads((tmp_path / f'{FORK_PREFIX}-fork-{FORK_STAMP}.json').read_text(encoding='utf-8'))
    assert saved['status'] == 'not_loaded'
    kept = (tmp_path / f'{FORK_PREFIX}-full-{FORK_STAMP}.jsonl').read_text(encoding='utf-8')
    assert 'fp32_ar_equal' not in kept


def _gap(arm, gap, index=10, matches=True, token=4):
    return {
        'arm': arm,
        'gap': gap,
        'first_divergence_index': index,
        'top1_matches': matches,
        'top1': token if matches else token + 1,
        'top2': 1,
        'reference_token': token,
    }


def _det_row(qid, double=True, path=True, gaps=None, length_ok=True):
    return {
        'qid': qid,
        'double_run_equal': double,
        'path_equal': path,
        'length_ok': length_ok,
        'gaps': [_gap('full', 3.5)] if gaps is None else gaps,
    }


def test_ar_determinism_labels_follow_the_registered_bands():
    from studies.trajectory_observation.structure.office_traj_determinism import describe_ar_determinism

    stable = describe_ar_determinism([_det_row(qid) for qid in range(4)])
    assert stable['label'] == '差距大且路径稳定'
    assert stable['noise_direction'] == 'negative'
    assert stable['double_run_identity_rate'] == 1
    assert stable['gap_median'] == 3.5
    assert 'argmax_all_passed' not in stable

    near = describe_ar_determinism([_det_row(qid, gaps=[_gap('full', 0.5)]) for qid in range(3)])
    assert near['label'] == '分叉贴近并列'
    assert near['noise_direction'] == 'positive'

    flipped = describe_ar_determinism([_det_row(0, double=False), _det_row(1)])
    assert flipped['label'] == 'AR路径不稳定'
    assert flipped['noise_direction'] == 'positive'
    assert flipped['gap_median'] == 3.5

    replay = describe_ar_determinism([_det_row(0, gaps=[_gap('full', 4.0, matches=False)])])
    assert replay['label'] == 'AR路径不稳定'
    assert replay['replay_top1_match_rate'] == 0

    middle = describe_ar_determinism([_det_row(qid, gaps=[_gap('full', 1.2)]) for qid in range(3)])
    assert middle['label'] is None
    assert middle['noise_direction'] is None
    assert middle['gap_median'] == 1.2

    tied_tail = describe_ar_determinism([
        _det_row(0, gaps=[_gap('full', 0.1, index=10), _gap('traj', 0.1, index=11)]),
        _det_row(1, gaps=[_gap('full', 3.0, index=12)]),
        _det_row(2, gaps=[_gap('full', 4.0, index=13)]),
        _det_row(3, gaps=[_gap('full', 5.0, index=14)]),
    ])
    assert tied_tail['gap_median'] == 3.0
    assert tied_tail['near_gap_rate'] == 0.4
    assert tied_tail['label'] is None

    shared = describe_ar_determinism([
        _det_row(0, gaps=[_gap('full', 3.0, index=10), _gap('traj', 3.0, index=10)]),
    ])
    assert shared['arm_gap_count'] == 2
    assert shared['gap_count'] == 1
    assert shared['label'] == '差距大且路径稳定'

    clash = describe_ar_determinism([
        _det_row(0, gaps=[_gap('full', 3.0, index=10), _gap('traj', 4.0, index=10)]),
    ])
    assert clash['status'] == 'unknown'
    assert clash['reason'] == 'repeated fork disagrees'

    short = describe_ar_determinism([_det_row(0, length_ok=False)])
    assert short['status'] == 'invalid'
    assert 'label' not in short
    assert describe_ar_determinism([_det_row(0, gaps=[])])['reason'] == 'divergence gap missing'


def _archive_arm(qid, equal, index=6, prompt_length=4, reference=None):
    reference = list(reference or [1, 2, 3, 4, 5, 6, 9, 8])
    row = {
        'qid': qid,
        'ar_equal': equal,
        'prompt_length': prompt_length,
        'reference_ids': reference,
        'first_divergence_index': None if equal else index,
    }
    if not equal:
        row['side_tokens'] = {'reference': reference[index], 'candidate': 1}
    return row


def test_gap_targets_require_the_archived_union(tmp_path):
    from studies.trajectory_observation.structure.office_traj_determinism import (
        gap_targets_for_qids, load_mismatch_targets,
    )

    full = {
        0: _archive_arm(0, False),
        1: _archive_arm(1, False),
        2: _archive_arm(2, True),
    }
    traj = {
        0: _archive_arm(0, True),
        1: _archive_arm(1, False),
        2: _archive_arm(2, False, index=5),
    }
    with pytest.raises(RuntimeError, match='archived union'):
        gap_targets_for_qids(full, traj, [0, 1])
    targets = gap_targets_for_qids(full, traj, [0, 1, 2])
    assert [(item['qid'], item['arm'], item['first_divergence_index']) for item in targets] == [
        (0, 'full', 6), (1, 'full', 6), (1, 'traj', 6), (2, 'traj', 5),
    ]
    inside = {0: traj[0], 1: traj[1], 2: _archive_arm(2, False, index=1)}
    with pytest.raises(RuntimeError, match='outside the continuation'):
        gap_targets_for_qids(full, inside, [0, 1, 2])

    root = tmp_path
    full_path = root / 'full.jsonl'
    traj_path = root / 'traj.jsonl'
    full_path.write_text(''.join(json.dumps(full[qid]) + '\n' for qid in (0, 1, 2)), encoding='utf-8')
    traj_path.write_text(''.join(json.dumps(traj[qid]) + '\n' for qid in (0, 1, 2)), encoding='utf-8')
    listed = root / 'mismatch.json'
    listed.write_text(json.dumps({
        'source_full': str(full_path),
        'source_traj': str(traj_path),
        'mismatch_qids': [0, 1, 2],
    }), encoding='utf-8')
    loaded = load_mismatch_targets(listed)
    assert [item['qid'] for item in loaded] == [0, 1, 1, 2]


def test_measure_records_path_inequality_and_the_replayed_gap():
    import torch
    from studies.trajectory_observation.structure.office_traj_determinism import measure_determinism_prompt

    calls = []

    def generate(model, ids, limit, prompt_tokenwise=False):
        calls.append(prompt_tokenwise)
        base = [7, 8, 9, 1, 2]
        if prompt_tokenwise:
            base = [7, 8, 9, 1, 0]
        return base, {}

    def logits_at(model, reference, prompt_length, index):
        assert reference[index] == 4
        assert prompt_length == 3
        assert index == 5
        row = torch.zeros(6)
        row[4] = 3
        row[2] = 1
        return row

    reference = [7, 8, 9, 3, 5, 4]
    targets = [
        {
            'qid': 3, 'arm': 'full', 'prompt_length': 3, 'first_divergence_index': 5,
            'reference_token': 4, 'reference_ids': reference,
        },
        {
            'qid': 3, 'arm': 'traj', 'prompt_length': 3, 'first_divergence_index': 5,
            'reference_token': 4, 'reference_ids': reference,
        },
    ]
    row = measure_determinism_prompt(
        None, torch.tensor([[7, 8, 9]]), targets, {'max_new_tokens': 2}, generate, logits_at)
    assert calls == [False, False, True]
    assert row['double_run_equal'] is True
    assert row['path_equal'] is False
    assert row['length_ok'] is True
    assert row['gaps'][0]['gap'] == 2
    assert row['gaps'][0]['top1'] == 4
    assert row['gaps'][0]['top1_matches'] is True
    assert row['gaps'][1]['gap'] == row['gaps'][0]['gap']
    assert row['path_fork']['first_divergence_index'] == 4
    with pytest.raises(RuntimeError, match='tokenizer length'):
        measure_determinism_prompt(
            None, torch.tensor([[7, 8]]), targets, {'max_new_tokens': 2}, generate, logits_at)


def test_tokenwise_prompt_matches_only_when_the_prefill_width_does_not_matter():
    from types import SimpleNamespace

    import torch

    from spec_mtp.structure.algorithm.decode.loop import ar_context_logits, ar_generate

    class Session:
        def __init__(self):
            self.embedding_table = torch.eye(8)
            self.device, self.dtype = torch.device('cpu'), torch.float32
            self.vocab_size, self.max_positions = 8, 128

        def embed(self, ids):
            return self.embedding_table[ids]

        def new_cache(self):
            return SimpleNamespace(length=0)

        def __call__(self, embeds, positions, *, past_key_values, **_):
            tokens = (embeds.argmax(-1) + 1) % 8
            logits = torch.nn.functional.one_hot(tokens, 8).float() * 20
            past_key_values.length += embeds.shape[1]
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    class Sensitive(Session):
        def __call__(self, embeds, positions, *, past_key_values, **kwargs):
            out = super().__call__(embeds, positions, past_key_values=past_key_values, **kwargs)
            if embeds.shape[1] > 1:
                out.logits = torch.zeros_like(out.logits)
                out.logits[:, -1, 0] = 20
            return out

    limit = {'max_new_tokens': 3, 'temperature': 0.0, 'stop_on_eos': False}
    prompt = torch.tensor([[1, 2, 3]])
    model = Session()
    whole, whole_stats = ar_generate(model, prompt, limit)
    again, _stats = ar_generate(model, prompt, limit)
    tokenwise, token_stats = ar_generate(model, prompt, limit, prompt_tokenwise=True)
    assert torch.equal(whole, again)
    assert torch.equal(whole, tokenwise)
    assert whole_stats['ar_calls'] == 3
    assert token_stats['ar_calls'] == 5
    ids = whole[0].tolist()
    for index in range(3, 6):
        assert int(ar_context_logits(model, ids, 3, index).argmax()) == ids[index]
    with pytest.raises(ValueError, match='outside the continuation'):
        ar_context_logits(model, ids, 3, 2)

    sensitive = Sensitive()
    wide, _stats = ar_generate(sensitive, prompt, limit)
    narrow, _stats = ar_generate(sensitive, prompt, limit, prompt_tokenwise=True)
    assert not torch.equal(wide, narrow)
    assert int(wide[0, 3]) == 0
    assert int(narrow[0, 3]) == 4


def test_ar_determinism_command_stops_without_the_mismatch_file(tmp_path):
    from studies.trajectory_observation.structure.office_traj_determinism import run_ar_determinism_main

    with pytest.raises(RuntimeError, match='mismatch'):
        run_ar_determinism_main(tmp_path, None)


def _bucket_diff(mean):
    return {'difference': {'mean': mean}}


def _length_piece(length, direction, mean, draft, verify, overhead, residual, complete=True):
    return {
        'length': length,
        'complete': complete,
        'wall': {'wall_direction': direction, 'wall_difference': {'mean': mean}},
        'buckets': {
            'draft_seconds': _bucket_diff(draft),
            'verify_seconds': _bucket_diff(verify),
            'overhead_seconds': _bucket_diff(overhead),
            'residual_seconds': _bucket_diff(residual),
        },
    }


def test_row_buckets_keep_a_negative_residual_and_a_missing_draft():
    split = row_buckets({
        'wall_seconds': 1.0, 'prefill_time': 0.2, 'draft_seconds': 0.5, 'verify_seconds': 0.5,
    })
    assert split['overhead_seconds'] == 0.2
    assert split['residual_seconds'] == pytest.approx(-0.2)
    assert row_buckets({
        'wall_seconds': 1.0, 'prefill_time': 0.2, 'verify_seconds': 0.5,
    }) is None
    assert row_buckets({
        'wall_seconds': 1.0, 'prefill_time': -1.0, 'draft_seconds': 0.1, 'verify_seconds': 0.1,
    }) is None


def test_length_decision_follows_the_pre_registered_order():
    cfg = round_config('wall_by_length')
    assert cfg['protocol'] == PROTOCOL_LENGTH
    assert cfg['lengths'] == LENGTHS == (64, 256, 512)
    assert cfg['stamp'] == '20261104-1420'
    growing = [
        _length_piece(64, 'shorter', -0.10, -0.02, -0.07, -0.01, 0.0),
        _length_piece(256, 'shorter', -0.40, -0.05, -0.30, -0.02, -0.03),
        _length_piece(512, 'shorter', -1.00, -0.10, -0.80, -0.04, -0.06),
    ]
    decided = length_decision(growing)
    assert decided == {
        'length_label': 'length-dependent',
        'decomp_label': 'decomp-verify',
        'decomp_length': 512,
    }
    flat = [
        _length_piece(64, 'shorter', -0.10, -0.08, -0.01, 0.0, -0.01),
        _length_piece(256, 'shorter', -0.11, -0.09, -0.01, 0.0, -0.01),
        _length_piece(512, 'shorter', -0.12, -0.10, -0.01, 0.0, -0.01),
    ]
    assert length_decision(flat)['length_label'] == 'length-flat'
    assert length_decision(flat)['decomp_label'] == 'decomp-draft'
    zeros = [
        _length_piece(64, 'flat', 0.0, 0.0, 0.0, 0.0, 0.0),
        _length_piece(256, 'flat', 0.0, 0.0, 0.0, 0.0, 0.0),
        _length_piece(512, 'flat', 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    assert length_decision(zeros)['length_label'] == 'length-flat'
    assert length_decision(zeros)['decomp_label'] == 'decomp-split'
    dead = [
        _length_piece(64, 'shorter', -0.2, -0.2, 0.0, 0.0, 0.0),
        _length_piece(256, 'longer', 0.3, 0.3, 0.0, 0.0, 0.0),
        _length_piece(512, 'flat', 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    assert length_decision(dead)['length_label'] == 'dead-metric'
    mixed = [
        _length_piece(64, 'shorter', -0.2, -0.02, 0.0, -0.18, 0.0),
        _length_piece(256, 'longer', 0.4, 0.05, 0.0, 0.35, 0.0),
        _length_piece(512, 'shorter', -0.5, -0.05, 0.0, -0.45, 0.0),
    ]
    mixed_decision = length_decision(mixed)
    assert mixed_decision['length_label'] == 'mixed'
    assert mixed_decision['decomp_label'] == 'decomp-overhead'
    assert mixed_decision['decomp_length'] == 512
    partial = growing[:2]
    assert length_decision(partial) == {
        'length_label': 'incomplete', 'decomp_label': None, 'decomp_length': None,
    }
    failed = [dict(piece, complete=False) for piece in growing]
    assert length_decision(failed)['length_label'] == 'incomplete'


def test_gscan_round_reuses_the_summarization_slice_and_a_distinct_prefix():
    cfg = round_config('gscan')
    assert cfg['protocol'] == PROTOCOL_GSCAN == 'office-8b-traj-g-scan-v1'
    assert cfg['prefix'] == GSCAN_PREFIX == 'sd-office-8b-traj-g-scan-v1'
    assert cfg['stamp'] == GSCAN_STAMP == '20261104-1813'
    assert cfg['take'] == 128
    assert cfg['select'].__name__ == 'prompts_summarization'
    assert cfg['check'].__name__ == 'assert_registered_xcat7_slice'
    assert 'xcat7' not in cfg['prefix']
    assert cfg['prefix'] != LENGTH_PREFIX
    assert cfg['prefix'] != 'sd-office-8b-traj-wall-v1'


def test_four_specs_share_the_esp_mask_and_only_the_gain_changes():
    specs = dict(gscan_specs())
    assert list(specs) == ['g0', 'g025', 'g05', 'g075']
    assert [spec['observation']['gain'] for spec in specs.values()] == [0.0, 0.25, 0.5, 0.75]
    bare = {key: value for key, value in specs['g0'].items() if key != 'observation'}
    for spec in specs.values():
        assert {key: value for key, value in spec.items() if key != 'observation'} == bare
        assert spec['observation']['condition'] == 'none'
        assert spec['observation']['form'] == 'mean'
        assert spec['mask'] == MASK
        assert spec['tree']['branches'] == [4]
        assert spec['tree']['pruning'] is True
        assert spec['method'] == 'esp'
    assert specs['g0']['tree']['branches'] is not specs['g025']['tree']['branches']
    assert TREE['branches'] == [4]
    limit = gscan_limit()
    assert limit['max_new_tokens'] == 256
    assert limit['temperature'] == 0.0
    assert limit['stop_on_eos'] is False
    with pytest.raises(RuntimeError, match='pre-registered'):
        gscan_arm_spec('g1')


def _gscan_rows(accepted, tau=1.5, wall=None):
    rows = {}
    for name, value in accepted.items():
        arm = []
        for qid in range(128):
            row = {
                'qid': qid,
                'accepted': value,
                'tau': tau,
                'argmax_passed': True,
                'ar_equal': True,
                'max_nodes': 4,
                'min_nodes': 4,
                'block_complexity': 10,
                'max_new_tokens': 256,
            }
            if wall is not None:
                row['wall_seconds'] = wall
            arm.append(row)
        rows[name] = arm
    return rows


def _gscan_accepted():
    return _gscan_rows({'g0': 3, 'g025': 5, 'g05': 3, 'g075': 1})


def test_summary_pairs_each_gain_with_g0_and_skips_the_sign_test_when_invalid():
    summary = gscan_summary(_gscan_accepted(), EVAL_SHA256_XCAT7, ())
    assert summary['valid'] is True
    assert summary['reason'] is None
    assert summary['blend'] == 'trajectory-observation-blend-v1'
    assert list(summary['comparisons']) == ['g025', 'g05', 'g075']
    higher = summary['comparisons']['g025']
    assert higher['wins_losses_ties'] == [128, 0, 0]
    assert higher['accepted_difference']['mean'] == 2.0
    assert higher['accepted_difference']['seed'] == 0
    assert higher['accepted_difference']['resamples'] == 10000
    assert higher['accepted_difference']['excludes_zero'] is True
    assert higher['tau_difference']['resamples'] == 10000
    assert higher['exploratory'] is True
    assert higher['confirmatory'] is False
    assert 'p_below_0.05' not in higher
    assert summary['comparisons']['g05']['wins_losses_ties'] == [0, 0, 128]
    assert summary['comparisons']['g05']['p'] is None
    assert summary['comparisons']['g05']['direction'] == 'flat'
    assert summary['comparisons']['g075']['wins_losses_ties'] == [0, 128, 0]
    assert summary['tau_mean']['g0'] == 1.5

    broken = _gscan_accepted()
    broken['g05'][4]['argmax_passed'] = False
    failed = gscan_summary(broken, EVAL_SHA256_XCAT7, ())
    assert failed['valid'] is False
    assert failed['reason'] == 'argmax'
    assert 'comparisons' not in failed

    nodes = _gscan_accepted()
    nodes['g0'][0]['max_nodes'] = 3
    assert gscan_summary(nodes, EVAL_SHA256_XCAT7, ())['reason'] == 'draft nodes'
    assert 'comparisons' not in gscan_summary(nodes, EVAL_SHA256_XCAT7, ())

    short = _gscan_accepted()
    short['g075'][2]['max_new_tokens'] = 64
    assert gscan_summary(short, EVAL_SHA256_XCAT7, ())['reason'] == 'length'


def test_missing_wall_stays_missing_and_does_not_drop_the_acceptance_pair():
    quiet = gscan_summary(_gscan_accepted(), EVAL_SHA256_XCAT7, ())
    assert quiet['comparisons']['g025']['wall'] == {'present': False}
    timed = _gscan_rows({'g0': 3, 'g025': 5, 'g05': 3, 'g075': 1}, wall=1.0)
    for row in timed['g0']:
        row['wall_seconds'] = 0.25
    reported = gscan_summary(timed, EVAL_SHA256_XCAT7, ())
    assert reported['valid'] is True
    assert reported['comparisons']['g025']['wall']['present'] is True
    assert reported['comparisons']['g025']['wall']['mean'] == pytest.approx(0.75)
    timed['g075'][9]['wall_seconds'] = -1.0
    negative = gscan_summary(timed, EVAL_SHA256_XCAT7, ())
    assert negative['valid'] is True
    assert negative['comparisons']['g075']['wall'] == {'present': False}
    assert '"wall_seconds": 0' not in json.dumps(negative)


def _gscan_prompts():
    return [
        {'qid': 0, 'category': 'summarization', 'text': 'one'},
        {'qid': 1, 'category': 'summarization', 'text': 'two'},
        {'qid': 2, 'category': 'summarization', 'text': 'three'},
    ]


def _gscan_decoded(prompt, argmax=True, nodes=4, wall=None):
    row = {
        'accepted': 3, 'argmax_passed': argmax, 'max_nodes': nodes, 'block_complexity': 10,
        'tau': 1.0, 'ar_equal': True, 'min_nodes': nodes,
    }
    if wall is not None:
        row['wall_seconds'] = wall
    return {name: dict(row) for name, _spec in gscan_specs()}


def test_smoke_runs_two_prompts_and_four_gains_without_a_wall_gate(tmp_path, monkeypatch):
    from studies.trajectory_observation.structure import office_traj as traj

    seen = []

    def decode(model, tokenizer, prompt, specs, limit):
        seen.append((prompt['qid'], limit['max_new_tokens'], [name for name, _spec in specs]))
        return _gscan_decoded(prompt, wall=None)

    monkeypatch.setattr(traj, 'decode_measured_prompt', decode)
    cfg = round_config('gscan')
    payload = gscan_smoke(None, None, _gscan_prompts(), tmp_path, cfg)
    assert seen == [(0, 8, ['g0', 'g025', 'g05', 'g075']), (1, 8, ['g0', 'g025', 'g05', 'g075'])]
    assert payload['argmax_all_passed'] is True
    assert payload['tokens'] == 8
    assert 'wall_seconds' not in payload['arms'][0]['g0']

    def failed(model, tokenizer, prompt, specs, limit):
        return _gscan_decoded(prompt, argmax=False)

    monkeypatch.setattr(traj, 'decode_measured_prompt', failed)
    with pytest.raises(RuntimeError, match='argmax'):
        gscan_smoke(None, None, _gscan_prompts(), tmp_path, cfg)


def test_formal_warms_four_gains_then_measures_256_tokens(tmp_path, monkeypatch):
    from studies.trajectory_observation.structure import office_traj as traj

    events = []

    def arm_row(model, spec, ids, limit):
        events.append(('warmup', spec['observation']['gain'], limit['max_new_tokens']))
        return {'argmax_passed': True}

    def decode(model, tokenizer, prompt, specs, limit, keep_sequences=False):
        events.append(('decode', prompt['qid'], limit['max_new_tokens'], [name for name, _spec in specs]))
        row = _gscan_decoded(prompt)['g0']
        row['qid'] = prompt['qid']
        return {name: dict(row) for name, _spec in specs}

    monkeypatch.setattr(
        'studies.trajectory_observation.structure.office_tree.arm_row', arm_row)
    monkeypatch.setattr('spec_mtp.model.sync_device', lambda device: events.append(('sync', device)))
    monkeypatch.setattr(traj, 'decode_measured_prompt', decode)
    monkeypatch.setattr(traj, '_run_identity', lambda: {'git_head': 'abc', 'dirty': False})
    cfg = round_config('gscan')

    class _Tokenizer:
        def __call__(self, text, return_tensors='pt'):
            return type('Batch', (), {'input_ids': [[1, 2]]})()

    summary = gscan_formal(
        type('Model', (), {'device': 'cpu'})(), _Tokenizer(), _gscan_prompts()[:1], (),
        EVAL_SHA256_XCAT7, tmp_path, cfg,
    )
    assert events[:5] == [
        ('warmup', 0.0, 8), ('warmup', 0.25, 8), ('warmup', 0.5, 8), ('warmup', 0.75, 8),
        ('sync', 'cpu'),
    ]
    assert events[5] == ('decode', 0, 256, ['g0', 'g025', 'g05', 'g075'])
    header = json.loads((tmp_path / f'{GSCAN_PREFIX}-header-{GSCAN_STAMP}.json').read_text(encoding='utf-8'))
    assert header['eval']['max_new_tokens'] == 256
    assert header['arms']['g025']['observation']['gain'] == 0.25
    assert header['sign_test'].startswith('exploratory')
    line = (tmp_path / f'{GSCAN_PREFIX}-g075-{GSCAN_STAMP}.jsonl').read_text(encoding='utf-8')
    assert '"max_new_tokens": 256' in line
    assert summary['valid'] is False
    assert summary['reason'] == 'row count'
    assert 'comparisons' not in summary
