"""Slice and observation switch for the Qwen3-8B trajectory ablation."""

import pytest
import torch

from spec_mtp.masks import build_mask_provider
from studies.trajectory_observation.structure.office_qwen8b import ESP
from studies.trajectory_observation.structure.office_traj import (
    CATEGORY_XCAT, CATEGORY_XCAT2, CATEGORY_XCAT3, CATEGORY_XCAT4, CATEGORY_XCAT5, CATEGORY_XCAT6,
    CATEGORY_XCAT7, EVAL_QIDS, EVAL_QIDS_XCAT, EVAL_QIDS_XCAT2, EVAL_QIDS_XCAT3, EVAL_QIDS_XCAT4,
    EVAL_QIDS_XCAT5, EVAL_QIDS_XCAT6, EVAL_QIDS_XCAT7, EVAL_SHA256, EVAL_SHA256_XCAT, EVAL_SHA256_XCAT2,
    EVAL_SHA256_XCAT3, EVAL_SHA256_XCAT4, EVAL_SHA256_XCAT5, EVAL_SHA256_XCAT6, EVAL_SHA256_XCAT7,
    GAIN_FULL, GAIN_TRAJ, PROTOCOL, PROTOCOL_XCAT, PROTOCOL_XCAT2, PROTOCOL_XCAT3, PROTOCOL_XCAT4,
    PROTOCOL_XCAT5, PROTOCOL_XCAT6, PROTOCOL_XCAT7, SKIPPED_QIDS, SKIPPED_QIDS_XCAT, SKIPPED_QIDS_XCAT2,
    SKIPPED_QIDS_XCAT3, SKIPPED_QIDS_XCAT4, SKIPPED_QIDS_XCAT5, SKIPPED_QIDS_XCAT6, SKIPPED_QIDS_XCAT7,
    TAKE, TAKE_XCAT, TAKE_XCAT2, TAKE_XCAT3, TAKE_XCAT4, TAKE_XCAT5, TAKE_XCAT6, TAKE_XCAT7,
    ablation_summary, arm_spec, arm_specs, prompts_brainstorming, prompts_classification, prompts_closed_qa,
    prompts_from_647, prompts_general_qa, prompts_information_extraction, prompts_open_qa, prompts_summarization,
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
