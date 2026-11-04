"""Full-win subset description. No model and no archived jsonl."""

import json

from studies.trajectory_observation.structure.office_traj import (
    EVAL_SHA256, EVAL_SHA256_XCAT, EVAL_SHA256_XCAT2, EVAL_SHA256_XCAT3, EVAL_SHA256_XCAT4,
    EVAL_SHA256_XCAT5, EVAL_SHA256_XCAT6, EVAL_SHA256_XCAT7,
)
from studies.trajectory_observation.structure.office_traj_failset import (
    GROUPS, describe_paired, load_local_tokenizer, report_group,
)


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
