"""Full-win subset description. No model and no archived jsonl."""

import json

from studies.trajectory_observation.structure.office_traj import (
    EVAL_SHA256, EVAL_SHA256_XCAT, EVAL_SHA256_XCAT2, EVAL_SHA256_XCAT3, EVAL_SHA256_XCAT4,
    EVAL_SHA256_XCAT5, EVAL_SHA256_XCAT6, EVAL_SHA256_XCAT7,
)
from studies.trajectory_observation.structure.office_traj_failset import (
    GROUPS, MISSING_FIELDS, acceptance_label, choose_conclusion, classify_pair, describe_paired,
    jaccard, jaccard_band, load_local_tokenizer, pair_relation, report_group, wall_group,
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
