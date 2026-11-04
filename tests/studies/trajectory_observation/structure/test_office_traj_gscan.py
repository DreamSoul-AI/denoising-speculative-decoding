"""CPU checks for the frozen four-gain scan. No model load."""

import json

import pytest

from studies.trajectory_observation.structure.office_traj import (
    EVAL_SHA256_XCAT7, MASK, TREE, round_config,
)
from studies.trajectory_observation.structure.office_traj_gscan import (
    GSCAN_PREFIX, GSCAN_STAMP, PROTOCOL_GSCAN, gscan_arm_spec, gscan_formal,
    gscan_limit, gscan_smoke, gscan_specs, gscan_summary,
)
from studies.trajectory_observation.structure.office_traj_length import LENGTH_PREFIX


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


def _rows(accepted, tau=1.5, wall=None):
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


def _accepted():
    return _rows({'g0': 3, 'g025': 5, 'g05': 3, 'g075': 1})


def test_summary_pairs_each_gain_with_g0_and_skips_the_sign_test_when_invalid():
    summary = gscan_summary(_accepted(), EVAL_SHA256_XCAT7, ())
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

    broken = _accepted()
    broken['g05'][4]['argmax_passed'] = False
    failed = gscan_summary(broken, EVAL_SHA256_XCAT7, ())
    assert failed['valid'] is False
    assert failed['reason'] == 'argmax'
    assert 'comparisons' not in failed

    nodes = _accepted()
    nodes['g0'][0]['max_nodes'] = 3
    assert gscan_summary(nodes, EVAL_SHA256_XCAT7, ())['reason'] == 'draft nodes'
    assert 'comparisons' not in gscan_summary(nodes, EVAL_SHA256_XCAT7, ())

    short = _accepted()
    short['g075'][2]['max_new_tokens'] = 64
    assert gscan_summary(short, EVAL_SHA256_XCAT7, ())['reason'] == 'length'


def test_missing_wall_stays_missing_and_does_not_drop_the_acceptance_pair():
    quiet = gscan_summary(_accepted(), EVAL_SHA256_XCAT7, ())
    assert quiet['comparisons']['g025']['wall'] == {'present': False}
    timed = _rows({'g0': 3, 'g025': 5, 'g05': 3, 'g075': 1}, wall=1.0)
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


def _prompts():
    return [
        {'qid': 0, 'category': 'summarization', 'text': 'one'},
        {'qid': 1, 'category': 'summarization', 'text': 'two'},
        {'qid': 2, 'category': 'summarization', 'text': 'three'},
    ]


def _decoded(prompt, argmax=True, nodes=4, wall=None):
    row = {
        'accepted': 3, 'argmax_passed': argmax, 'max_nodes': nodes, 'block_complexity': 10,
        'tau': 1.0, 'ar_equal': True, 'min_nodes': nodes,
    }
    if wall is not None:
        row['wall_seconds'] = wall
    return {name: dict(row) for name, _spec in gscan_specs()}


def test_smoke_runs_two_prompts_and_four_gains_without_a_wall_gate(tmp_path, monkeypatch):
    from studies.trajectory_observation.structure import office_traj_gscan as gscan

    seen = []

    def decode(model, tokenizer, prompt, specs, limit):
        seen.append((prompt['qid'], limit['max_new_tokens'], [name for name, _spec in specs]))
        return _decoded(prompt, wall=None)

    monkeypatch.setattr(gscan, 'decode_measured_prompt', decode)
    cfg = round_config('gscan')
    payload = gscan_smoke(None, None, _prompts(), tmp_path, cfg)
    assert seen == [(0, 8, ['g0', 'g025', 'g05', 'g075']), (1, 8, ['g0', 'g025', 'g05', 'g075'])]
    assert payload['argmax_all_passed'] is True
    assert payload['tokens'] == 8
    assert 'wall_seconds' not in payload['arms'][0]['g0']

    def failed(model, tokenizer, prompt, specs, limit):
        return _decoded(prompt, argmax=False)

    monkeypatch.setattr(gscan, 'decode_measured_prompt', failed)
    with pytest.raises(RuntimeError, match='argmax'):
        gscan_smoke(None, None, _prompts(), tmp_path, cfg)


def test_formal_warms_four_gains_then_measures_256_tokens(tmp_path, monkeypatch):
    from studies.trajectory_observation.structure import office_traj_gscan as gscan

    events = []

    def arm_row(model, spec, ids, limit):
        events.append(('warmup', spec['observation']['gain'], limit['max_new_tokens']))
        return {'argmax_passed': True}

    def decode(model, tokenizer, prompt, specs, limit, keep_sequences=False):
        events.append(('decode', prompt['qid'], limit['max_new_tokens'], [name for name, _spec in specs]))
        row = _decoded(prompt)['g0']
        row['qid'] = prompt['qid']
        return {name: dict(row) for name, _spec in specs}

    monkeypatch.setattr(
        'studies.trajectory_observation.structure.office_tree.arm_row', arm_row)
    monkeypatch.setattr('spec_mtp.model.sync_device', lambda device: events.append(('sync', device)))
    monkeypatch.setattr(gscan, 'decode_measured_prompt', decode)
    monkeypatch.setattr(gscan, '_run_identity', lambda: {'git_head': 'abc', 'dirty': False})
    cfg = round_config('gscan')

    class _Tokenizer:
        def __call__(self, text, return_tensors='pt'):
            return type('Batch', (), {'input_ids': [[1, 2]]})()

    summary = gscan_formal(
        type('Model', (), {'device': 'cpu'})(), _Tokenizer(), _prompts()[:1], (),
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
