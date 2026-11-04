"""The trajectory ablation command refuses a missing output directory."""

import json
from pathlib import Path

import pytest

from studies.trajectory_observation.structure.office_traj import (
    EVAL, PROTOCOL_WALL, _formal, _smoke, main, round_config,
)


def test_command_rejects_a_missing_destination():
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(RuntimeError, match='already exist'):
        main(['--dest', str(Path('missing-office-traj-output')), '--only', 'smoke'])


def test_xcat_round_keeps_the_cross_category_names():
    cfg = round_config('xcat')
    assert cfg['protocol'] == 'office-8b-traj-xcat-v1'
    assert cfg['prefix'] == 'sd-office-8b-traj-xcat'
    assert cfg['stamp'] == '20261103-0929'
    assert cfg['take'] == 128
    assert round_config('ablate')['protocol'] == 'office-8b-traj-ablate-v1'
    assert round_config('ablate')['stamp'] == '20261103-0905'


def test_xcat2_round_keeps_the_second_cross_category_names():
    cfg = round_config('xcat2')
    assert cfg['protocol'] == 'office-8b-traj-xcat2-v1'
    assert cfg['prefix'] == 'sd-office-8b-traj-xcat2'
    assert cfg['stamp'] == '20261103-0958'
    assert cfg['take'] == 128


def test_xcat3_round_keeps_the_third_cross_category_names():
    cfg = round_config('xcat3')
    assert cfg['protocol'] == 'office-8b-traj-xcat3-v1'
    assert cfg['prefix'] == 'sd-office-8b-traj-xcat3'
    assert cfg['stamp'] == '20261103-1024'
    assert cfg['take'] == 128


def test_xcat4_round_keeps_the_fourth_cross_category_names():
    cfg = round_config('xcat4')
    assert cfg['protocol'] == 'office-8b-traj-xcat4-v1'
    assert cfg['prefix'] == 'sd-office-8b-traj-xcat4'
    assert cfg['stamp'] == '20261104-032457'
    assert cfg['take'] == 128


def test_xcat5_round_keeps_the_fifth_cross_category_names():
    cfg = round_config('xcat5')
    assert cfg['protocol'] == 'office-8b-traj-xcat5-v1'
    assert cfg['prefix'] == 'sd-office-8b-traj-xcat5'
    assert cfg['stamp'] == '20261104-035230'
    assert cfg['take'] == 128


def test_xcat6_round_keeps_the_sixth_cross_category_names():
    cfg = round_config('xcat6')
    assert cfg['protocol'] == 'office-8b-traj-xcat6-v1'
    assert cfg['prefix'] == 'sd-office-8b-traj-xcat6'
    assert cfg['stamp'] == '20261104-042546'
    assert cfg['take'] == 128


def test_xcat7_round_keeps_the_seventh_cross_category_names():
    cfg = round_config('xcat7')
    assert cfg['protocol'] == 'office-8b-traj-xcat7-v1'
    assert cfg['prefix'] == 'sd-office-8b-traj-wall-v1'
    assert cfg['stamp'] == '20261104-0758'
    assert cfg['take'] == 128


def test_first_fork_round_keeps_the_registered_stamp():
    cfg = round_config('first_fork')
    assert cfg['protocol'] == 'office-ar-first-fork-v1'
    assert cfg['prefix'] == 'sd-office-ar-first-fork-v1'
    assert cfg['stamp'] == '20261104-1244'
    assert cfg['take'] == 128


def _prompts():
    return [
        {'qid': 0, 'category': 'summarization', 'text': 'one'},
        {'qid': 1, 'category': 'summarization', 'text': 'two'},
    ]


def _decoded(prompt, wall=0.2):
    row = {
        'qid': prompt['qid'], 'category': prompt['category'], 'accepted': 3, 'committed': 8,
        'decode_calls': 4, 'tau': 2.0, 'argmax_passed': True, 'ar_equal': True,
        'max_nodes': 4, 'block_complexity': 10, 'output_sha256': 'abc',
    }
    if wall is not None:
        row['wall_seconds'] = wall
        row['prefill_time'] = 0.01
        row['decode_time'] = 0.02
    return {'full': dict(row, accepted=2), 'traj': dict(row, accepted=5)}


def test_smoke_requires_a_finite_wall_on_both_arms(tmp_path, monkeypatch):
    from studies.trajectory_observation.structure import office_traj

    def missing_wall(model, tokenizer, prompt, specs, limit):
        decoded = _decoded(prompt, wall=None)
        return decoded

    monkeypatch.setattr(office_traj, 'decode_measured_prompt', missing_wall)
    cfg = round_config('xcat7')
    with pytest.raises(RuntimeError, match='wall clock'):
        _smoke(None, None, _prompts(), tmp_path, cfg)

    def recorded(model, tokenizer, prompt, specs, limit):
        assert limit['max_new_tokens'] == 8
        assert [name for name, _spec in specs] == ['full', 'traj']
        return _decoded(prompt, wall=0.2)

    monkeypatch.setattr(office_traj, 'decode_measured_prompt', recorded)
    payload = _smoke(None, None, _prompts(), tmp_path, cfg)
    assert payload['wall_protocol'] == PROTOCOL_WALL
    assert payload['wall_all_recorded'] is True
    assert payload['arms'][0]['full']['wall_seconds'] == 0.2
    assert payload['arms'][0]['traj']['accepted'] == 5


def test_formal_warms_each_arm_before_the_measured_rows(tmp_path, monkeypatch):
    from studies.trajectory_observation.structure import office_traj

    events = []

    def warmup(model, tokenizer, prompt):
        events.append(('warmup', prompt['qid']))

    def decode(model, tokenizer, prompt, specs, limit, keep_sequences=False):
        events.append(('decode', prompt['qid'], limit['max_new_tokens'], keep_sequences))
        return _decoded(prompt)

    monkeypatch.setattr(office_traj, '_warmup_arms', warmup)
    monkeypatch.setattr(office_traj, 'decode_measured_prompt', decode)
    monkeypatch.setattr(office_traj, '_run_identity', lambda: {'git_head': 'abc', 'dirty': False})
    monkeypatch.setattr(office_traj, 'ablation_summary', lambda *args, **kwargs: {'wall': {'complete': True}})
    cfg = round_config('xcat7')
    _formal(None, None, _prompts()[:1], (), 'digest', tmp_path, cfg)
    assert events == [('warmup', 0), ('decode', 0, EVAL['max_new_tokens'], False)]
    header = json.loads((tmp_path / f"{cfg['prefix']}-header-{cfg['stamp']}.json").read_text(encoding='utf-8'))
    assert header['wall_protocol'] == PROTOCOL_WALL
    assert header['wall_delta'] == 'wall_seconds_traj - wall_seconds_full'
    line = (tmp_path / f"{cfg['prefix']}-full-{cfg['stamp']}.jsonl").read_text(encoding='utf-8')
    assert '"wall_seconds": 0.2' in line
    assert '"accepted": 2' in line


def test_formal_stops_when_warmup_sync_or_argmax_fails(tmp_path, monkeypatch):
    from studies.trajectory_observation.structure import office_traj

    events = []

    def arm_row(model, spec, ids, limit):
        events.append((spec['observation']['gain'], limit['max_new_tokens']))
        if spec['observation']['gain'] == 0.0:
            raise RuntimeError('argmax audit failed')
        return {'argmax_passed': True}

    monkeypatch.setattr(
        'studies.trajectory_observation.structure.office_tree.arm_row', arm_row)
    monkeypatch.setattr(office_traj, 'decode_measured_prompt', lambda *args, **kwargs: events.append('decode'))
    monkeypatch.setattr(office_traj, '_run_identity', lambda: {'git_head': 'abc', 'dirty': False})
    cfg = round_config('xcat7')

    class _Tokenizer:
        def __call__(self, text, return_tensors='pt'):
            return type('Batch', (), {'input_ids': [[1]]})()

    with pytest.raises(RuntimeError, match='argmax audit failed'):
        _formal(type('Model', (), {'device': 'cpu'})(), _Tokenizer(), _prompts()[:1], (), 'digest', tmp_path, cfg)
    assert events == [(0.5, 8), (0.0, 8)]
    assert 'decode' not in events

    events.clear()

    def both_arms(model, spec, ids, limit):
        events.append('arm')
        return {'argmax_passed': True}

    def fail_sync(device):
        events.append('sync')
        raise RuntimeError('sync failed')

    monkeypatch.setattr(
        'studies.trajectory_observation.structure.office_tree.arm_row', both_arms)
    monkeypatch.setattr('spec_mtp.model.sync_device', fail_sync)
    with pytest.raises(RuntimeError, match='sync failed'):
        _formal(type('Model', (), {'device': 'cpu'})(), _Tokenizer(), _prompts()[:1], (), 'digest', tmp_path, cfg)
    assert events == ['arm', 'arm', 'sync']
    assert 'decode' not in events
