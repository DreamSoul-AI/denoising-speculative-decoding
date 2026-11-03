"""The trajectory ablation command stays in flow and refuses a missing output directory."""

from pathlib import Path

import pytest

from studies.trajectory_observation.flow.execute.office_traj import main, round_config


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
