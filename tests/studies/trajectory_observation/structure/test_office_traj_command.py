"""The trajectory ablation command refuses a missing output directory."""

from pathlib import Path

import pytest

from studies.trajectory_observation.structure.office_traj import main, round_config


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
    assert cfg['prefix'] == 'sd-office-8b-traj-xcat7'
    assert cfg['stamp'] == '20261104-045236'
    assert cfg['take'] == 128
