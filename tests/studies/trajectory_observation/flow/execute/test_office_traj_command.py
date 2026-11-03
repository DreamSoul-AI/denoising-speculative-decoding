"""The trajectory ablation command stays in flow and refuses a missing output directory."""

from pathlib import Path

import pytest

from studies.trajectory_observation.flow.execute.office_traj import main


def test_command_rejects_a_missing_destination():
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(RuntimeError, match='already exist'):
        main(['--dest', str(Path('missing-office-traj-output')), '--only', 'smoke'])
