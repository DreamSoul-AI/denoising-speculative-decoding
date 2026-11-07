"""The tree-shape command refuses a missing output directory."""

from pathlib import Path

import pytest

from studies.trajectory_observation.flow.execute.office import main


def test_command_rejects_a_missing_destination():
    with pytest.raises(SystemExit):
        main(['tree', '--only', 'smoke'])
    with pytest.raises(SystemExit):
        main(['tree', '--dest', str(Path('missing-office-tree-output')), '--only', 'smoke'])
    with pytest.raises(SystemExit):
        main(['tree', '--dest', str(Path('missing-office-8b-tree-output')), '--model', '8b', '--only', 'smoke'])
