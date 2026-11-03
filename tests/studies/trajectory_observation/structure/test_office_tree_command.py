"""The tree-shape command refuses a missing output directory."""

from pathlib import Path

import pytest

from studies.trajectory_observation.structure.office_tree import main


def test_command_rejects_a_missing_destination():
    with pytest.raises(SystemExit):
        main(['--only', 'smoke'])
    with pytest.raises(RuntimeError, match='already exist'):
        main(['--dest', str(Path('missing-office-tree-output')), '--only', 'smoke'])
    with pytest.raises(RuntimeError, match='already exist'):
        main(['--dest', str(Path('missing-office-8b-tree-output')), '--model', '8b', '--only', 'smoke'])
