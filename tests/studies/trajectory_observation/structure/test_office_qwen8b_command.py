"""The 8B command refuses a missing output directory."""

from pathlib import Path

import pytest

from studies.trajectory_observation.structure.office_qwen8b import main


def test_command_rejects_a_missing_destination():
    with pytest.raises(SystemExit):
        main(['--only', 'smoke'])
    with pytest.raises(RuntimeError, match='already exist'):
        main(['--dest', str(Path('missing-office-output')), '--only', 'smoke'])
