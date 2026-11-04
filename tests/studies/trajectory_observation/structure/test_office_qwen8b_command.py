"""The 8B command refuses a missing output directory."""

from pathlib import Path

import pytest

from studies.trajectory_observation.flow.execute.office import main


def test_command_rejects_a_missing_destination():
    with pytest.raises(SystemExit):
        main(['qwen8b', '--only', 'smoke'])
    with pytest.raises(SystemExit):
        main(['qwen8b', '--dest', str(Path('missing-office-output')), '--only', 'smoke'])
