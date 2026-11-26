"""Frozen frontier comparison entry."""
import json
import sys
from pathlib import Path
from studies.trajectory_observation.structure.frontier_matrix import run


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
