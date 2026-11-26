"""Run a frozen input-state experiment configuration."""

import argparse
import json
from pathlib import Path

from studies.trajectory_observation.structure.input_state import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    run_experiment(json.loads(args.config.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
