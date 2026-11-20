"""Read-only model-free E1 results analysis."""

import argparse
from pathlib import Path

from studies.trajectory_observation.structure.input_state_analysis import analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    analyze_run(args.run_root)


if __name__ == '__main__':
    main()
