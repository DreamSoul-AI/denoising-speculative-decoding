"""Study CLI without failed-job retry; preserves source and external-root checks.

Usage: python -m studies._shared.flow.launch_once --help
"""
from studies._shared.flow.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
