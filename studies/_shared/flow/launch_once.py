"""Study CLI without failed-job retry; preserves source and external-root checks."""
from studies._shared.flow.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
