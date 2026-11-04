"""Native CLI without failed-job retry: python studies/_shared/flow/launch_once.py launch studies/<name> --round 1."""
import sys
from functools import partial
from unittest.mock import patch
import spec_mtp.flow.study_cli
from spec_mtp.flow.study_cli import main as native_main


def main(argv=None):
    original = spec_mtp.flow.study_cli.launch_jobs
    with patch.object(spec_mtp.flow.study_cli, 'launch_jobs', partial(original, retry_failed=False)):
        return native_main(sys.argv[1:] if argv is None else argv)


if __name__ == '__main__':
    raise SystemExit(main())
