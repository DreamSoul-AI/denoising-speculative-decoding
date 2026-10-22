"""Thin entry over ``rpipe.flow.cli``.

``make`` and ``launch --remake`` write the SpecMTP source lock. ``run`` and a
plain ``launch`` keep an existing lock so ``freeze: true`` can see source drift
after make. A first ``run`` without a lock still writes one, unless the Study
sets ``freeze: true`` (then recipe reports the missing lock). Everything else
goes to RPipe unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LOCK_COMMANDS = ('make', 'run', 'launch')


def _study_arg(argv: list[str]) -> Path | None:
    args = argv[1:]
    if argv[0] == 'study' and args[:1] == ['run']:
        args = args[1:]
    for item in args:
        if not item.startswith('-'):
            path = Path(item)
            return path if (path / 'study.yaml').is_file() else None
    return None


def _command(argv: list[str]) -> str | None:
    if not argv:
        return None
    if argv[0] == 'study' and argv[1:2] == ['run']:
        return 'run'
    return argv[0]


def _study_freeze(study: Path) -> bool:
    try:
        import yaml
    except ImportError:
        return False
    try:
        body = yaml.safe_load((study / 'study.yaml').read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(body, dict) and body.get('freeze') is True


def write_lock_for(argv: list[str]) -> Path | None:
    command = _command(argv)
    if command not in _LOCK_COMMANDS:
        return None
    study = _study_arg(argv)
    if study is None:
        return None
    from studies._shared.structure.sources import load_lock, write_lock

    remake = command == 'launch' and '--remake' in argv
    if command == 'make' or remake:
        return write_lock(study)
    if load_lock(study) is not None:
        return None
    if _study_freeze(study):
        return None
    return write_lock(study)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lock = write_lock_for(argv)
    if lock is not None:
        print(f'spec_mtp sources: {lock}', flush=True)
    if _command(argv) in ('run', 'launch', 'run-one', 'process', 'report'):
        study = _study_arg(argv)
        if study is not None:
            from rpipe.structure.make.expand import load_study_yaml
            from rpipe.structure.artifact.provenance import check_frozen as check_rpipe
            from studies._shared.structure.sources import check_frozen, manifest

            declared = load_study_yaml(study)
            check_frozen(study, declared, manifest())
            check_rpipe(study, declared)
            if _command(argv) == 'report':
                from rpipe.flow.context import StudyProcessContext
                from rpipe.structure.artifact.readout.report import render_numbers
                from rpipe.structure.artifact.result import load_result
                from studies._shared.flow.process.run import study_gate

                # Reject stale identity before the quality gate can update the envelope.
                render_numbers(study)
                context = StudyProcessContext(study, state={'process': load_result(study / 'process.json')})
                study_gate(context)
    from rpipe.flow.cli import main as rpipe_main

    return int(rpipe_main(argv) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
