"""Thin entry over ``spec_mtp.flow.study_cli``.

Usage: python -m studies._shared.flow.cli --help

``make`` and ``launch --remake`` write the SpecMTP source lock. ``run`` and a
plain ``launch`` keep an existing lock so ``freeze: true`` can see source drift
after make. A first ``run`` without a lock still writes one, unless the Study
sets ``freeze: true`` (then recipe reports the missing lock). Everything else
goes to spec_mtp.flow unchanged.
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
    from spec_mtp.structure.artifact.layout import ensure_study_layout

    ensure_study_layout(study)
    from studies._shared.structure.sources import load_lock, write_lock

    remake = command == 'launch' and '--remake' in argv
    if command == 'make' or remake:
        return write_lock(study)
    if load_lock(study) is not None:
        return None
    if _study_freeze(study):
        return None
    return write_lock(study)


def check_study_inputs(study_dir: Path | str) -> dict:
    """Admit saved Study inputs before any launch/revalidation side effects."""
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.make.expand import load_study_yaml
    from spec_mtp.structure.artifact.provenance import check_frozen as check_host
    from studies._shared.structure.sources import check_frozen, manifest

    study = validate_study_root(study_dir)
    declared = load_study_yaml(study)
    check_frozen(study, declared, manifest())
    check_host(study, declared)
    return declared


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from spec_mtp.flow.study_cli import prepare_study_argv

    argv = prepare_study_argv(argv)
    if any(item in ('-h', '--help') for item in argv):
        from spec_mtp.flow.study_cli import main as rpipe_main
        return int(rpipe_main(argv) or 0)
    lock = write_lock_for(argv)
    if lock is not None:
        print(f'spec_mtp sources: {lock}', flush=True)
    if _command(argv) in ('run', 'launch', 'run-one', 'process', 'report'):
        study = _study_arg(argv)
        if study is not None:
            check_study_inputs(study)
            if _command(argv) == 'report':
                from spec_mtp.flow.context import StudyProcessContext
                from spec_mtp.structure.artifact.readout.report import render_numbers
                from spec_mtp.structure.artifact.result import load_result
                from studies._shared.flow.process.run import study_gate

                # Validate saved identity before reading the scientific evidence.
                render_numbers(study)
                context = StudyProcessContext(study, state={'process': load_result(study / 'process.json')})
                study_gate(context, persist=False)
    from spec_mtp.flow.study_cli import main as rpipe_main

    return int(rpipe_main(argv) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
