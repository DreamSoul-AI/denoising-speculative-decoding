"""Write Run configs and the Study index."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

from rpipe.structure.artifact import (
    artifact_layout,
    build_index,
    experiment_entries,
    load_config,
    write_config,
    write_index,
)
from rpipe.structure.artifact.execution import run_lock
from rpipe.structure.artifact.paths import RUNS_DIRNAME
from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge
from rpipe.structure.make.expand import axis_keys_from_study


def _run_configs(study_dir: Path, patches: list[dict[str, Any]]):
    base_path = study_dir / 'experiment_config.yaml'
    if not base_path.is_file():
        raise FileNotFoundError(f'missing experiment_config.yaml: {base_path}')
    base = experiment_config_from_mapping(load_config(base_path))
    runs = [run_config_from_merge(base, patch) for patch in patches]
    ids = [run.id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate Run in Study expansion; use a new version for repeated computation')
    return runs


def _write_configs(study_dir: Path, runs) -> list[Path]:
    written: list[Path] = []
    for run in runs:
        cfg = run.to_mapping()
        layout = artifact_layout(study_dir, run.id)
        written.append(write_config(layout.config_path, cfg))
    return written


def write_run_configs(study_dir: Path, patches: list[dict[str, Any]]) -> list[Path]:
    runs = _run_configs(study_dir, patches)
    with ExitStack() as stack:
        for run in runs:
            stack.enter_context(run_lock(study_dir / RUNS_DIRNAME / run.id))
        return _write_configs(study_dir, runs)


def write_study_index(
    study_dir: Path,
    study: dict[str, Any],
    config_paths: list[Path],
) -> Path:
    loaded = [(path, load_config(path)) for path in config_paths]
    experiments = experiment_entries(
        configs=loaded,
        axis_keys=axis_keys_from_study(study),
        study_dir=study_dir,
    )
    index = build_index(
        study=str(study.get('study') or study_dir.name),
        description=str(study.get('description') or ''),
        experiments=experiments,
    )
    return write_index(study_dir, index)
