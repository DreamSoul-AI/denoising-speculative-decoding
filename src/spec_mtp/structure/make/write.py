"""Write Run configs and the Study index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spec_mtp.structure.artifact.asset import tree as diagnostic_assets
from spec_mtp.structure.artifact.layout import artifact_layout
from spec_mtp.structure.artifact.index import build_index, experiment_entries, index_path, load_index, write_index
from spec_mtp.structure.artifact.config.io import load_config, write_config
from spec_mtp.structure.artifact.paths import CONFIG_NAME, RUNS_DIRNAME
from spec_mtp.structure.control.run_config import experiment_config_from_mapping, run_config_from_merge
from spec_mtp.structure.make.expand import axis_keys_from_study, expand_patches, load_study_yaml
from spec_mtp.structure.artifact.layout import ensure_study_layout, materialize_study
from spec_mtp.structure.artifact.provenance import provenance_path, write_provenance
from spec_mtp.structure.make.recipe import load_recipe


def write_run_configs(study_dir: Path, patches: list[dict[str, Any]]) -> list[Path]:
    base_path = study_dir / 'experiment_config.yaml'
    if not base_path.is_file():
        raise FileNotFoundError(f'missing experiment_config.yaml: {base_path}')
    base = experiment_config_from_mapping(load_config(base_path))
    runs = [run_config_from_merge(base, patch) for patch in patches]
    ids = [run.id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate Run in Study expansion; use a new version for repeated computation')
    written: list[Path] = []
    for run in runs:
        cfg = run.to_mapping()
        layout = artifact_layout(study_dir, run.id)
        written.append(write_config(layout.config_path, cfg))
    return written


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


def expand_study(study_dir: Path | str, *, run_root: Path | str | None = None) -> dict[str, Any]:
    """Write Run configs and index from ``study.yaml``."""
    if run_root is not None:
        study_dir = materialize_study(study_dir, run_root)
    resolved = ensure_study_layout(Path(study_dir).resolve())
    study = load_study_yaml(resolved)
    load_recipe(resolved, study)
    patches = expand_patches(study)
    if diagnostic_assets.begin_diagnostic_budget(resolved, len(patches)):
        index_file = index_path(resolved)
        index = load_index(resolved)
        configs = [resolved / RUNS_DIRNAME / (run.get('run_dir') or run['id']) / CONFIG_NAME
                   for experiment in index.get('experiments') or [] for run in experiment.get('runs') or []]
        provenance_file = provenance_path(resolved)
    else:
        configs = write_run_configs(resolved, patches)
        index_file = write_study_index(resolved, study, configs)
        provenance_file = write_provenance(resolved, study)
        diagnostic_assets.finish_diagnostic_budget(resolved, configs)
    return {
        'study_dir': resolved,
        'index': index_file,
        'configs': configs,
        'provenance': provenance_file,
        'study': study,
    }
