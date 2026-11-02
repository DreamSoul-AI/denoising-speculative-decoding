"""Artifact IO: layout / config / result / asset / index."""

from spec_mtp.structure.artifact.asset import ensure_assets, list_asset_files
from spec_mtp.structure.artifact.config import load_config, write_config
from spec_mtp.structure.artifact.errors import (
    ArtifactError,
    CorruptArtifactError,
    MissingConfigError,
)
from spec_mtp.structure.artifact.index import (
    build_index,
    experiment_entries,
    index_path,
    load_index,
    write_index,
)
from spec_mtp.structure.artifact.layout import (
    ArtifactLayout,
    artifact_layout,
    ensure_study_layout,
)
from spec_mtp.structure.artifact.result import (
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    load_result,
    validate_result,
    write_result,
)

__all__ = [
    'ArtifactError',
    'ArtifactLayout',
    'CorruptArtifactError',
    'MissingConfigError',
    'STATUS_FAILED',
    'STATUS_SUCCEEDED',
    'artifact_layout',
    'build_index',
    'ensure_assets',
    'ensure_study_layout',
    'experiment_entries',
    'index_path',
    'list_asset_files',
    'load_config',
    'load_index',
    'load_result',
    'validate_result',
    'write_config',
    'write_index',
    'write_result',
]
