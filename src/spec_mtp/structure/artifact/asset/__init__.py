"""Asset path helpers."""

from __future__ import annotations

from pathlib import Path

from spec_mtp.structure.artifact.asset import kinds
from spec_mtp.structure.artifact.asset.tree import list_asset_files
from spec_mtp.structure.artifact.layout import ArtifactLayout


def ensure_assets(layout: ArtifactLayout) -> Path:
    layout.assets_dir.mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / kinds.TRACKER).mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / kinds.LOGS).mkdir(parents=True, exist_ok=True)
    (layout.assets_dir / kinds.CHECKPOINTS).mkdir(parents=True, exist_ok=True)
    return layout.assets_dir


__all__ = ['ensure_assets', 'list_asset_files', 'kinds']
