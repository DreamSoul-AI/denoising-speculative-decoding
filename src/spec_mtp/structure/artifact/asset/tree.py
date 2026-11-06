"""List files under a Run assets directory."""

from __future__ import annotations

from pathlib import Path
import re
import stat


def list_asset_files(assets_dir: Path | str) -> list[str]:
    root = Path(assets_dir)
    if not root.is_dir():
        return []
    files = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]
    return sorted(files)


def diagnostic_files(path: Path | str) -> list[Path]:
    """Return the anchor and contiguous numbered parts without reading byte bodies."""
    path = Path(path)
    if path.name != 'events.jsonl':
        return [path]
    parent = path.parent.resolve()
    path = parent / path.name
    parts = {0: path}
    for candidate in parent.glob('events.*.jsonl'):
        match = re.fullmatch(r'events\.([0-9]+)\.jsonl', candidate.name)
        if match is None:
            continue
        index = int(match[1])
        if index in parts:
            raise ValueError('duplicate diagnostic part index')
        parts[index] = candidate
    if len(parts) > 1 and not path.is_file():
        raise ValueError('diagnostic parts require their anchor')
    indices = sorted(parts)
    if indices != list(range(len(parts))):
        raise ValueError('diagnostic part indices must be contiguous')
    identities = set()
    for candidate in parts.values():
        if candidate.resolve() != candidate:
            raise ValueError('diagnostic file must not be an alias')
        try:
            info = candidate.stat()
        except FileNotFoundError:
            if len(parts) == 1:
                continue
            raise
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('diagnostic parts must be regular files')
        if info.st_nlink > 1:
            raise ValueError('diagnostic file must not be hardlinked')
        identity = (info.st_dev, info.st_ino)
        if identity in identities:
            raise ValueError('diagnostic files must not alias one another')
        identities.add(identity)
    return [parts[index] for index in indices]
