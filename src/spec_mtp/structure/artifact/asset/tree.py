"""List files under a Run assets directory."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import stat
from typing import Any

CAPTURE_MAX_BYTES = 64 * 1024 * 1024
CAPTURE_WINDOW = timedelta(days=7)


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
    try:
        candidates = [candidate for candidate in parent.iterdir()
                      if candidate.name.startswith('events.') and candidate.name.endswith('.jsonl')]
    except FileNotFoundError:
        candidates = []
    for candidate in candidates:
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


def read_capture_policy(anchor: Path | str, now: datetime) -> tuple[int, datetime, datetime, int] | None:
    anchor = Path(anchor)
    path = anchor.parent.resolve() / 'events.policy.json'
    if not path.exists() and not path.is_symlink():
        return None
    if path.resolve() != path:
        raise ValueError('capture policy must not be an alias')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('capture policy must be a regular unlinked file')
    if not anchor.is_file():
        raise ValueError('capture policy requires its existing anchor')
    with path.open('rb') as handle:
        raw = handle.read(4097)
    if len(raw) > 4096:
        raise ValueError('capture policy exceeds 4096 bytes')

    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for name, value in pairs:
            if name in fields:
                raise ValueError('capture policy contains duplicate fields')
            fields[name] = value
        return fields

    body = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_fields)
    expected = {'schema', 'owner', 'max_bytes', 'created_at', 'write_until'}
    if (not isinstance(body, dict) or set(body) != expected
            or type(body['schema']) is not int or body['schema'] != 1
            or body['owner'] != 'run_operator'
            or type(body['max_bytes']) is not int or not 0 < body['max_bytes'] <= CAPTURE_MAX_BYTES
            or not isinstance(body['created_at'], str) or not isinstance(body['write_until'], str)):
        raise ValueError('invalid capture policy')
    try:
        created = datetime.fromisoformat(body['created_at'].replace('Z', '+00:00'))
        deadline = datetime.fromisoformat(body['write_until'].replace('Z', '+00:00'))
        if (created.utcoffset() != timedelta(0) or deadline.utcoffset() != timedelta(0)
                or created > now or deadline != created + CAPTURE_WINDOW):
            raise ValueError('untrusted capture policy window')
    except (ValueError, OverflowError) as exc:
        raise ValueError('invalid capture policy window') from exc
    return body['max_bytes'], created, deadline, len(raw)


def diagnostic_inventory(anchor: Path | str, now: datetime) -> dict[str, Any]:
    """Observe only known diagnostic files, retaining every original byte."""
    anchor = Path(anchor).absolute()
    if anchor.name != 'events.jsonl' or anchor.parent.resolve() != anchor.parent:
        raise ValueError('inventory requires its canonical diagnostic anchor')

    def snapshot() -> dict[str, tuple[int, int, int, int, int]]:
        parts = diagnostic_files(anchor)
        members = [*parts, anchor.with_name('events.policy.json'), anchor.with_name('run.log')]
        observed: dict[str, tuple[int, int, int, int, int]] = {}
        identities = set()
        for member in members:
            if member.resolve() != member:
                raise ValueError('inventory member must not be an alias')
            try:
                info = member.stat()
            except FileNotFoundError:
                if member in parts and (member != anchor or len(parts) != 1):
                    raise
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('inventory members must be regular unlinked files')
            identity = (info.st_dev, info.st_ino)
            if identity in identities:
                raise ValueError('inventory members must retain separate identities')
            identities.add(identity)
            observed[member.name] = (*identity, info.st_nlink, info.st_size, info.st_mtime_ns)
        return observed

    before = snapshot()
    policy = read_capture_policy(anchor, now)
    after = snapshot()
    if before != after:
        raise ValueError('diagnostic inventory changed while observed')
    capture_bytes = sum(info[3] for name, info in before.items() if name != 'run.log')
    capture: dict[str, Any] = {'state': 'unbound', 'max_bytes': None,
                               'created_at': None, 'write_until': None}
    if policy is not None:
        maximum, created, deadline, _ = policy
        capture = {
            'state': 'expired' if now >= deadline else 'capacity' if capture_bytes >= maximum else 'within_limits',
            'max_bytes': maximum,
            'created_at': created.isoformat().replace('+00:00', 'Z'),
            'write_until': deadline.isoformat().replace('+00:00', 'Z'),
        }
    return {'files': [{'name': name, 'bytes': info[3]} for name, info in before.items()],
            'bytes': sum(info[3] for info in before.values()), 'capture_bytes': capture_bytes,
            'capture': capture, 'disposition': 'retain'}
