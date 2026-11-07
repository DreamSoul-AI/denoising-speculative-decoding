"""List files under a Run assets directory."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterator

CAPTURE_MAX_BYTES = 64 * 1024 * 1024
CAPTURE_WINDOW = timedelta(days=7)
DIAGNOSTIC_BUDGET_OVERHEAD = 8192
DIAGNOSTIC_RUN_OVERHEAD = 16384


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

# These controls remain after purge so a capture cannot silently become writable again.
def _capture_controls(anchor: Path) -> dict[str, Path]:
    assets = anchor.parent.parent
    return {key: assets / ('.diagnostic-capture' + suffix) for key, suffix in
            (('required', '.json'), ('lock', '.lock'), ('seal', '.seal.json'), ('journal', '.purge.json'))}


def _capture_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _capture_identity(path: Path) -> tuple[int, int, int, int, int]:
    if path.resolve() != path:
        raise ValueError('capture control or member must not be an alias')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError('capture control or member must be regular and unlinked')
    return info.st_dev, info.st_ino, info.st_nlink, info.st_size, info.st_mtime_ns


def _capture_anchor(anchor: Path | str) -> Path:
    anchor = Path(anchor).absolute()
    if (anchor.name != 'events.jsonl' or anchor.parent.name != 'logs'
            or anchor.parent.parent.resolve() != anchor.parent.parent
            or anchor.parent.resolve() != anchor.parent):
        raise ValueError('capture requires its canonical diagnostic namespace')
    return anchor


def _read_capture_control(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    before = _capture_identity(path)
    with path.open('rb') as stream:
        raw = stream.read(4097)
    if len(raw) > 4096 or _capture_identity(path) != before:
        raise ValueError('capture control exceeded its bound or changed while read')
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('capture seal SHA256 differs')
    def unique(pairs):
        body = {}
        for key, value in pairs:
            if key in body: raise ValueError('duplicate capture control field')
            body[key] = value
        return body
    body = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    if not isinstance(body, dict): raise ValueError('capture control must be an object')
    return body


def _write_capture_control(path: Path, body: dict[str, Any], *, exclusive: bool = False) -> str:
    from spec_mtp.structure.artifact._atomic import atomic_write_text

    text = json.dumps(body, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':')) + '\n'
    if len(text) > 4096: raise ValueError('capture control exceeds 4096 bytes')
    required = path.parent / '.diagnostic-capture.json'
    reserved = (path.name == '.diagnostic-budget.json' or 'reservation' in body
                or (_capture_exists(required) and _read_capture_control(required).get('schema') == 2))
    if reserved:
        if any(_capture_exists(path.parent / name) for name in
               ('.diagnostic-budget.json.stage', '.diagnostic-capture.json.stage',
                '.diagnostic-capture.seal.json.stage', '.diagnostic-capture.purge.json.stage')):
            raise FileExistsError('diagnostic reservation has an unfinished control write')
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        _capture_identity(path)
        if reserved:
            stage = path.with_name(path.name + '.stage')
            with stage.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(stage, path)  # Failure leaves the bounded stage as a permanent refusal fence.
        else:
            atomic_write_text(path, text)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@contextmanager
def _diagnostic_lease(anchor: Path | str, *, initialize: bool = False) -> Iterator[dict[str, Any]]:
    anchor = _capture_anchor(anchor)
    paths = _capture_controls(anchor)
    def marker():
        body = _read_capture_control(paths['required'])
        reserved = body.get('schema') == 2
        if (set(body) != {'schema', 'state', 'assets'} | ({'reservation'} if reserved else set())
                or type(body['schema']) is not int or body['schema'] not in (1, 2)
                or body['state'] not in ('pending', 'ready', 'closed')
                or body['assets'] != str(anchor.parent.parent.resolve())):
            raise ValueError('invalid capture ownership marker')
        if reserved:
            reservation = body['reservation']
            if (not isinstance(reservation, dict)
                    or set(reservation) != {'ledger_sha256', 'max_bytes', 'policy_sha256'}
                    or type(reservation['max_bytes']) is not int
                    or not 4096 <= reservation['max_bytes'] <= CAPTURE_MAX_BYTES
                    or any(not isinstance(reservation[key], str)
                           or re.fullmatch('[0-9a-f]{64}', reservation[key]) is None
                           for key in ('ledger_sha256', 'policy_sha256'))):
                raise ValueError('invalid diagnostic reservation binding')
        return body
    initial = marker()
    expected_lock = None
    if initialize:
        if initial['state'] != 'pending': raise ValueError('capture lock initialization requires pending ownership')
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    else:
        expected_lock = _capture_identity(paths['lock'])[:3]
        flags = os.O_RDWR
    fd = os.open(paths['lock'], flags | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (expected_lock is not None and expected_lock != (info.st_dev, info.st_ino, info.st_nlink))
                or _capture_identity(paths['lock'])[:3] != (info.st_dev, info.st_ino, info.st_nlink)):
            raise ValueError('capture lock identity changed')
        if sys.platform == 'win32':
            import msvcrt
            try: msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise BlockingIOError('diagnostic capture writer is busy') from exc
        elif os.name == 'posix':
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            raise NotImplementedError('native diagnostic locking is unavailable')
        if _capture_identity(paths['lock'])[:3] != (info.st_dev, info.st_ino, info.st_nlink):
            raise ValueError('capture lock was replaced')
        yield marker()
    except BaseException as primary:
        try: os.close(fd)
        except BaseException as cleanup:
            setattr(primary, 'cleanup_errors', (*getattr(primary, 'cleanup_errors', ()), cleanup))
        raise
    else:
        os.close(fd)  # Closing the owned descriptor releases the native lock, including on process exit.


@contextmanager
def diagnostic_write_guard(anchor: Path | str) -> Iterator[bool]:
    anchor = Path(anchor)
    controls = _capture_controls(anchor)
    if not any(_capture_exists(path) for path in controls.values()):
        _check_capture_reservation(anchor, None)
        yield False
        return
    if _read_capture_control(controls['required']).get('state') == 'closed':
        raise PermissionError('diagnostic capture is not writable')
    with _diagnostic_lease(anchor) as marker:
        if (marker['state'] != 'ready' or _capture_exists(controls['seal'])
                or _capture_exists(controls['journal'])):
            raise PermissionError('diagnostic capture is not writable')
        _check_capture_reservation(anchor, marker)
        yield True


def check_diagnostic_binding(anchor: Path | str, owned: bool) -> None:
    if not owned and any(_capture_exists(path) for path in _capture_controls(Path(anchor)).values()):
        raise BlockingIOError('diagnostic ownership changed; reopen the Logger')


def prepare_diagnostic_capture(anchor: Path | str, *, reservation: dict[str, Any] | None = None) -> dict[str, str]:
    anchor = _capture_anchor(anchor)
    paths = _capture_controls(anchor)
    if _capture_exists(anchor.parent) or any(_capture_exists(path) for path in paths.values()):
        raise FileExistsError('capture ownership requires a new diagnostic directory')
    if reservation is None:
        _check_capture_reservation(anchor, None)
    anchor.parent.parent.mkdir(parents=True, exist_ok=True)
    body = {'schema': 1, 'state': 'pending', 'assets': str(anchor.parent.parent.resolve())}
    if reservation is not None:
        body.update(schema=2, reservation=reservation)
    _write_capture_control(paths['required'], body, exclusive=True)
    with _diagnostic_lease(anchor, initialize=True) as marker:
        if marker['state'] != 'pending': raise ValueError('capture is not pending')
        anchor.parent.mkdir()  # An ordinary writer winning this race leaves the pending fence intact.
        _write_capture_control(paths['required'], {**body, 'state': 'ready'})
    return {'state': 'ready'}


def _budget_declaration(study: Path) -> int | None:
    from spec_mtp.structure.make.expand import load_study_yaml

    if not _capture_exists(study / 'study.yaml'):
        return None
    body = load_study_yaml(study)
    if 'diagnostic_budget_bytes' not in body:
        return None
    value = body['diagnostic_budget_bytes']
    if type(value) is not int or value <= 0:
        raise ValueError('diagnostic_budget_bytes must be a positive integer')
    return value


def _budget_share(budget: int, count: int) -> int:
    if type(count) is not int or count <= 0:
        raise ValueError('diagnostic reservations require a finite nonempty Run set')
    share = min(CAPTURE_MAX_BYTES, (budget - DIAGNOSTIC_BUDGET_OVERHEAD
                                   - count * DIAGNOSTIC_RUN_OVERHEAD) // count)
    if share < 4096:
        raise ValueError('diagnostic budget cannot reserve every Run and its controls')
    return share


def _budget_digest(path: Path) -> str:
    from spec_mtp.structure.artifact.provenance import file_sha256

    before = _capture_identity(path)
    digest = file_sha256(path)
    if _capture_identity(path) != before:
        raise ValueError('diagnostic budget input changed while read')
    return digest


def check_diagnostic_budget(study_dir: Path | str, run_id: str | None = None) -> int | None:
    """Validate an immutable reservation; closed/purged Runs remain charged."""
    from spec_mtp.structure.artifact.layout import validate_run_directory

    study = Path(study_dir).absolute()
    ledger = study / '.diagnostic-budget.json'
    budget = _budget_declaration(study)
    if budget is None and not _capture_exists(ledger):
        if run_id is not None:
            validate_run_directory(run_id)
            required = study / 'runs' / run_id / 'assets' / '.diagnostic-capture.json'
            if _capture_exists(required) and 'reservation' in _read_capture_control(required):
                raise ValueError('required diagnostic budget is missing')
        source = study / 'study_source.json'
        if _capture_exists(source):
            # A removed declaration must not erase the frozen required-budget identity.
            binding = json.loads(source.read_text(encoding='utf-8'))
            if binding['files']['study.yaml'] != _budget_digest(study / 'study.yaml'):
                raise ValueError('frozen diagnostic declaration changed')
        return None
    if study.resolve() != study:
        raise ValueError('diagnostic budget requires its canonical Study root')
    body = _read_capture_control(ledger)
    if _capture_exists(ledger.with_name(ledger.name + '.stage')):
        raise ValueError('diagnostic budget has an unfinished control write')
    if (set(body) != {'schema', 'state', 'study', 'budget_bytes', 'run_count', 'max_bytes',
                     'study_sha256', 'base_sha256', 'source_sha256', 'index_sha256', 'runs'}
            or type(body['schema']) is not int or body['schema'] != 1 or body['state'] != 'ready'
            or body['study'] != str(study) or type(body['budget_bytes']) is not int
            or body['budget_bytes'] != budget or budget is None or not isinstance(body['runs'], dict)
            or type(body['run_count']) is not int or body['run_count'] != len(body['runs'])
            or type(body['max_bytes']) is not int
            or body['max_bytes'] != _budget_share(budget, body['run_count'])):
        raise ValueError('diagnostic budget is incomplete or changed')
    for name, key in (('study.yaml', 'study_sha256'), ('experiment_config.yaml', 'base_sha256'),
                      ('study_source.json', 'source_sha256'), ('index.json', 'index_sha256')):
        if _budget_digest(study / name) != body[key]:
            raise ValueError('diagnostic budget source binding changed')
    if run_id is not None and run_id not in body['runs']:
        raise ValueError('Run has no diagnostic reservation')
    for name in ([run_id] if run_id is not None else body['runs']):
        validate_run_directory(name)
        if _budget_digest(study / 'runs' / name / 'config.yaml') != body['runs'][name]:
            raise ValueError('diagnostic reservation config changed')
        anchor = study / 'runs' / name / 'assets' / 'logs' / 'events.jsonl'
        controls = _capture_controls(anchor)
        marker = _read_capture_control(controls['required'])
        binding = marker.get('reservation')
        if (type(marker.get('schema')) is not int or marker['schema'] != 2
                or marker.get('assets') != str(anchor.parent.parent)
                or marker.get('state') not in ('ready', 'closed')
                or not isinstance(binding, dict) or binding.get('max_bytes') != body['max_bytes']
                or binding.get('ledger_sha256') != _budget_digest(ledger)):
            raise ValueError('diagnostic Run reservation binding changed')
        if _capture_identity(controls['lock'])[3] != 0:
            raise ValueError('diagnostic reservation lock changed')
        if any(_capture_exists(path.with_name(path.name + '.stage'))
               for key, path in controls.items() if key != 'lock'):
            raise ValueError('diagnostic reservation has an unfinished control write')
        if marker['state'] == 'ready':
            policy = read_capture_policy(anchor, datetime.now(timezone.utc))
            if (policy is None or policy[0] != body['max_bytes']
                    or _budget_digest(anchor.with_name('events.policy.json')) != binding.get('policy_sha256')):
                raise ValueError('diagnostic reservation policy changed')
    return body['max_bytes']


def begin_diagnostic_budget(study_dir: Path | str, run_count: int) -> bool:
    """Fence a new explicit external Study before any Run config is written."""
    from spec_mtp.structure.artifact.layout import check_study_snapshot, _reject_source_checkout

    study = Path(study_dir).absolute()
    budget = _budget_declaration(study)
    ledger = study / '.diagnostic-budget.json'
    if _capture_exists(ledger):
        check_diagnostic_budget(study)
        if _read_capture_control(ledger)['run_count'] != run_count:
            raise ValueError('diagnostic reservation Run set changed')
        return True
    if budget is None:
        return False
    if study.resolve() != study or not _capture_exists(study / 'study_source.json'):
        raise ValueError('diagnostic budget requires a new materialized external Study')
    _reject_source_checkout(study)
    check_study_snapshot(study)
    maximum = _budget_share(budget, run_count)
    if _capture_exists(study / 'index.json') or (study / 'runs').exists() and any((study / 'runs').iterdir()):
        raise FileExistsError('diagnostic budget cannot adopt existing Runs')
    _write_capture_control(ledger, {'schema': 1, 'state': 'pending', 'study': str(study),
                                  'budget_bytes': budget, 'run_count': run_count, 'max_bytes': maximum}, exclusive=True)
    return False


def finish_diagnostic_budget(study_dir: Path | str, configs: list[Path]) -> None:
    """Reserve all fixed shares before publishing ready; failures remain fenced."""
    from spec_mtp.structure.artifact.index import load_index
    from spec_mtp.structure.artifact.layout import validate_run_directory

    study = Path(study_dir).absolute()
    ledger = study / '.diagnostic-budget.json'
    budget = _budget_declaration(study)
    if budget is None and not _capture_exists(ledger):
        return
    pending = _read_capture_control(ledger)
    if (budget is None or pending.get('state') != 'pending' or pending.get('study') != str(study)
            or pending.get('budget_bytes') != budget or pending.get('run_count') != len(configs)
            or pending.get('max_bytes') != _budget_share(budget, len(configs))):
        raise ValueError('diagnostic budget cannot be resumed or resized')
    runs = {}
    for config in configs:
        config = Path(config).absolute()
        name = config.parent.name
        validate_run_directory(name)
        if config != study / 'runs' / name / 'config.yaml' or name in runs:
            raise ValueError('diagnostic reservation requires unique canonical Runs')
        runs[name] = _budget_digest(config)
    index = load_index(study)
    indexed = [row for experiment in index['experiments'] for row in experiment['runs']]
    if {row['run_dir'] for row in indexed} != set(runs) or len(indexed) != len(runs):
        raise ValueError('diagnostic budget index differs from its finite Run set')
    body = {**pending, 'state': 'ready', 'runs': runs}
    for name, key in (('study.yaml', 'study_sha256'), ('experiment_config.yaml', 'base_sha256'),
                      ('study_source.json', 'source_sha256'), ('index.json', 'index_sha256')):
        body[key] = _budget_digest(study / name)
    # Serialize before preparing any Run; oversized ledgers leave only the pending fence.
    payload = json.dumps(body, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':')) + '\n'
    if len(payload) > 4096:
        raise ValueError('diagnostic budget control exceeds 4096 bytes')
    ledger_sha = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    now = datetime.now(timezone.utc)
    policy = {'schema': 1, 'owner': 'run_operator', 'max_bytes': pending['max_bytes'],
              'created_at': now.isoformat().replace('+00:00', 'Z'),
              'write_until': (now + CAPTURE_WINDOW).isoformat().replace('+00:00', 'Z')}
    policy_payload = json.dumps(policy, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':')) + '\n'
    policy_sha = hashlib.sha256(policy_payload.encode('utf-8')).hexdigest()
    for name in runs:
        anchor = study / 'runs' / name / 'assets' / 'logs' / 'events.jsonl'
        prepare_diagnostic_capture(anchor, reservation={'ledger_sha256': ledger_sha,
                                   'max_bytes': pending['max_bytes'], 'policy_sha256': policy_sha})
        with anchor.open('xb'):
            pass
        _write_capture_control(anchor.with_name('events.policy.json'), policy, exclusive=True)
    _write_capture_control(ledger, body)
    check_diagnostic_budget(study)


def _check_capture_reservation(anchor: Path, marker: dict[str, Any] | None) -> None:
    assets = anchor.parent.parent.absolute()
    run = assets.parent
    reservation = marker.get('reservation') if marker is not None else None
    if assets.name != 'assets' or run.parent.name != 'runs':
        if reservation is not None:
            raise ValueError('diagnostic reservation is outside its Run')
        return
    study = run.parent.parent
    maximum = check_diagnostic_budget(study, run.name)
    if maximum is None and reservation is None:
        return
    if maximum is None or not isinstance(reservation, dict):
        raise ValueError('diagnostic reservation is missing or changed')


def _sealed_diagnostic_files(anchor: Path) -> list[dict[str, Any]]:
    from spec_mtp.structure.artifact.provenance import file_sha256

    members = diagnostic_files(anchor)
    if not _capture_exists(anchor): raise ValueError('capture has no diagnostic anchor')
    policy = anchor.with_name('events.policy.json')
    read_capture_policy(anchor, datetime.now(timezone.utc))
    if _capture_exists(policy): members.append(policy)
    expected = {path.name for path in members}
    if {path.name for path in anchor.parent.iterdir()} != expected:
        raise ValueError('capture contains legacy or unknown diagnostic members')
    rows = []
    for path in members:
        before = _capture_identity(path)
        digest = file_sha256(path)
        if _capture_identity(path) != before: raise ValueError('capture member changed while sealed')
        rows.append({'name': path.name, 'bytes': before[3], 'sha256': digest})
    if {path.name for path in anchor.parent.iterdir()} != expected:
        raise ValueError('capture family changed while sealed')
    return rows


def seal_diagnostic_capture(anchor: Path | str) -> str:
    anchor = _capture_anchor(anchor)
    paths = _capture_controls(anchor)
    with _diagnostic_lease(anchor) as marker:
        if marker['state'] != 'ready' or _capture_exists(paths['seal']) or _capture_exists(paths['journal']):
            raise PermissionError('capture cannot be sealed again')
        _write_capture_control(paths['required'], {**marker, 'state': 'closed'})
        files = _sealed_diagnostic_files(anchor)
        return _write_capture_control(paths['seal'], {'schema': 1, 'assets': marker['assets'], 'files': files}, exclusive=True)


def purge_sealed_diagnostics(anchor: Path | str, expected_seal_sha256: str) -> dict[str, Any]:
    anchor = _capture_anchor(anchor)
    paths = _capture_controls(anchor)
    if not isinstance(expected_seal_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_seal_sha256):
        raise ValueError('purge requires an explicit seal SHA256')
    with _diagnostic_lease(anchor) as marker:
        if marker['state'] != 'closed': raise PermissionError('capture has not been closed')
        if _capture_exists(paths['journal']): raise FileExistsError('purge journal already exists; inspect manually')
        seal = _read_capture_control(paths['seal'], expected_sha256=expected_seal_sha256)
        if (set(seal) != {'schema', 'assets', 'files'} or type(seal['schema']) is not int
                or seal['schema'] != 1 or seal['assets'] != marker['assets']
                or seal['files'] != _sealed_diagnostic_files(anchor)):
            raise ValueError('capture differs from its sealed family')
        journal = {'schema': 1, 'state': 'pending', 'seal_sha256': expected_seal_sha256,
                   'files': [row['name'] for row in seal['files']]}
        identities = {row['name']: _capture_identity(anchor.parent / row['name']) for row in seal['files']}
        directory = anchor.parent.stat()
        directory_id = (directory.st_dev, directory.st_ino)
        def unchanged_directory():
            _capture_anchor(anchor)
            current = anchor.parent.stat()
            if (current.st_dev, current.st_ino) != directory_id:
                raise ValueError('capture directory identity changed')
        _write_capture_control(paths['journal'], journal, exclusive=True)
        unchanged_directory()
        if _sealed_diagnostic_files(anchor) != seal['files']:
            raise ValueError('capture changed after purge journal')
        for row in seal['files']:
            unchanged_directory()
            member = anchor.parent / row['name']
            if _capture_identity(member) != identities[row['name']]:
                raise ValueError('capture member identity changed before unlink')
            member.unlink()
        unchanged_directory()
        if any(anchor.parent.iterdir()):
            raise ValueError('capture gained members during purge')
        completed = {**journal, 'state': 'purged'}
        _write_capture_control(paths['journal'], completed)
        return completed
