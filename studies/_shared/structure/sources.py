"""SpecMTP source manifest: provenance for the package that lives outside the Study.

RPipe's ``provenance.json`` hashes the installed ``rpipe`` package and the Study
two columns only. ``spec_mtp`` is an external project package, so its sources
are not covered by Study auto-discovery. This module hashes ``spec_mtp/**/*.py``,
records the git head / dirty flag of the checkout and the installed RPipe
commit, and gives the freeze check for those files.

- make writes the Study lock ``spec_mtp_sources.json`` (gitignored);
- every Run copies the live manifest into ``assets/spec_mtp_sources.json``;
- with ``freeze: true`` a digest different from the lock fails the Run before
  construction.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCK_NAME = 'spec_mtp_sources.json'
RUN_ASSET_NAME = 'spec_mtp_sources.json'


def package_dir() -> Path:
    import spec_mtp

    return Path(spec_mtp.__file__).resolve().parent


def package_files() -> dict[str, str]:
    root = package_dir()
    files = {
        'spec_mtp/' + path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*.py'))
        if '__pycache__' not in path.parts
    }
    shared = Path(__file__).resolve().parents[1]
    files.update({'studies/_shared/' + p.relative_to(shared).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(shared.rglob('*.py')) if '__pycache__' not in p.parts})
    return files


def files_digest(files: dict[str, str]) -> str:
    text = json.dumps(files, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _git(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return {}
        dirty = subprocess.run(
            ['git', 'status', '--porcelain', '--', '.'], cwd=root, capture_output=True, text=True, timeout=30,
        )
        return {'head': head.stdout.strip(), 'package_dirty': bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {}


def rpipe_origin() -> dict[str, Any]:
    """Vendored ``src/rpipe`` commit, or the installed distribution if that is missing."""
    import rpipe

    origin = Path(rpipe.__file__).resolve().parent / 'ORIGIN'
    if origin.is_file():
        commit = origin.read_text(encoding='utf-8').strip().split()[0]
        return {
            'version': getattr(rpipe, '__version__', None),
            'commit': commit,
            'vendored': True,
            'files_digest': files_digest({p.relative_to(origin.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(origin.parent.rglob('*.py')) if '__pycache__' not in p.parts}),
            'path': str(Path(rpipe.__file__).resolve().parent),
        }
    from importlib import metadata

    out: dict[str, Any] = {}
    try:
        dist = metadata.distribution('rpipe')
    except metadata.PackageNotFoundError:
        return {'installed': False}
    out['version'] = dist.version
    raw = dist.read_text('direct_url.json')
    if raw:
        body = json.loads(raw)
        out['url'] = body.get('url')
        vcs = body.get('vcs_info') or {}
        if vcs.get('commit_id'):
            out['commit'] = vcs['commit_id']
        if (body.get('dir_info') or {}).get('editable'):
            out['editable'] = True
    return out


def manifest() -> dict[str, Any]:
    files = package_files()
    return {
        'package': 'spec_mtp',
        'digest': files_digest(files),
        'files': files,
        'git': _git(package_dir()),
        'rpipe': rpipe_origin(),
    }


class SpecMtpSourceChangedError(RuntimeError):
    pass


def lock_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / LOCK_NAME


def write_lock(study_dir: Path | str) -> Path:
    body = manifest()
    body['inputs'] = declared_inputs(Path(study_dir))
    body['created_at'] = datetime.now(timezone.utc).isoformat()
    path = lock_path(study_dir)
    old = load_lock(study_dir)
    if old is not None and (changed_files(old, body) or old.get('rpipe') != body.get('rpipe') or old.get('inputs') != body.get('inputs')):
        from rpipe.structure.artifact.config import load_config
        from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge
        from rpipe.structure.make.expand import expand_patches, load_study_yaml

        study = load_study_yaml(Path(study_dir))
        base = experiment_config_from_mapping(load_config(Path(study_dir) / 'experiment_config.yaml'))
        for patch in expand_patches(study):
            run = run_config_from_merge(base, patch)
            if (Path(study_dir) / 'runs' / run.id / 'result.json').exists():
                raise SpecMtpSourceChangedError('new source requires a new algorithm.config.version before make/remake')
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def load_lock(study_dir: Path | str) -> dict[str, Any] | None:
    """Return the Study lock, or None when the file is absent.

    Raises ``SpecMtpSourceChangedError`` when the file exists but is not valid
    JSON or is not a JSON object, so freeze / make failures name the lock path.
    """
    path = lock_path(study_dir)
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise SpecMtpSourceChangedError(
            f'{path} is not valid JSON ({error.msg} at char {error.pos}); '
            f'remove it or run `python -m spec_mtp make {study_dir}` again') from error
    if not isinstance(body, dict):
        raise SpecMtpSourceChangedError(
            f'{path} must be a JSON object, got {type(body).__name__}; '
            f'remove it or run `python -m spec_mtp make {study_dir}` again')
    return body


def changed_files(lock: dict[str, Any], current: dict[str, Any]) -> list[str]:
    before, now = lock.get('files') or {}, current.get('files') or {}
    return sorted(name for name in set(before) | set(now) if before.get(name) != now.get(name))


def check_frozen(study_dir: Path | str, study: dict[str, Any], current: dict[str, Any]) -> None:
    """Raise when ``freeze: true`` and the SpecMTP sources differ from the make lock."""
    lock = load_lock(study_dir)
    if lock is None and study.get('freeze') is not True:
        return
    if lock is None:
        raise SpecMtpSourceChangedError(
            f'freeze: true but {LOCK_NAME} is missing; run `python -m spec_mtp make {study_dir}`')
    from studies._shared.structure.evidence import check_inputs

    check_inputs({'files': lock.get('inputs') or {}})
    if lock.get('inputs', {}) != declared_inputs(Path(study_dir)):
        raise SpecMtpSourceChangedError('declared input files changed after make')
    changed = changed_files(lock, current)
    rpipe_changed = lock.get('rpipe') != current.get('rpipe')
    if changed or rpipe_changed:
        shown = ', '.join(changed[:5]) + (' …' if len(changed) > 5 else '')
        if rpipe_changed:
            shown = (shown + ', ' if shown else '') + 'rpipe commit'
        raise SpecMtpSourceChangedError(
            f'SpecMTP sources changed after make ({shown}); run `python -m spec_mtp make` again to accept')


def declared_inputs(study_dir: Path) -> dict[str, str]:
    """Freeze local prompt/model files already available at make, without downloading."""
    from rpipe.structure.artifact.config import load_config
    from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge
    from rpipe.structure.make.expand import expand_patches, load_study_yaml
    from studies._shared.structure.evidence import file_hash

    if not (study_dir / 'experiment_config.yaml').is_file():
        return {}
    base = experiment_config_from_mapping(load_config(study_dir / 'experiment_config.yaml'))
    paths: set[Path] = set()
    for patch in expand_patches(load_study_yaml(study_dir)):
        config = run_config_from_merge(base, patch).to_mapping()
        data = config['data']
        data_cfg = data.get('config') or {}
        prompt = data_cfg.get('path')
        if not prompt and str(data.get('name')).lower() == 'specbench':
            prompt = package_dir().parents[1] / 'datasets/spec_bench/question.jsonl'
        model_cfg = config['model'].get('config') or {}
        for value in (prompt, model_cfg.get('name_or_path'), data_cfg.get('tokenizer_name')):
            if value and Path(value).exists():
                path = Path(value).resolve()
                paths.update([path] if path.is_file() else (p for p in path.rglob('*') if p.is_file()))
        if prompt and not Path(prompt).is_file():
            raise FileNotFoundError(f'declared prompt file is missing: {prompt}')
    return {str(p): file_hash(p) for p in sorted(paths)}
