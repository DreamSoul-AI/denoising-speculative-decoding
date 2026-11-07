"""Study Artifact directory layout (shared + one Run subtree)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from spec_mtp.structure.artifact import paths

STUDY_SOURCE_NAME = 'study_source.json'


def _reject_source_checkout(root: Path) -> None:
    if any((parent / '.git').exists() or (parent / 'pyproject.toml').is_file()
           for parent in (root, *root.parents)):
        raise ValueError('source checkouts require an explicit external --run-root')


def _study_files(study: Path) -> dict[str, str]:
    from spec_mtp.structure.artifact.provenance import file_sha256, study_source_paths
    from spec_mtp.structure.make.expand import load_study_yaml

    members = study_source_paths(study, load_study_yaml(study))
    sizes = [path.stat().st_size for path in members.values()]
    if any(size > 16 * 1024 * 1024 for size in sizes) or sum(sizes) > 32 * 1024 * 1024:
        raise ValueError('Study snapshot exceeds the 16 MiB member / 32 MiB total bound')
    return {name: file_sha256(path) for name, path in members.items()}


def check_study_snapshot(study_dir: Path | str) -> None:
    """Check both the frozen Study and its still-imported original sources."""
    import json

    root = Path(study_dir).resolve()
    marker = root / STUDY_SOURCE_NAME
    if not marker.exists() and not marker.is_symlink():
        return
    if marker.is_symlink():
        raise ValueError('Study source binding must be a regular local file')
    body = json.loads(marker.read_text(encoding='utf-8'))
    if (not isinstance(body, dict) or type(body.get('schema')) is not int or body['schema'] != 1
            or not isinstance(body.get('source_dir'), str) or not body['source_dir']
            or not Path(body['source_dir']).is_absolute()
            or not isinstance(body.get('files'), dict) or not body['files']
            or not isinstance(body.get('git'), dict)):
        raise ValueError('invalid Study source binding')
    source = Path(body['source_dir']).resolve(strict=True)
    paths.validate_output_root(root, source)
    if _study_files(source) != body['files'] or _study_files(root) != body['files']:
        raise ValueError('Study source or snapshot changed; use a new run root')


def materialize_study(study_dir: Path | str, run_root: Path | str) -> Path:
    """Freeze declared static Study inputs into a separate, explicit runtime root."""
    import json
    import hashlib
    from spec_mtp.structure.artifact.provenance import capture_git, file_sha256

    source = Path(study_dir).resolve(strict=True)
    target = paths.validate_output_root(run_root, source)
    _reject_source_checkout(target)
    if target.exists() or target.is_symlink():
        marker = target / STUDY_SOURCE_NAME
        if not marker.is_file():
            raise FileExistsError('existing run root has no verified Study binding')
        check_study_snapshot(target)
        body = json.loads(marker.read_text(encoding='utf-8'))
        if Path(body['source_dir']).resolve() != source:
            raise ValueError('run root is already bound to another Study')
        return target
    files = _study_files(source)
    if STUDY_SOURCE_NAME in files:
        raise ValueError('Study input collides with the source binding')
    payloads: dict[str, bytes] = {}
    total = 0
    for name in files:
        with (source / name).open('rb') as handle:
            data = handle.read(16 * 1024 * 1024 + 1)
        total += len(data)
        if len(data) > 16 * 1024 * 1024 or total > 32 * 1024 * 1024:
            raise ValueError('Study inputs grew beyond the snapshot bound')
        payloads[name] = data
    if (any(hashlib.sha256(data).hexdigest() != files[name] for name, data in payloads.items())
            or _study_files(source) != files):
        raise ValueError('Study source changed while freezing')
    body = {'schema': 1, 'source_dir': str(source), 'files': files, 'git': capture_git(source)}
    target.mkdir(parents=True, exist_ok=False)
    for name, sha in files.items():
        copied, data = target / name, payloads[name]
        copied.parent.mkdir(parents=True, exist_ok=True)
        with copied.open('xb') as handle:
            handle.write(data)
        if file_sha256(copied) != sha:
            raise ValueError('Study snapshot write did not preserve bytes')
    with (target / STUDY_SOURCE_NAME).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(body, ensure_ascii=False, indent=2) + '\n')
    check_study_snapshot(target)
    return target


def validate_run_directory(run_dir: str) -> None:
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{n}' for prefix in ('COM', 'LPT') for n in range(1, 10)}
    if (not isinstance(run_dir, str) or not run_dir or run_dir in ('.', '..')
            or run_dir != run_dir.strip() or run_dir.endswith('.')
            or any(char in run_dir for char in '/\\:<>"|?*') or any(ord(char) < 32 for char in run_dir)
            or run_dir.split('.')[0].upper() in reserved):
        raise ValueError(f'invalid Run directory: {run_dir!r}')


@dataclass(frozen=True)
class ArtifactLayout:
    """Paths for one Run under ``<study>/runs/<run_dir>/``."""

    root: Path
    study_dir: Path

    @property
    def config_path(self) -> Path:
        return self.root / paths.CONFIG_NAME

    @property
    def result_path(self) -> Path:
        return self.root / paths.RESULT_NAME

    @property
    def assets_dir(self) -> Path:
        return self.root / paths.ASSETS_DIRNAME

    @property
    def shared_dir(self) -> Path:
        return self.study_dir / paths.SHARED_DIRNAME

    @property
    def shared_data_dir(self) -> Path:
        return self.shared_dir / 'data'

    @property
    def shared_model_dir(self) -> Path:
        return self.shared_dir / 'model'

    @property
    def docs_dir(self) -> Path:
        return self.study_dir / paths.DOCS_DIRNAME

    def ensure(self) -> None:
        if self.root.resolve() != self.root.absolute():
            raise ValueError('Run root must retain its admitted identity inside the Study')
        for member in (self.config_path, self.result_path, self.assets_dir,
                       self.assets_dir / 'logs', self.assets_dir / 'checkpoints'):
            if not member.resolve().is_relative_to(self.root.resolve()):
                raise ValueError('Run artifacts must remain inside their own Run root')
        for member in (self.shared_data_dir, self.shared_model_dir, self.docs_dir):
            if not member.resolve().is_relative_to(self.study_dir.resolve()):
                raise ValueError('shared artifacts must remain inside the admitted Study root')
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.shared_data_dir.mkdir(parents=True, exist_ok=True)
        self.shared_model_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


def validate_study_root(study_dir: Path | str) -> Path:
    """Validate a runtime root without creating directories or updating evidence."""
    study = paths.validate_output_root(study_dir)
    _reject_source_checkout(study)
    check_study_snapshot(study)
    for name in (paths.DOCS_DIRNAME, paths.SHARED_DIRNAME, paths.RUNS_DIRNAME,
                 paths.SHARED_DIRNAME + '/data', paths.SHARED_DIRNAME + '/model'):
        if (study / name).resolve() != (study / name).absolute():
            raise ValueError('Study artifacts must remain inside the admitted runtime root')
    return study


def ensure_study_layout(study_dir: Path | str) -> Path:
    """Ensure Study skeleton: ``docs/``, ``shared/{data,model}/``, ``runs/``."""
    study = validate_study_root(study_dir)
    (study / paths.DOCS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (study / paths.SHARED_DIRNAME / 'data').mkdir(parents=True, exist_ok=True)
    (study / paths.SHARED_DIRNAME / 'model').mkdir(parents=True, exist_ok=True)
    (study / paths.RUNS_DIRNAME).mkdir(parents=True, exist_ok=True)
    return study


def artifact_layout(study_dir: Path | str, run_dir: str) -> ArtifactLayout:
    """Build layout under ``study_dir/runs/<run_dir>/``.

    ``run_dir`` is the content-derived Config ``id``.
    """
    validate_run_directory(run_dir)
    study = ensure_study_layout(study_dir)
    layout = ArtifactLayout(root=study / paths.RUNS_DIRNAME / run_dir, study_dir=study)
    layout.ensure()
    return layout
