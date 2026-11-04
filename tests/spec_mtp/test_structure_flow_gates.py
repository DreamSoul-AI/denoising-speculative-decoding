"""CPU gates for the structure/flow layout.

Each legacy allowlist must equal the violations present now. A new violation
fails. An allowlist entry that no longer violates fails, so the list can only
shrink. Once the same file exists on ``origin/dev``, a later edit may not add
entries either.

The byte-protected ESP upstream tree is inside ``structure/`` and is excluded
from the content gates. Its bytes are not a legacy list we can shrink.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_DIR = Path(__file__).resolve().parent / 'legacy_allowlists'
UPSTREAM = 'src/spec_mtp/structure/algorithm/reference/esp_reproduction_artifacts/upstream/'
HOST_TOP = frozenset({'__init__.py', '__main__.py', 'cli.py'})
STUDY_TOP = frozenset({'__init__.py', 'recipe.py'})
SUBLAYERS = frozenset({
    'api', 'control', 'data', 'model', 'algorithm', 'system', 'artifact', 'make',
})
DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:(?:\\\\|\\|/)[^\s'\"<>|]*")
MAIN_GUARD = re.compile(r"""^\s*if\s+__name__\s*==\s*['"]__main__['"]\s*:""", re.M)


def _posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _skip_noise(path: Path) -> bool:
    return any(part in {'__pycache__', '.git'} for part in path.parts)


def _under_upstream(rel: str) -> bool:
    return rel.startswith(UPSTREAM)


def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _py_files(*roots: Path):
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.py'):
            if _skip_noise(path):
                continue
            yield path


def file_inventory() -> list[str]:
    """Python files outside structure/flow, other than the thin entries."""
    found: list[str] = []
    host = ROOT / 'src' / 'spec_mtp'
    for path in _py_files(host):
        parts = path.relative_to(host).parts
        if parts[0] in {'structure', 'flow'}:
            continue
        if len(parts) == 1 and parts[0] in HOST_TOP:
            continue
        found.append(_posix(path))
    studies = ROOT / 'studies'
    for path in _py_files(studies):
        parts = path.relative_to(studies).parts
        if len(parts) == 1 and parts[0] == '__init__.py':
            continue
        if len(parts) < 2:
            found.append(_posix(path))
            continue
        tail = parts[1:]
        if tail[0] in {'structure', 'flow'}:
            continue
        if len(tail) == 1 and tail[0] in STUDY_TOP:
            continue
        found.append(_posix(path))
    return sorted(set(found))


def _package_name(path: Path) -> str:
    base = ROOT / 'src' if _posix(path).startswith('src/') else ROOT
    parts = list(path.relative_to(base).parts)
    return '.'.join(parts[:-1])


def _resolve(package: str, level: int, module: str | None) -> str:
    if not level:
        return module or ''
    parts = package.split('.') if package else []
    if level > len(parts):
        base: list[str] = []
    else:
        base = parts[:-level]
    if module:
        base.extend(module.split('.'))
    return '.'.join(base)


def _sublayer(module: str) -> str | None:
    parts = module.split('.')
    if 'structure' not in parts:
        return None
    index = parts.index('structure') + 1
    if index >= len(parts):
        return None
    return parts[index]


def _is_flow(module: str) -> bool:
    return 'flow' in module.split('.')


def _is_outside(module: str) -> bool:
    head = module.split('.')[0]
    return head in {'rpipe', 'studies'}


def _source_layer(rel: str) -> str | None:
    parts = rel.split('/')
    if 'structure' in parts:
        return 'structure'
    if 'flow' in parts:
        return 'flow'
    return None


def _import_targets(tree: ast.AST, package: str) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(package, node.level, node.module)
            if base:
                targets.append(base)
            for alias in node.names:
                expanded = f'{base}.{alias.name}' if base else alias.name
                if expanded and expanded != base:
                    targets.append(expanded)
        elif isinstance(node, ast.Call):
            func = node.func
            called = ''
            if isinstance(func, ast.Attribute):
                called = func.attr
            elif isinstance(func, ast.Name):
                called = func.id
            if called in {'import_module', '__import__'} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    targets.append(arg.value)
    return targets


def _shortest(module: str, hit) -> str | None:
    parts = module.split('.')
    for end in range(1, len(parts) + 1):
        candidate = '.'.join(parts[:end])
        if hit(candidate):
            return candidate
    return None


def import_direction() -> list[str]:
    """structure→flow, api→other sublayers, and spec_mtp→rpipe/studies."""
    found: list[str] = []
    roots = [ROOT / 'src' / 'spec_mtp', ROOT / 'studies']
    for path in _py_files(*roots):
        rel = _posix(path)
        if _under_upstream(rel):
            continue
        layer = _source_layer(rel)
        in_host = rel.startswith('src/spec_mtp/')
        if layer != 'structure' and not in_host:
            continue
        try:
            tree = ast.parse(_read(path))
        except SyntaxError as exc:
            raise AssertionError(f'{rel} does not parse: {exc}') from exc
        source_sub = None
        if layer == 'structure' and '/structure/' in rel:
            head = rel.split('/structure/', 1)[1].split('/')[0]
            if not head.endswith('.py'):
                source_sub = head
        for module in _import_targets(tree, _package_name(path)):
            if layer == 'structure':
                hit = _shortest(module, _is_flow)
                if hit:
                    found.append(f'{rel} -> {hit} [structure_imports_flow]')
            if source_sub == 'api':
                def other(candidate: str, _sub=SUBLAYERS) -> bool:
                    layer_name = _sublayer(candidate)
                    return layer_name in _sub and layer_name != 'api'
                hit = _shortest(module, other)
                if hit:
                    found.append(f'{rel} -> {hit} [api_imports_other_sublayer]')
            if in_host:
                hit = _shortest(module, _is_outside)
                if hit:
                    found.append(f'{rel} -> {hit} [spec_mtp_imports_rpipe_or_studies]')
    return sorted(set(found))


def _has_argparse(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == 'argparse' or alias.name.startswith('argparse.') for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'argparse':
            return True
        elif isinstance(node, ast.Name) and node.id == 'argparse':
            return True
        elif isinstance(node, ast.Attribute) and node.attr == 'argparse':
            return True
    return False


def structure_scripts() -> list[str]:
    """``structure/`` must not be a script bin: no main guard and no argparse.

    The four folded trajectory_observation commands are a closed exception.
    """
    found: list[str] = []
    for path in _py_files(ROOT / 'src' / 'spec_mtp', ROOT / 'studies'):
        rel = _posix(path)
        if '/structure/' not in rel or _under_upstream(rel):
            continue
        text = _read(path)
        if MAIN_GUARD.search(text):
            found.append(f'{rel} [main]')
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise AssertionError(f'{rel} does not parse: {exc}') from exc
        if _has_argparse(tree):
            found.append(f'{rel} [argparse]')
    return sorted(set(found))


def drive_letters() -> list[str]:
    """No drive-letter literals in structure/flow code."""
    found: list[str] = []
    roots = [ROOT / 'src' / 'spec_mtp', ROOT / 'studies']
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in {'.py', '.yml', '.yaml'}:
                continue
            if _skip_noise(path):
                continue
            rel = _posix(path)
            if _under_upstream(rel) or _source_layer(rel) is None:
                continue
            for match in DRIVE.finditer(_read(path)):
                token = match.group(0).rstrip('.,);]')
                found.append(f'{rel} {token}')
    return sorted(set(found))


def _assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _environ_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Module-level names bound to the ``os`` module or the ``environ`` mapping."""
    os_names = {'os'}
    environ_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'os':
                    os_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == 'os':
            for alias in node.names:
                if alias.name == 'environ':
                    environ_names.add(alias.asname or alias.name)
    return os_names, environ_names


def _is_environ(node: ast.AST, os_names: set[str], environ_names: set[str]) -> bool:
    if isinstance(node, ast.Name) and node.id in environ_names:
        return True
    if isinstance(node, ast.Attribute) and node.attr == 'environ' and isinstance(node.value, ast.Name):
        return node.value.id in os_names
    return False


def _writes_environ(node: ast.AST, os_names: set[str], environ_names: set[str]) -> bool:
    targets: list[ast.AST] = []
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets.extend(node.targets if isinstance(node, ast.Assign) else [node.target])
    elif isinstance(node, ast.AugAssign):
        targets.append(node.target)
    elif isinstance(node, ast.Delete):
        targets.extend(node.targets)
    for target in targets:
        if isinstance(target, ast.Subscript) and _is_environ(target.value, os_names, environ_names):
            return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr in {'setdefault', 'update', 'pop', 'clear'}:
            if _is_environ(func.value, os_names, environ_names):
                return True
        if isinstance(func, ast.Attribute) and func.attr in {'putenv', 'unsetenv'}:
            if isinstance(func.value, ast.Name) and func.value.id in os_names:
                return True
        if isinstance(func, ast.Name) and func.id in {'putenv', 'unsetenv'}:
            return True
    return False


def _import_time_nodes(nodes: list[ast.stmt]):
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If) and _is_main_if(node):
            yield from _import_time_nodes(node.orelse)
            continue
        yield node
        for body in _nested_bodies(node):
            yield from _import_time_nodes(body)


def _is_main_if(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    values = []
    for side in (test.left, test.comparators[0]):
        if isinstance(side, ast.Name):
            values.append(side.id)
        elif isinstance(side, ast.Constant):
            values.append(side.value)
    return values == ['__name__', '__main__'] or values == ['__main__', '__name__']


def _nested_bodies(node: ast.AST) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for name in ('body', 'orelse', 'finalbody'):
        value = getattr(node, name, None)
        if value and isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            bodies.append(value)
    if isinstance(node, ast.Try):
        for handler in node.handlers:
            bodies.append(handler.body)
    return bodies


def import_time_environ() -> list[str]:
    """No ``os.environ`` writes while a host or Study module is importing."""
    found: list[str] = []
    for path in _py_files(ROOT / 'src' / 'spec_mtp', ROOT / 'studies'):
        rel = _posix(path)
        if _under_upstream(rel):
            continue
        text = _read(path)
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise AssertionError(f'{rel} does not parse: {exc}') from exc
        os_names, environ_names = _environ_aliases(tree)
        for node in _import_time_nodes(tree.body):
            if not _writes_environ(node, os_names, environ_names):
                continue
            segment = ast.get_source_segment(text, node) or ''
            segment = ' '.join(segment.split())
            found.append(f'{rel} {segment}')
    return sorted(set(found))


def _load(name: str) -> list[str]:
    path = ALLOWLIST_DIR / name
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise AssertionError(f'{name} must be a JSON list of strings')
    if data != sorted(set(data)):
        raise AssertionError(f'{name} must be sorted and unique')
    return data


def _baseline(name: str) -> list[str] | None:
    rel = _posix(ALLOWLIST_DIR / name)
    for ref in ('origin/dev', 'dev'):
        try:
            base = subprocess.check_output(
                ['git', 'merge-base', 'HEAD', ref],
                cwd=ROOT, text=True, encoding='utf-8', stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        try:
            text = subprocess.check_output(
                ['git', 'show', f'{base}:{rel}'],
                cwd=ROOT, text=True, encoding='utf-8', stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return None
        data = json.loads(text)
        if not isinstance(data, list):
            raise AssertionError(f'baseline {name} is not a list')
        return [item for item in data if isinstance(item, str)]
    return None


def _raise_if_drift(name: str, actual: list[str], allowed: list[str]) -> None:
    extra = sorted(set(actual) - set(allowed))
    stale = sorted(set(allowed) - set(actual))
    if extra or stale:
        raise AssertionError(
            f'{name}: {len(extra)} new violation(s), {len(stale)} stale allowlist entry(ies)\n'
            + '\n'.join(f'+ {item}' for item in extra)
            + ('\n' if extra and stale else '')
            + '\n'.join(f'- {item}' for item in stale)
        )


def assert_frozen(name: str, actual: list[str]) -> None:
    allowed = _load(name)
    _raise_if_drift(name, actual, allowed)
    baseline = _baseline(name)
    if baseline is None:
        return
    grew = sorted(set(allowed) - set(baseline))
    if grew:
        raise AssertionError(
            name + ' grew against origin/dev; remove these entries:\n'
            + '\n'.join(grew)
        )


def test_legacy_allowlist_rejects_a_new_violation():
    with pytest.raises(AssertionError, match='new violation'):
        _raise_if_drift(
            'example',
            ['studies/legacy.py', 'studies/new_script.py'],
            ['studies/legacy.py'],
        )


def test_legacy_allowlist_rejects_a_stale_entry():
    with pytest.raises(AssertionError, match='stale'):
        _raise_if_drift('example', ['studies/legacy.py'], ['studies/legacy.py', 'studies/fixed.py'])


def test_legacy_allowlist_rejects_growth_against_dev(monkeypatch):
    monkeypatch.setattr(f'{__name__}._load', lambda name: ['kept', 'added'])
    monkeypatch.setattr(f'{__name__}._baseline', lambda name: ['kept'])
    with pytest.raises(AssertionError, match='grew'):
        assert_frozen('file_inventory.json', ['kept', 'added'])


def test_structure_flow_file_inventory_matches_legacy_allowlist():
    assert_frozen('file_inventory.json', file_inventory())


def test_layer_import_direction_matches_legacy_allowlist():
    assert_frozen('import_direction.json', import_direction())


def test_structure_has_no_unlisted_main_or_argparse():
    assert_frozen('structure_scripts.json', structure_scripts())


def test_structure_flow_has_no_unlisted_drive_letter_paths():
    assert_frozen('drive_letters.json', drive_letters())


def test_structure_flow_has_no_unlisted_import_time_environ_mutation():
    assert_frozen('import_environ.json', import_time_environ())
