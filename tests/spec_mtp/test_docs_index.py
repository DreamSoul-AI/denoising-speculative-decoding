"""Check that canonical documents are reachable and scientific entry points match their design."""

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r'\]\((?:<([^>]+)>|([^\s)]+))\)')


def test_every_doc_is_linked_from_agents():
    docs = (ROOT / 'docs').resolve()
    canonical = set(docs.glob('*.md')) | set((docs / '调研 Search').glob('*/*.md'))
    pending = [ROOT / 'AGENTS.md']
    visited = set()
    while pending:
        page = pending.pop()
        if page in visited:
            continue
        visited.add(page)
        for angle, plain in LINK.findall(page.read_text(encoding='utf-8')):
            href = unquote((angle or plain).partition('#')[0])
            if not href or '://' in href:
                continue
            target = (page.parent / href).resolve()
            if target.is_relative_to(docs) and target.suffix == '.md' and target.is_file():
                pending.append(target)

    missing = sorted(path.relative_to(docs).as_posix() for path in canonical - visited)
    assert not missing, f'add to the documentation navigation linked from AGENTS.md: {missing}'


def test_scientific_modules_match_the_design():
    expected = {'__init__', '__main__', 'cli', 'data', 'model', 'audit', 'analysis'}
    assert {p.stem for p in (ROOT / 'src/spec_mtp').glob('*.py')} == expected


def test_every_tool_has_a_usage_docstring():
    root = ROOT / 'studies' / '_shared' / 'structure'
    scripts = [path for path in sorted(root.glob('*.py')) if path.name.startswith((
        'diagnose_', 'analyze_', 'audit_', 'capture_', 'profile_', 'decode_', 'gpu_', 'reconcile_'))]
    assert scripts
    for path in scripts:
        head = path.read_text(encoding='utf-8').lstrip()
        assert head.startswith('"""') and 'python studies/_shared/structure' in head.split('"""')[1].replace('\\\\', '/'), path.name
