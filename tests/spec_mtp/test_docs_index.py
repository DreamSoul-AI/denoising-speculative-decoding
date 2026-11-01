"""Every doc and tool is reachable: docs/*.md linked from AGENTS.md, tools/*.py named in layout.md."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_doc_is_linked_from_agents():
    agents = (ROOT / 'AGENTS.md').read_text(encoding='utf-8')
    missing = [p.name for p in sorted((ROOT / 'docs').glob('*.md')) if f'docs/{p.name}' not in agents]
    assert not missing, f'add to AGENTS.md §2: {missing}'


def test_scientific_modules_match_the_design():
    expected = {'__init__', '__main__', 'cli', 'contracts', 'data', 'model', 'decode', 'masks', 'tree', 'audit', 'analysis'}
    assert {p.stem for p in (ROOT / 'src/spec_mtp').glob('*.py')} == expected


def test_every_tool_has_a_usage_docstring():
    root = ROOT / 'studies' / '_shared' / 'structure'
    scripts = [path for path in sorted(root.glob('*.py')) if path.name.startswith((
        'diagnose_', 'analyze_', 'audit_', 'capture_', 'profile_', 'decode_', 'gpu_', 'reconcile_'))]
    assert scripts
    for path in scripts:
        head = path.read_text(encoding='utf-8').lstrip()
        assert head.startswith('"""') and 'python studies/_shared/structure' in head.split('"""')[1].replace('\\\\', '/'), path.name
