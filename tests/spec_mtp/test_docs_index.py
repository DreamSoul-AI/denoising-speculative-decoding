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
    for path in sorted((ROOT / 'tools').glob('*.py')):
        head = path.read_text(encoding='utf-8').lstrip()
        assert head.startswith('"""') and 'python tools' in head.split('"""')[1].replace('\\\\', '/'), path.name
