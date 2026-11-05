"""Project documents are reachable and actual shared flow commands document their usage."""

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_every_doc_is_linked_from_agents():
    agents = (ROOT / 'AGENTS.md').read_text(encoding='utf-8')
    missing = [p.name for p in sorted((ROOT / 'docs').glob('*.md')) if f'docs/{p.name}' not in agents]
    assert not missing, f'add to AGENTS.md §2: {missing}'


def test_scientific_modules_match_the_design():
    expected = {'__init__', '__main__', 'cli'}
    assert {p.stem for p in (ROOT / 'src/spec_mtp').glob('*.py')} == expected


def test_every_tool_has_a_usage_docstring():
    root = ROOT / 'studies' / '_shared' / 'flow'
    scripts = [path for path in sorted(root.glob('*.py'))
               if re.search(r'''if\s+__name__\s*==\s*['"]__main__['"]''', path.read_text(encoding='utf-8'))]
    assert scripts
    for path in scripts:
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding='utf-8')))
        usages = (f'python -m studies._shared.flow.{path.stem}', f'python studies/_shared/flow/{path.name}')
        assert docstring and any(usage in docstring for usage in usages), path.name
