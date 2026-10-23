"""Enforce scientific ownership, including imports inside function bodies."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ['contracts', 'data', 'decode', 'masks', 'tree', 'audit', 'analysis'])
def test_scientific_modules_do_not_import_assembly_or_framework(name):
    tree = ast.parse((ROOT / 'src/spec_mtp' / (name + '.py')).read_text(encoding='utf-8'))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or '')
    forbidden = ('rpipe', 'studies', 'tools', 'spec_mtp.cli', 'spec_mtp.model')
    assert not [module for module in imports if any(module == edge or module.startswith(edge + '.') for edge in forbidden)]


def test_package_and_help_do_not_import_torch_transformers_or_rpipe():
    code = ('import sys; import spec_mtp; from spec_mtp.cli import main; main([]); '
            'assert not {"torch", "transformers", "rpipe", "studies"}.intersection(sys.modules)')
    subprocess.run([sys.executable, '-c', code], check=True, timeout=15)
