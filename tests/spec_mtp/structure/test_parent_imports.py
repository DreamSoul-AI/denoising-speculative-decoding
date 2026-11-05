"""CPU fresh-process import boundaries; never load tensors, runtime factories or a model."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

PARENTS=['spec_mtp.structure','spec_mtp.structure.control','spec_mtp.structure.algorithm','spec_mtp.structure.data',
         'spec_mtp.structure.model','spec_mtp.structure.system','spec_mtp.structure.artifact','spec_mtp.structure.make']
LEAVES=['spec_mtp.structure.control.hashing','spec_mtp.structure.control.mapping','spec_mtp.structure.control.run_config',
        'spec_mtp.structure.control.contract','spec_mtp.structure.algorithm.config','spec_mtp.structure.data.config',
        'spec_mtp.structure.model.config','spec_mtp.structure.system.config','spec_mtp.structure.artifact.config.io',
        'spec_mtp.structure.artifact.index','spec_mtp.structure.make.expand','spec_mtp.structure.make.write']
ROOT=Path(__file__).resolve().parents[3]
BARRIER="""
import importlib.abc
import importlib
import sys
BLOCK_FLOW=True
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'torch','torchvision','transformers','numpy','studies'}
                or (BLOCK_FLOW and fullname.startswith('spec_mtp.flow')) or fullname.endswith('.factory')):
            raise AssertionError('unexpected runtime dependency '+fullname)
sys.meta_path.insert(0,RejectRuntime())
"""


def child(script):
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1',
                 'PYTHONPATH':os.pathsep.join((str(ROOT/'src'),str(ROOT)))}
    result=subprocess.run([sys.executable,'-B','-c',BARRIER+script],cwd=ROOT,env=environment,
                          capture_output=True,text=True,encoding='utf-8',timeout=20)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('module',PARENTS)
def test_responsibility_parent_imports_do_not_assemble_or_export_runtime(module):
    child('module=importlib.import_module('+repr(module)+')\n'
          'assert not [name for name in vars(module) if not name.startswith("_")]\n'
          'assert "__getattr__" not in vars(module)\n')


@pytest.mark.parametrize('module',LEAVES)
def test_configuration_and_artifact_leaf_imports_remain_framework_free(module):
    child('importlib.import_module('+repr(module)+')\n')


def test_scientific_help_does_not_import_runtime_after_parent_cleanup():
    child('BLOCK_FLOW=False\nimport runpy\nsys.argv=["spec_mtp","--help"]\n'
          'try: runpy.run_module("spec_mtp",run_name="__main__")\n'
          'except SystemExit as error: assert error.code==0\n')


def test_package_initializers_do_not_define_business_functions():
    import ast

    package = ROOT / 'src/spec_mtp'
    violations = []
    for path in package.rglob('__init__.py'):
        rel = path.relative_to(package).as_posix()
        if rel.startswith(('flow/', 'structure/')):
            continue
        body = ast.parse(path.read_text(encoding='utf-8'))
        if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in body.body):
            violations.append(path.relative_to(package).as_posix())
    assert not violations, violations
