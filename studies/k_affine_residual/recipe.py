"""Register this Study's construction; no extraction or downloads at import."""
import importlib.util
import sys
from pathlib import Path

from rpipe.structure.algorithm.factory import AlgorithmRegistry
from rpipe.structure.data.factory import DataRegistry
from spec_mtp.structure.model.registry import register_models

path = Path(__file__).parent / 'structure/affine.py'
spec = importlib.util.spec_from_file_location('k_affine_residual_study', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
register_models()
DataRegistry.register('cached_documents', 'k_affine_residual', module.build_data)
AlgorithmRegistry.register('eval', 'k_affine_residual', module.AffineAlgorithm)


def register(ctx):
    module.prepare_binding(ctx)
