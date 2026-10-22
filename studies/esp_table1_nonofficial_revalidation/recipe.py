"""Register existing sources and the new read-only resource/input guards."""
from studies._shared.structure.recipe import register as native_register, register_sources
from revalidation_contract import RevalidationEval, admitted_build
from rpipe.structure.algorithm.factory import AlgorithmRegistry
from rpipe.structure.model.factory import ModelRegistry


def register_overrides():
    ModelRegistry.register('hf_causal', 'spec_mtp', admitted_build)
    AlgorithmRegistry.register('eval', 'spec_mtp', RevalidationEval)


register_sources()
register_overrides()


def register(ctx):
    native_register(ctx)
    register_overrides()
