"""Native sources plus this Study's untimed tied-BF16 adapters."""
from studies._shared.structure.recipe import register as native_register, register_sources
from untimed_pairing import register_overrides
register_sources()
register_overrides()


def register(ctx):
    native_register(ctx)
    register_overrides()
