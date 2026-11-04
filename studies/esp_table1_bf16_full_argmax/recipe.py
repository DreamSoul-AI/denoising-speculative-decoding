"""Native registrations plus only this Study's untimed/direct-placement adapters."""
from studies._shared.structure.recipe import register as native_register, register_sources
from untimed_bf16 import register_overrides
register_sources()
register_overrides()


def register(ctx):
    native_register(ctx)
    register_overrides()
