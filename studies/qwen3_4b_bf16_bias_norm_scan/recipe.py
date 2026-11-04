"""Native recipes plus fixed-direction and native-step-scale observation."""
from studies._shared.structure.recipe import register as native_register, register_sources
from norm_control import register_overrides

register_sources()
register_overrides()


def register(ctx):
    native_register(ctx)
    register_overrides()
