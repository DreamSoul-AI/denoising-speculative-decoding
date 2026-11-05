"""Native registrations plus only this Study's untimed/direct-placement adapters."""
from studies._shared.structure.recipe import register as native_register, register_untimed


def register(ctx):
    native_register(ctx)
    register_untimed(ctx.study_dir)
