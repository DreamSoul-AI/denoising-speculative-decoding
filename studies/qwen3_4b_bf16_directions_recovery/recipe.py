"""Native source admission precedes explicit direction/pairing root registration."""
from studies._shared.structure.recipe import register as native_register, register_direction


def register(ctx):
    native_register(ctx)
    register_direction(ctx.study_dir, admission_assets=ctx.assets_dir, recovery=True)
