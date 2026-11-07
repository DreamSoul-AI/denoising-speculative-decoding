"""Source-gated registration of the explicit paired/norm control protocol."""
from studies._shared.structure.recipe import register as native_register, register_control


def register(ctx):
    native_register(ctx)
    register_control(ctx.study_dir, admission_assets=ctx.assets_dir, kind='continuation')
