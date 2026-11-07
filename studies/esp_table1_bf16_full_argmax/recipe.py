"""Native source admission followed by explicit ESP execution policy binding."""
from studies._shared.structure.recipe import register as native_register, register_esp


def register(ctx):
    native_register(ctx)
    register_esp(ctx.study_dir, admission_assets=ctx.assets_dir, kind='argmax')
