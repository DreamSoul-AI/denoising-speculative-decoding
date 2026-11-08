"""Native source admission precedes explicit paired-BF16 registration."""
from studies._shared.structure.recipe import register as native_register, register_pairing


def register(ctx):
    native_register(ctx)
    register_pairing(ctx.study_dir)
