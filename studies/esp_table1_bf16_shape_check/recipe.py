"""ESP-only local backend controls, registered on the native Run path."""
import os
from studies._shared.structure.recipe import register as native_register, register_untimed
from studies._shared.structure.esp_controls import register_controls

def register(ctx):
    native_register(ctx)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    register_untimed(ctx.study_dir)
    register_controls(shape=True)
