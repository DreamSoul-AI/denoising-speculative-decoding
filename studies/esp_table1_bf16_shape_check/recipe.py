"""ESP-only local backend controls, registered on the native Run path."""
import os
from studies._shared.structure.recipe import register as native_register, register_untimed
from studies.esp_table1_bf16_shape_check.numerics import register_controls
from studies.esp_table1_bf16_shape_check.shape_control import register_shape

def register(ctx):
    native_register(ctx)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    register_untimed(ctx.study_dir)
    register_controls()
    register_shape()
