"""ESP-only local backend controls, registered on the native Run path."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from studies._shared.structure.recipe import register as native_register, register_sources
from untimed_4b import register_overrides
from numerics import register_controls
from shape_control import register_shape
register_sources()
register_overrides()
register_controls()
register_shape()

def register(ctx):
    native_register(ctx)
    register_overrides()
    register_controls()
    register_shape()
