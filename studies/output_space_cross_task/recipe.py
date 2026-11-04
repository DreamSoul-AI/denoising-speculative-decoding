"""Register sources without constructing inputs, loading weights or decoding."""

from studies._shared.structure.recipe import register, register_sources

register_sources()

__all__ = ['register']
