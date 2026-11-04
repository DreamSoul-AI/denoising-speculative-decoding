"""Register the SpecMTP sources for this Study."""

from studies._shared.structure.recipe import register, register_sources

register_sources()

__all__ = ['register']
