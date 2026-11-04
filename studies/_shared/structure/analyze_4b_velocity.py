"""Re-export shim; helpers live in geometry_4b."""
try:
    from geometry_4b import velocity_geometry, transition_counts, template_metrics
except ImportError:
    from studies._shared.structure.geometry_4b import velocity_geometry, transition_counts, template_metrics
__all__ = ['velocity_geometry', 'transition_counts', 'template_metrics']
