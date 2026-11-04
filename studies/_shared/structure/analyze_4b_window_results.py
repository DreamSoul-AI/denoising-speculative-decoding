"""Re-export shim; helpers live in geometry_4b."""
try:
    from geometry_4b import rank_interval, coverage, plot_results
except ImportError:
    from studies._shared.structure.geometry_4b import rank_interval, coverage, plot_results
__all__ = ['rank_interval', 'coverage', 'plot_results']
