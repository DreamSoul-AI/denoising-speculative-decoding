"""Re-export shim; helpers live in geometry_4b.

    python studies/_shared/structure/analyze_4b_geometry.py
"""
try:
    from geometry_4b import POSITION_EDGES, sha, distribution, trace_metrics, summarize_geometry, summarize_cell, audit_saved
except ImportError:
    from studies._shared.structure.geometry_4b import POSITION_EDGES, sha, distribution, trace_metrics, summarize_geometry, summarize_cell, audit_saved
__all__ = ['POSITION_EDGES', 'sha', 'distribution', 'trace_metrics', 'summarize_geometry', 'summarize_cell', 'audit_saved']
