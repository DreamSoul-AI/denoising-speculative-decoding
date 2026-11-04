"""Re-export shim; helpers live in q178_diagnostics.

    python studies/_shared/structure/diagnose_q178_qproj.py
"""
try:
    from q178_diagnostics import linear_row, run__diagnose_q178_qproj as run, exact_block
except ImportError:
    from studies._shared.structure.q178_diagnostics import linear_row, run__diagnose_q178_qproj as run, exact_block
__all__ = ['linear_row', 'run', 'exact_block']
