"""Re-export shim; helpers live in q178_diagnostics.

    python studies/_shared/structure/diagnose_q178_phi_rms.py
"""
try:
    from q178_diagnostics import phi64, norm_row, run__diagnose_q178_phi_rms as run
except ImportError:
    from studies._shared.structure.q178_diagnostics import phi64, norm_row, run__diagnose_q178_phi_rms as run
__all__ = ['phi64', 'norm_row', 'run']
