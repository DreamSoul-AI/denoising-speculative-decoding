"""Re-export shim; helpers live in q178_diagnostics.

    python studies/_shared/structure/diagnose_q178_norm_kernel.py
"""
try:
    from q178_diagnostics import NATIVE, mapped, run
except ImportError:
    from studies._shared.structure.q178_diagnostics import NATIVE, mapped, run
__all__ = ['NATIVE', 'mapped', 'run']
