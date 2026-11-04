"""Re-export shim; helpers live in q178_diagnostics."""
try:
    from q178_diagnostics import delta, phi_factors, rms_factors
except ImportError:
    from studies._shared.structure.q178_diagnostics import delta, phi_factors, rms_factors
__all__ = ['delta', 'phi_factors', 'rms_factors']
