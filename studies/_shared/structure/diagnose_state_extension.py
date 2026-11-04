"""Re-export shim; helpers live in extension_diagnostics."""
try:
    from extension_diagnostics import frozen_targets
except ImportError:
    from studies._shared.structure.extension_diagnostics import frozen_targets
__all__ = ['frozen_targets']
