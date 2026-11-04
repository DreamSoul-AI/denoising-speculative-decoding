"""Re-export shim; helpers live in q178_diagnostics."""
try:
    from q178_diagnostics import map_masks, NATIVE, replay, profile_norm
except ImportError:
    from studies._shared.structure.q178_diagnostics import map_masks, NATIVE, replay, profile_norm
__all__ = ['map_masks', 'NATIVE', 'replay', 'profile_norm']
