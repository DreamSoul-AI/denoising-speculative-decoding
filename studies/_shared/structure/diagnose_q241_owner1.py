"""Re-export shim; helpers live in q241_diagnostics."""
try:
    from q241_diagnostics import capture_frames__diagnose_q241_owner1 as capture_frames
except ImportError:
    from studies._shared.structure.q241_diagnostics import capture_frames__diagnose_q241_owner1 as capture_frames
__all__ = ['capture_frames']
