"""Re-export shim; helpers live in q241_diagnostics."""
try:
    from q241_diagnostics import capture_frames, fixed_masks, replay_fixed_block
except ImportError:
    from studies._shared.structure.q241_diagnostics import capture_frames, fixed_masks, replay_fixed_block
__all__ = ['capture_frames', 'fixed_masks', 'replay_fixed_block']
