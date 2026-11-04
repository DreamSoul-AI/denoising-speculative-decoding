"""Re-export shim; helpers live in window_diagnostics.

    python studies/_shared/structure/diagnose_output_windows.py
"""
try:
    from window_diagnostics import SEMANTIC_KEYS, TRACE_KEYS, margin, cosine, observe, file_hash
except ImportError:
    from studies._shared.structure.window_diagnostics import SEMANTIC_KEYS, TRACE_KEYS, margin, cosine, observe, file_hash
__all__ = ['SEMANTIC_KEYS', 'TRACE_KEYS', 'margin', 'cosine', 'observe', 'file_hash']
