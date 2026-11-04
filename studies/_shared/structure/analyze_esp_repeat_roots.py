"""Re-export shim; helpers live in esp_diagnostics.

    python studies/_shared/structure/analyze_esp_repeat_roots.py
"""
try:
    from esp_diagnostics import ROOT, count_repeats
except ImportError:
    from studies._shared.structure.esp_diagnostics import ROOT, count_repeats
__all__ = ['ROOT', 'count_repeats']
