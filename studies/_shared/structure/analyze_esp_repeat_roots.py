"""Re-export shim; helpers live in esp_diagnostics."""
try:
    from esp_diagnostics import ROOT, count_repeats
except ImportError:
    from studies._shared.structure.esp_diagnostics import ROOT, count_repeats
__all__ = ['ROOT', 'count_repeats']
