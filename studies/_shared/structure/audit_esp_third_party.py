"""Re-export shim; helpers live in esp_diagnostics.

    python studies/_shared/structure/audit_esp_third_party.py
"""
try:
    from esp_diagnostics import ROOT, VENDOR, load_original, audit_step, ArgmaxRejected
except ImportError:
    from studies._shared.structure.esp_diagnostics import ROOT, VENDOR, load_original, audit_step, ArgmaxRejected
__all__ = ['ROOT', 'VENDOR', 'load_original', 'audit_step', 'ArgmaxRejected']
