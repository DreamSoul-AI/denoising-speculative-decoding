"""Re-export shim; helpers live in esp_diagnostics.

    python studies/_shared/structure/analyze_esp_table1_gap.py
"""
try:
    from esp_diagnostics import ROOT, MT_CATEGORIES, FILES, summarize, paired, group_summary
except ImportError:
    from studies._shared.structure.esp_diagnostics import ROOT, MT_CATEGORIES, FILES, summarize, paired, group_summary
__all__ = ['ROOT', 'MT_CATEGORIES', 'FILES', 'summarize', 'paired', 'group_summary']
