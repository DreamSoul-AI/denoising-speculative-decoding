"""Flow: prepare → execute → collect → summarize → write → process."""

import sys

# Windows: load Torch before NumPy so the OpenMP/BLAS DLLs agree. This used to
# live in the embedded RPipe package import.
if sys.platform == 'win32':
    try:
        import torch  # noqa: F401
    except ImportError:
        import numpy  # noqa: F401

from spec_mtp.flow.cli import main, run_study
from spec_mtp.flow.context import FlowContext
from spec_mtp.flow.process import run_study as process_study
from spec_mtp.flow.runner import FlowRunner, PHASES

__all__ = ['FlowContext', 'FlowRunner', 'PHASES', 'main', 'process_study', 'run_study']
