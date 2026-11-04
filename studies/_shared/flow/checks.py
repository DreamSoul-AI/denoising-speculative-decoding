"""Compatibility exports; phase orchestration lives in flow/<phase>/run.py."""
from studies._shared.flow.process.run import study_gate
from studies._shared.flow.summarize.run import run_gate
from studies._shared.structure.validation import (
    GATE_KEY, STUDY_CHECK_NAME, SpecMtpGateError, declared_factors, evaluate_run, verify_evidence,
)

__all__ = ['GATE_KEY', 'STUDY_CHECK_NAME', 'SpecMtpGateError', 'declared_factors',
           'evaluate_run', 'verify_evidence', 'run_gate', 'study_gate']
