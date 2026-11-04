"""Operational continuation only; reuse the frozen norm science unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
for folder in ('qwen3_4b_bf16_bias_norm_scan', 'qwen3_4b_bf16_paired_revalidation'):
    sys.path.insert(0, str(ROOT / 'studies' / folder))
import norm_control
from paired_admission import wait_for_admission

def check_refs():
    refs = json.loads((STUDY / 'docs/source_refs.json').read_text(encoding='utf-8'))
    for name, expected in refs['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    p = refs['project_plan']
    assert hashlib.sha256(subprocess.check_output(['git', 'show', p['commit'] + ':' + p['path']], cwd=ROOT)).hexdigest() == p['sha256']
    norm_control.check_refs()

def admitted_build(*args, **kwargs):
    check_refs()
    assets = kwargs.get('assets_dir', args[1] if len(args) > 1 else None)
    admission = wait_for_admission(assets, initial=True)
    model = norm_control.build_cuda(*args, **kwargs)
    model.meta['norm_scan_admission'] = admission
    return model

class ContinuationEval(norm_control.NormEval):
    def run(self, data, model, system, tracker):
        check_refs()
        with patch.object(norm_control, 'wait_for_admission', wait_for_admission):
            result = super().run(data, model, system, tracker)
        check_refs()
        return result

def register_overrides():
    norm_control.ModelRegistry.register('hf_causal', 'spec_mtp', admitted_build)
    norm_control.AlgorithmRegistry.register('eval', 'spec_mtp', ContinuationEval)
