"""Reuse the original five methods; add source checks and resource admission only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BASE = ROOT / 'studies/qwen3_4b_bf16_five_methods'
sys.path.insert(0, str(STUDY))
for name in ('qwen3_4b_bf16_five_methods', 'qwen3_4b_bf16_bias_norm_scan', 'esp_table1_nonofficial_revalidation'):
    sys.path.insert(0, str(ROOT / 'studies' / name))
from untimed_pairing import PairedUntimedEval, build_cuda  # noqa: E402
from paired_admission import wait_for_admission  # noqa: E402
from rpipe.structure.algorithm.factory import AlgorithmRegistry  # noqa: E402
from rpipe.structure.model.factory import ModelRegistry  # noqa: E402
from rpipe.structure.control.merge import deep_merge  # noqa: E402
from studies._shared.structure import ar_baseline
from spec_mtp import decode as decoding  # noqa: E402


def check_refs():
    refs = json.loads((STUDY / 'docs/source_refs.json').read_text(encoding='utf-8'))
    for name, expected in refs['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    plan = refs['project_plan']
    assert hashlib.sha256(subprocess.check_output(['git', 'show', plan['commit'] + ':' + plan['path']], cwd=ROOT)).hexdigest() == plan['sha256']
    return matrix_settings()


def matrix_settings():
    """Static matrix contract, independently inspectable without private Run evidence."""
    original = yaml.safe_load((BASE / 'experiment_config.yaml').read_text(encoding='utf-8'))
    original['experiment'] = STUDY.name
    original['algorithm']['config']['version'] = 'qwen3-4b-bf16-paired-independent-20261016-v1'
    assert yaml.safe_load((STUDY / 'experiment_config.yaml').read_text(encoding='utf-8')) == original
    decl = yaml.safe_load((STUDY / 'study.yaml').read_text(encoding='utf-8'))
    old = yaml.safe_load((BASE / 'study.yaml').read_text(encoding='utf-8'))
    assert decl['axes'] == old['axes'] and decl['seeds'] == old['seeds'] == [123]
    return [deep_merge(original['algorithm']['config'], setting) for setting in decl['axes']['algorithm.config']]


def admitted_build(*args, **kwargs):
    check_refs()
    assets = kwargs.get('assets_dir', args[1] if len(args) > 1 else None)
    admission = wait_for_admission(assets, initial=True)
    model = build_cuda(*args, **kwargs)
    model.meta['paired_repeat_admission'] = admission
    return model


class PairedRepeatEval(PairedUntimedEval):
    def run(self, data, model, system, tracker):
        check_refs()
        generate, ar = decoding.SpecDecoder.generate, ar_baseline.ar_generate
        def ready_generate(decoder, prompt):
            wait_for_admission(system.assets_dir, initial=False)
            return generate(decoder, prompt)
        def ready_ar(*args, **kwargs):
            wait_for_admission(system.assets_dir, initial=False)
            return ar(*args, **kwargs)
        with patch.object(decoding.SpecDecoder, 'generate', ready_generate), patch.object(ar_baseline, 'ar_generate', ready_ar):
            result = super().run(data, model, system, tracker)
        check_refs()
        return result


def register_overrides():
    ModelRegistry.register('hf_causal', 'spec_mtp', admitted_build)
    AlgorithmRegistry.register('eval', 'spec_mtp', PairedRepeatEval)
