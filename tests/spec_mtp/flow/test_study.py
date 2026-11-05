"""CPU dispatch spies preserve the actual host Study phase callable boundary."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('name', ['qwen3_4b_bf16_paired_revalidation', 'qwen3_4b_bf16_bias_norm_continuation'])
@pytest.mark.parametrize('phase', ['collect', 'summarize', 'write', 'process'])
def test_native_phase_dispatches_exported_callable(monkeypatch, name, phase):
    from spec_mtp.flow.study import run_study_phase
    native = importlib.import_module('studies._shared.flow.' + phase)
    calls = []
    monkeypatch.setattr(native, 'run', lambda ctx: calls.append(ctx))
    folder = ROOT / 'studies' / name
    decl = yaml.safe_load((folder / 'study.yaml').read_text(encoding='utf-8'))
    ctx = SimpleNamespace()
    assert run_study_phase(folder, phase, ctx, decl)
    assert calls == [ctx]
