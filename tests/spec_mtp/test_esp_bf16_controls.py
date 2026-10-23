"""BF16 reproduction controls restore backend state even if native decoding fails."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('reduced,deterministic', [(True,False),(False,False),(True,True),(False,True)])
@pytest.mark.parametrize('failure', [False,True])
def test_bf16_controls_restore_and_record(monkeypatch, tmp_path, reduced, deterministic, failure):
    monkeypatch.syspath_prepend(str(ROOT/'studies/esp_table1_bf16_numerics'))
    import numerics
    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    old_det = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    def run(*args):
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is reduced
        assert torch.are_deterministic_algorithms_enabled() is deterministic
        if failure:
            raise RuntimeError('injected native failure')
        return {'ok':True}
    monkeypatch.setattr(numerics.UntimedEval, 'run', run)
    algo = numerics.NumericsEval.__new__(numerics.NumericsEval)
    controls = {'bf16_reduced_precision_reduction':reduced,'deterministic':deterministic}
    algo.config = SimpleNamespace(setting=lambda _: controls)
    args = (None,SimpleNamespace(module=SimpleNamespace(dtype=torch.bfloat16)),SimpleNamespace(assets_dir=tmp_path),None)
    if failure:
        with pytest.raises(RuntimeError,match='injected native failure'):
            algo.run(*args)
    else:
        assert algo.run(*args)=={'ok':True}
    assert json.loads((tmp_path/'numerics_controls.json').read_text())==dict(controls,dtype='torch.bfloat16')
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is old_reduced
    assert torch.are_deterministic_algorithms_enabled() is old_det
    assert torch.is_deterministic_algorithms_warn_only_enabled() is old_warn
