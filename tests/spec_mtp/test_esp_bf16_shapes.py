"""Check the BF16 linear shape control and model-local patch cleanup."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('shape', [(1,4,16),(2,3,16),(16,),(0,16)])
def test_rowwise_bf16_is_partition_and_permutation_invariant(monkeypatch, shape):
    monkeypatch.syspath_prepend(str(ROOT/'studies/esp_table1_bf16_shape_check'))
    import shape_control
    torch.manual_seed(123)
    layer = torch.nn.Linear(16,7,bias=True,dtype=torch.bfloat16)
    before = {k:v.clone() for k,v in layer.state_dict().items()}
    x = torch.randn(shape,dtype=torch.bfloat16)
    actual = shape_control.rowwise_linear(layer.forward,x)
    flat = x.reshape(-1,16)
    if len(flat):
        perm = torch.arange(len(flat)-1,-1,-1)
        shuffled = shape_control.rowwise_linear(layer.forward,flat[perm])[perm]
        assert torch.equal(actual.reshape(-1,7),shuffled)
        prefix = shape_control.rowwise_linear(layer.forward,flat[:1])
        assert torch.equal(actual.reshape(-1,7)[:1],prefix)
    assert actual.dtype==torch.bfloat16 and actual.shape==(*shape[:-1],7)
    assert all(torch.equal(layer.state_dict()[k],v) for k,v in before.items())

@pytest.mark.parametrize('enabled', [False,True])
@pytest.mark.parametrize('failure', [False,True])
def test_shape_patch_is_local_and_restores_on_failure(monkeypatch,tmp_path,enabled,failure):
    monkeypatch.syspath_prepend(str(ROOT/'studies/esp_table1_bf16_shape_check'))
    import shape_control
    net = torch.nn.Sequential(torch.nn.Linear(16,8,dtype=torch.bfloat16),torch.nn.SiLU(),torch.nn.Linear(8,4,dtype=torch.bfloat16))
    layers = [m for m in net.modules() if isinstance(m,torch.nn.Linear)]
    before = {k:v.clone() for k,v in net.state_dict().items()}
    def parent(*args):
        assert all(('forward' in m.__dict__)==enabled for m in layers)
        assert net(torch.ones(1,2,16,dtype=torch.bfloat16)).dtype==torch.bfloat16
        if failure:
            raise RuntimeError('injected shape failure')
        return {'ok':True}
    monkeypatch.setattr(shape_control.NumericsEval,'run',parent)
    algo = shape_control.ShapeEval.__new__(shape_control.ShapeEval)
    algo.config = SimpleNamespace(setting=lambda _: {'linear_rows':enabled})
    args = (None,SimpleNamespace(module=SimpleNamespace(hf=net,dtype=torch.bfloat16)),SimpleNamespace(assets_dir=tmp_path),None)
    if failure:
        with pytest.raises(RuntimeError,match='injected shape failure'):
            algo.run(*args)
    else:
        assert algo.run(*args)=={'ok':True}
    assert all('forward' not in m.__dict__ for m in layers)
    assert all(torch.equal(net.state_dict()[k],v) for k,v in before.items())
