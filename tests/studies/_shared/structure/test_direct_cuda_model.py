"""Direct CUDA placement forwards loader settings without materializing any model."""
import pytest

from studies._shared.structure import model


@pytest.mark.parametrize('failure', [False, True])
def test_direct_cuda_materialization_preserves_kwargs_and_restores_loader(monkeypatch, failure):
    from transformers import AutoModelForCausalLM
    seen = {}
    data_meta = {'synthetic': True}
    def original(*args, **kwargs):
        seen.update(kwargs)
        if failure:
            raise RuntimeError('injected loader failure')
        return 'model'
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', original)
    def factory(config, assets, actual_meta):
        assert (config, assets, actual_meta) == ('configuration', 'assets', data_meta)
        return AutoModelForCausalLM.from_pretrained('cache', dtype='synthetic-dtype', local_files_only=True)
    monkeypatch.setattr(model, '_build_hf', factory)
    if failure:
        with pytest.raises(RuntimeError, match='loader failure'):
            model.build_cuda('configuration', 'assets', data_meta)
    else:
        assert model.build_cuda('configuration', 'assets', data_meta) == 'model'
    assert seen == {'dtype': 'synthetic-dtype', 'local_files_only': True, 'device_map': 'cuda'}
    assert AutoModelForCausalLM.from_pretrained is original
