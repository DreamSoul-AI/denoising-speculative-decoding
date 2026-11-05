"""Explicit placement and local-cache loader settings without models or downloads."""
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


def test_modelscope_respects_local_files_only(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    calls = []

    def snapshot_download(name, **kwargs):
        calls.append((name, kwargs))
        return str(tmp_path / 'snapshot')

    monkeypatch.setitem(sys.modules, 'modelscope', SimpleNamespace(snapshot_download=snapshot_download))
    result = model._materialize_hub({'name_or_path': 'Qwen/Qwen3-0.6B', 'hub': 'modelscope',
                               'local_files_only': True}, tmp_path)
    assert calls == [('Qwen/Qwen3-0.6B', {'cache_dir': str(tmp_path / 'modelscope'),
                                       'local_files_only': True})]
    assert result['local_files_only'] is True and result['name_or_path'] == str(tmp_path / 'snapshot')
