"""The offline model setting must reach ModelScope before HF construction."""
import sys
from types import SimpleNamespace
from studies._shared.structure.model import _materialize_hub


def test_modelscope_respects_local_files_only(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(name, **kwargs):
        calls.append((name, kwargs))
        return str(tmp_path / 'snapshot')

    monkeypatch.setitem(sys.modules, 'modelscope', SimpleNamespace(snapshot_download=snapshot_download))
    result = _materialize_hub({'name_or_path': 'Qwen/Qwen3-0.6B', 'hub': 'modelscope',
                               'local_files_only': True}, tmp_path)
    assert calls == [('Qwen/Qwen3-0.6B', {'cache_dir': str(tmp_path / 'modelscope'),
                                       'local_files_only': True})]
    assert result['local_files_only'] is True and result['name_or_path'] == str(tmp_path / 'snapshot')
