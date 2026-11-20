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


def test_shared_model_consumer_uses_pool_without_study_or_repo_namespace(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    pool = root / '.local/models'
    pool.mkdir(parents=True)
    assets = root / 'studies/topic/shared/model'
    calls = []
    monkeypatch.setitem(sys.modules, 'modelscope', SimpleNamespace(
        snapshot_download=lambda name, **kw: calls.append((name, kw)) or str(pool / 'snapshot')))
    _materialize_hub({'name_or_path': 'Qwen/Qwen3-8B', 'hub': 'modelscope',
                      'local_files_only': True, 'revision': 'fixture-revision'}, assets)
    assert calls[0][1]['cache_dir'] == str(pool / 'modelscope')
    assert calls[0][1]['revision'] == 'fixture-revision'
    assert not assets.exists()


def test_adopted_layout_uses_data_and_models_pools(tmp_path):
    from spec_mtp.structure.artifact.layout import artifact_layout
    root = tmp_path / 'repo'
    for name in ('data', 'models'):
        (root / '.local' / name).mkdir(parents=True)
    study = root / 'studies/topic'
    layout = artifact_layout(study, 'run1')
    assert layout.shared_data_dir == root / '.local/data'
    assert layout.shared_model_dir == root / '.local/models'
    assert not (study / 'shared').exists()


def test_shared_data_does_not_overwrite_different_identity(tmp_path):
    import pytest
    from studies._shared.structure.data import _land_specbench
    root = tmp_path / 'repo'
    dest = root / '.local/data/specbench/question.jsonl'
    dest.parent.mkdir(parents=True)
    dest.write_text('old', encoding='utf-8')
    src = tmp_path / 'source.jsonl'
    src.write_text('new', encoding='utf-8')
    with pytest.raises(ValueError, match='identity conflict'):
        _land_specbench({'path': str(src)}, root / 'studies/topic/shared/data')
    assert dest.read_text(encoding='utf-8') == 'old'
