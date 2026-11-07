"""Persisted YAML config rejects non-mappings before replacing saved input."""
import pytest

from spec_mtp.structure.artifact.config.format import decode_config
from spec_mtp.structure.artifact.config.io import load_config, write_config
from spec_mtp.structure.artifact.errors import CorruptArtifactError


@pytest.mark.parametrize('text', ['false', '0', '[]', "''", '[1]', 'true'])
def test_yaml_non_mapping_is_never_an_empty_config(text):
    with pytest.raises(CorruptArtifactError, match='mapping'):
        decode_config(text)


@pytest.mark.parametrize('text', ['', '# intentionally empty\n', 'null', '{}'])
def test_empty_documents_and_explicit_mappings_keep_the_existing_contract(text):
    assert decode_config(text) == {}
    assert decode_config('algorithm:\n  config:\n    enabled: false\n    budget: 0\n') == {
        'algorithm': {'config': {'enabled': False, 'budget': 0}}}


@pytest.mark.parametrize('value', [False, 0, [], None])
def test_invalid_config_write_preserves_existing_file_and_creates_no_directory(tmp_path, value):
    target = tmp_path / 'old.yaml'
    target.write_bytes(b'algorithm: {config: {budget: 0}}\n')
    with pytest.raises(CorruptArtifactError, match='mapping'):
        write_config(target, value)
    assert target.read_bytes() == b'algorithm: {config: {budget: 0}}\n'
    missing = tmp_path / 'absent' / 'config.yaml'
    with pytest.raises(CorruptArtifactError, match='mapping'):
        write_config(missing, value)
    assert not missing.parent.exists()
    assert load_config(target)['algorithm']['config']['budget'] == 0
