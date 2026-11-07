"""Optional result reads preserve the strict reader and never rewrite artifacts."""

import json

import pytest


@pytest.mark.parametrize('case', ['missing', 'directory', 'valid', 'malformed', 'nonmapping',
                                 'os_error', 'type_error', 'value_error', 'corrupt_error', 'runtime_error'])
def test_optional_result_preserves_read_failures_and_never_writes(tmp_path, monkeypatch, case):
    from spec_mtp.structure.artifact.errors import CorruptArtifactError
    from spec_mtp.structure.artifact.result import io
    from spec_mtp.structure.artifact.result.io import load_optional_result

    path = tmp_path / 'result.json'
    body = {'status': 'failed', 'error': 'saved failure'}
    failures = {'os_error': OSError, 'type_error': TypeError, 'value_error': ValueError,
                'corrupt_error': CorruptArtifactError, 'runtime_error': RuntimeError}
    error = None
    if case == 'directory':
        path.mkdir()
    elif case != 'missing':
        path.write_text('{' if case == 'malformed' else '[]' if case == 'nonmapping'
                        else json.dumps(body), encoding='utf-8')
    if case in failures:
        error = failures[case]('read failure')

        def fail_read(_):
            raise error

        monkeypatch.setattr(io, 'load_result', fail_read)
    before = {item.relative_to(tmp_path): None if item.is_dir() else item.read_bytes()
              for item in tmp_path.rglob('*')}
    if case in ('nonmapping', 'corrupt_error', 'runtime_error'):
        with pytest.raises(RuntimeError if case == 'runtime_error' else CorruptArtifactError):
            load_optional_result(path)
    else:
        assert load_optional_result(path) == (body if case == 'valid' else None)
    if case == 'valid':
        assert io.load_result(path) == body
    else:
        strict_error = (failures[case] if case in failures else
                        CorruptArtifactError if case == 'nonmapping' else
                        ValueError if case == 'malformed' else OSError)
        with pytest.raises(strict_error):
            io.load_result(path)
    assert before == {item.relative_to(tmp_path): None if item.is_dir() else item.read_bytes()
                      for item in tmp_path.rglob('*')}
