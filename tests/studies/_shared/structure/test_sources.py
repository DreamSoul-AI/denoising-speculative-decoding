"""Nested-package Git identity is scoped, explicit and fail closed."""

from types import SimpleNamespace

import pytest

from studies._shared.structure import sources


@pytest.mark.parametrize('fault', ['none', 'head', 'status', 'no_checkout'])
def test_source_git_identity_trusts_checkout_and_rejects_unknown_status(tmp_path, monkeypatch, fault):
    nested = tmp_path / 'src' / 'spec_mtp'
    nested.mkdir(parents=True)
    if fault != 'no_checkout':
        (tmp_path / '.git').mkdir()
    calls = []
    def run(command, **options):
        calls.append(command)
        head = command[-2:] == ['rev-parse', 'HEAD']
        return SimpleNamespace(returncode=int((head and fault == 'head') or (not head and fault == 'status')),
                               stdout='a' * 40 if head else '')
    monkeypatch.setattr(sources.subprocess, 'run', run)
    result = sources._git(nested)
    assert result == ({'head': 'a' * 40, 'package_dirty': False} if fault == 'none' else {})
    if fault == 'no_checkout':
        assert not calls
    else:
        assert all(command[:5] == ['git', '-c', 'safe.directory=' + str(tmp_path), '-C', str(nested)] for command in calls)
        if fault != 'head':
            assert calls[-1][-4:] == ['status', '--porcelain', '--', '.']
