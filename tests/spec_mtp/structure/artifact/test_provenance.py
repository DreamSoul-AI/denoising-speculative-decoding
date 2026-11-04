"""Actual-file hashing works without Python 3.11's file_digest API."""
import hashlib
from importlib import import_module

import pytest


@pytest.mark.parametrize('module,name', [
    ('spec_mtp.structure.artifact.provenance', 'file_sha256'),
    ('studies._shared.structure.evidence', 'file_hash'),
    ('studies._shared.structure.window_diagnostics', 'file_hash'),
    ('studies._shared.structure.geometry_4b', 'sha'),
    ('studies.output_space_small_matrix.analyze_transfers', 'sha'),
    ('studies.output_space_followup.analyze_results', 'sha'),
])
def test_file_hash_supports_python310_and_real_bytes(tmp_path, monkeypatch, module, name):
    payload = b'actual\x00content' * 100000
    path = tmp_path / 'source.bin'
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    actual_hash = getattr(import_module(module), name)
    assert actual_hash(path) == expected
    with pytest.raises(FileNotFoundError):
        actual_hash(tmp_path / 'missing.bin')


PINNED_MEMBER = 'src/spec_mtp/cli.py'
PINNED_SHA256 = 'f80473ecdb6cbdd6f31f82b5479319339b1c48d984ea121e45183fe727cb769a'
PINNED_BYTES = (b'"""Stable console entry; command implementation lives in flow."""\n'
                b'from spec_mtp.flow.cli import main\n\n__all__ = ["main"]\n')


@pytest.fixture
def historical_repo(tmp_path):
    import os
    import subprocess

    root = tmp_path / 'historical-repository'
    root.mkdir()
    environment = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    command = ['git', '-c', f'safe.directory={root}', '-c', 'user.name=Historical Byte Test',
               '-c', 'user.email=historical-bytes@example.invalid', '-c', 'commit.gpgSign=false',
               '-C', str(root)]

    def git(*arguments, payload=None):
        return subprocess.check_output([*command, *arguments], input=payload, env=environment,
                                       stderr=subprocess.DEVNULL, timeout=20).strip().decode('ascii')

    git('init', '--quiet', '--object-format=sha1', '--template=')
    blob = git('hash-object', '-w', '--stdin', payload=PINNED_BYTES)
    git('update-index', '--add', '--cacheinfo', f'100644,{blob},{PINNED_MEMBER}')
    tree = git('write-tree')
    commit = git('commit-tree', tree, '-m', 'Local historical source fixture')
    (root / 'src').mkdir()
    return root, commit, blob


def test_historical_bytes_preserve_pinned_git_member_without_worktree_reads(tmp_path, monkeypatch, historical_repo):
    from pathlib import Path
    from spec_mtp.structure.artifact.provenance import read_historical_bytes

    root, commit, _ = historical_repo
    monkeypatch.chdir(tmp_path)
    def deny_current_file(*args, **kwargs):
        raise AssertionError('historical read opened a working-tree file')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_bytes', deny_current_file)
        assert read_historical_bytes(root, commit, PINNED_MEMBER, PINNED_SHA256) == PINNED_BYTES


@pytest.mark.parametrize('commit,member,sha', [
    ('HEAD', PINNED_MEMBER, PINNED_SHA256),
    ('0' * 12, PINNED_MEMBER, PINNED_SHA256),
    (None, '../cli.py', PINNED_SHA256),
    (None, 'sub/../cli.py', PINNED_SHA256),
    (None, '/outside.py', PINNED_SHA256),
    (None, 'C:/outside.py', PINNED_SHA256),
    (None, '', PINNED_SHA256),
    (None, PINNED_MEMBER, 'bad-hash'),
])
def test_historical_bytes_reject_unpinned_or_escaping_requests(historical_repo, commit, member, sha):
    from spec_mtp.structure.artifact.provenance import read_historical_bytes

    root, pinned_commit, _ = historical_repo
    with pytest.raises(ValueError):
        read_historical_bytes(root, commit or pinned_commit, member, sha)


@pytest.mark.parametrize('damage', ['missing_commit', 'missing_member', 'wrong_digest', 'non_commit'])
def test_historical_bytes_fail_closed_on_missing_or_mismatched_evidence(historical_repo, damage):
    import subprocess
    from spec_mtp.structure.artifact.provenance import read_historical_bytes

    root, commit, blob = historical_repo
    member, sha = PINNED_MEMBER, PINNED_SHA256
    error = subprocess.CalledProcessError
    if damage == 'missing_commit':
        commit = '0' * 40
    elif damage == 'missing_member':
        member = 'no-such-historical-member'
    elif damage == 'wrong_digest':
        sha, error = '0' * 64, ValueError
    else:
        commit = blob
        error = ValueError
    with pytest.raises(error):
        read_historical_bytes(root, commit, member, sha)


def test_historical_bytes_reject_oversized_member_before_blob_read(monkeypatch, historical_repo):
    import subprocess
    from spec_mtp.structure.artifact.provenance import read_historical_bytes

    root, commit, _ = historical_repo
    original = subprocess.check_output
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        return b'16777217' if '-s' in command else original(command, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, 'check_output', output)
        with pytest.raises(ValueError, match='16 MiB'):
            read_historical_bytes(root, commit, PINNED_MEMBER, PINNED_SHA256)
    assert not any('blob' in command for command in calls)
    assert all('--no-lazy-fetch' in command and '--no-replace-objects' in command for command in calls)


@pytest.mark.parametrize('root_kind', ['subdirectory', 'not_a_repository'])
def test_historical_bytes_require_the_explicit_repository_root(tmp_path, historical_repo, root_kind):
    import subprocess
    from spec_mtp.structure.artifact.provenance import read_historical_bytes

    root, commit, _ = historical_repo
    candidate = root / 'src' if root_kind == 'subdirectory' else tmp_path
    error = ValueError if root_kind == 'subdirectory' else subprocess.CalledProcessError
    with pytest.raises(error):
        read_historical_bytes(candidate, commit, PINNED_MEMBER, PINNED_SHA256)
