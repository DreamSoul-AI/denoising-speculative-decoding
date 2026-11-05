"""Nested-package Git identity is scoped, explicit and fail closed."""

from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from studies._shared.structure import sources

_SOURCE_PROPERTY_SETTINGS = settings(max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture])
_SOURCE_NAMES = st.text(st.characters(codec='ascii', categories=['L', 'N']), min_size=1, max_size=8)
_SOURCE_FILES = st.dictionaries(_SOURCE_NAMES.map(lambda value: value + '.py'),
                                st.text('0123456789abcdef', min_size=4, max_size=8), max_size=6)


@_SOURCE_PROPERTY_SETTINGS
@given(_SOURCE_FILES)
def test_digest_ignores_insertion_order(files):
    assert sources.files_digest(files) == sources.files_digest(dict(reversed(list(files.items()))))


@_SOURCE_PROPERTY_SETTINGS
@given(_SOURCE_FILES, _SOURCE_FILES)
def test_changed_files_is_the_symmetric_difference_of_entries(before, now):
    changed = sources.changed_files({'files': before}, {'files': now})
    assert changed == sorted(changed)
    assert set(changed) == {key for key in set(before) | set(now) if before.get(key) != now.get(key)}
    assert (changed == []) == (before == now)
    assert sources.changed_files({'files': now}, {'files': before}) == changed


@_SOURCE_PROPERTY_SETTINGS
@given(_SOURCE_FILES, _SOURCE_FILES)
def test_digest_changes_iff_files_change(a, b):
    assert (sources.files_digest(a) == sources.files_digest(b)) == (a == b)


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


def test_manifest_covers_package_sources():
    body = sources.manifest()
    assert {'studies/_shared/structure/recipe.py', 'studies/_shared/flow/collect/run.py',
            'studies/_shared/structure/validation.py',
            'spec_mtp/structure/algorithm/decode/loop.py'} <= set(body['files'])
    assert not any(name.startswith('rpipe/') for name in body['files'])
    assert body['digest'] == sources.files_digest(body['files'])


def test_load_lock_absent_is_none(tmp_path):
    assert sources.load_lock(tmp_path) is None


def test_load_lock_rejects_corrupt_and_non_object(tmp_path):
    path = tmp_path / sources.LOCK_NAME
    path.write_text('{not json', encoding='utf-8')
    with pytest.raises(sources.SpecMtpSourceChangedError, match='not valid JSON'):
        sources.load_lock(tmp_path)
    path.write_text('[1, 2]', encoding='utf-8')
    with pytest.raises(sources.SpecMtpSourceChangedError, match='JSON object'):
        sources.load_lock(tmp_path)
    path.write_text('{"digest": "abc"}\n', encoding='utf-8')
    assert sources.load_lock(tmp_path) == {'digest': 'abc'}
