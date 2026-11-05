"""Actual-file hashing works without Python 3.11's file_digest API."""
import hashlib
from importlib import import_module

import pytest


@pytest.mark.parametrize('module,name', [
    ('spec_mtp.structure.artifact.provenance', 'file_sha256'),
    ('studies._shared.structure.evidence', 'file_hash'),
    ('studies._shared.structure.window_diagnostics', 'file_hash'),
    ('studies._shared.structure.geometry_4b', 'sha'),
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


# Historical identity metadata frozen before navigation-only maintenance; no old source is imported.
NAVIGATION_PLAN_METADATA = {'esp_table1_bf16_full': 'fc0612edc512aebf46c93ceb8daa0a7a04c25c629baf58fdedae38158e047c92',
 'esp_table1_bf16_full_argmax': 'f276d6d6c32fc7f12e40bbe74530e47f1913573fdf92e579e0e7318255386521',
 'esp_table1_bf16_numerics': 'd4599b703de9e36a9666f2f5706871b362522a4935418bdf848de8140972148c',
 'esp_table1_bf16_preflight': '1bbebe8b1fcd4ded7387a090fe1b4e57518191c6019c655e29d10d41f215855b',
 'esp_table1_bf16_qwen_protocol': '3bd90a0ecbcc14e9ff468d71212f547fd84ed30081df6e87d8790c7c40bca8a3',
 'esp_table1_bf16_reduction_check': '5ecdaff3724846a9c5659412c916ce26df88331f1c4547fdc5b3853f85f6c980',
 'esp_table1_bf16_shape_check': '86a6abca92721110f092f35f5105fe02b3e71d4d315044bffc3daa117fb60626',
 'input_mask_baseline': '7612a70d2af9d0dcc45169c92202dc6b90c100ba9a485feec43be5689439fe7c',
 'output_space_4b_holdout': 'c405e13ce5c3adfb512db83ba86a9bf5187e6baa86535fb763f143a180cdf5c7',
 'output_space_4b_input_control': '1525440fc287635a10d3cb2add94cd7beffdc2b3e37733894ddd9327e78897f4',
 'output_space_4b_recheck': 'baee587b8b9d79d723f8b558f8bb6af6df2fd6508c9b2d69eb9f8ac91b781972',
 'output_space_4b_recheck_v2': '9118b7d3c5eb5874f6ac05c4d69cb41fde05e0e9b1d78d0fd10458c68785472b',
 'output_space_cross_task': 'd48104ebb3381a1044dc1b307ab9d95944850abb87c30c04b0341c1ff3758fa8',
 'output_space_extension': '2f5635d537121568d4750821ae6c2225c85a495694b54172f2d7c580c60ddc82',
 'output_space_followup': '29a914c54b5ce38ff0b873a31be9090814a68fe16a7acdf9d22bb783a88b4ec8',
 'output_space_prerequisite': '2fdc9b3637846682f308ef611c5a89f4eb2b5ac57a03897d2903141ffffd0889',
 'output_space_small_matrix': 'b7a8a69e19c8e9217e570b966345123a4d993b7bfc261a050f68b45ace712817',
 'qwen3_4b_bf16_directions': '4359f52f015d685a8a2e15e70b10e64647ce474e6f06a42fcfd30784234e91cf',
 'qwen3_4b_bf16_directions_recovery': 'adb9ceb8853859a62b33bc77b2a20e1f462dd8390eb7cf56b1ea1df555c68c62',
 'qwen3_4b_bf16_five_methods': '4ee4ed04ce26b2f37f3104fa3becf335abe7808088f5f8f18b1eab1752304218'}
PINNED_PLAN_METADATA = {'esp_table1_nonofficial_revalidation': 'd0fc97cd2358d794da3aad4b67e89bb3223a98495fe5f3bf7b3742d195337c1b',
 'qwen3_4b_bf16_bias_norm_continuation': '400daa52ff5251bc7f326c9128b80f42296601e534fb24521f3cdd34bcd7143c',
 'qwen3_4b_bf16_bias_norm_scan': 'd0fc97cd2358d794da3aad4b67e89bb3223a98495fe5f3bf7b3742d195337c1b',
 'qwen3_4b_bf16_paired_revalidation': 'b7ef0c3f2fe73519a228b1da8b46328b6b8de71c52b83584ac7e1f0a78bb2b55'}


def _declaration_plan(name):
    from pathlib import Path
    from spec_mtp.structure.make.expand import load_study_yaml
    root=Path(__file__).resolve().parents[4]
    study=root/'studies'/name
    return root,study,load_study_yaml(study)['provenance']['project_plan']


def _plan_metadata_digest(body):
    import json
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@pytest.mark.parametrize('study_name',sorted(NAVIGATION_PLAN_METADATA))
def test_current_plan_navigation_keeps_all_frozen_historical_members(study_name):
    root,study,plan=_declaration_plan(study_name)
    from spec_mtp.structure.artifact.provenance import study_source_paths
    current='docs/研究计划 Research Plan 去噪推测解码.md'
    assert plan['path']==current and (root/current).is_file()
    assert '<'+current+'>' in (root/'README.md').read_text(encoding='utf-8')
    assert 'commit' not in plan and 'sha256' not in plan and plan['historical_sections']
    members=study_source_paths(study,{'provenance':{'project_plan':plan}})
    assert members['study.yaml']==study/'study.yaml'
    assert (root/current).resolve() not in members.values()
    assert _plan_metadata_digest({key:value for key,value in plan.items() if key!='path'})==NAVIGATION_PLAN_METADATA[study_name]


@pytest.mark.parametrize('study_name',sorted(PINNED_PLAN_METADATA))
def test_fixed_plan_commit_path_and_digest_never_follow_current_navigation(study_name):
    _,_,plan=_declaration_plan(study_name)
    assert plan['path']=='docs/研究计划 Denoising Speculative Decoding.md'
    assert len(plan['commit'])==40 and len(plan['sha256'])==64
    assert _plan_metadata_digest(plan)==PINNED_PLAN_METADATA[study_name]


@pytest.mark.parametrize('study_name',['output_space_cross_task','qwen3_4b_bf16_bias_norm_scan'])
def test_navigation_cleanup_cannot_bypass_missing_frozen_study_inputs(tmp_path,study_name):
    from spec_mtp.structure.make.write import expand_study
    _,source,_=_declaration_plan(study_name)
    fixture=tmp_path/'source';fixture.mkdir()
    for name in ('study.yaml','experiment_config.yaml','recipe.py'):
        (fixture/name).write_bytes((source/name).read_bytes())
    before={p.name:p.read_bytes() for p in fixture.iterdir()}
    output=tmp_path/'external-runtime'
    with pytest.raises(FileNotFoundError,match='provenance.include matched nothing'):
        expand_study(fixture,run_root=output)
    assert not output.exists()
    assert {p.name:p.read_bytes() for p in fixture.iterdir()}==before
