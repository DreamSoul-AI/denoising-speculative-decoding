"""Verify the fixed upstream tree plus explicit local R1 patches: python tools/check_rpipe_pin.py."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '7a15c46b75c5eb951cccd60ed7b9fc3463db90f5'
BASE_RECORD_SHA256 = '69bb1b44fc73cdfa253e5e1c2d182f9db76a3c03717f85e5776bc604f4c9362c'
NEW_FILE = 'structure/artifact/execution.py'
PATCH_FILES = {'__init__.py', 'flow/cli.py', 'flow/runner.py', 'structure/make/schedule.py',
               'structure/artifact/readout/status.py', 'flow/process/aggregate.py',
               'flow/process/individual.py', NEW_FILE}


def check(root=ROOT):
    root = Path(root)
    vendor = root / 'src/rpipe'
    record = root / 'docs/rpipe-source/manifest.json'
    body = json.loads(record.read_text(encoding='utf-8'))
    assert set(body) == {'base_commit', 'patch_id', 'patch_source', 'hash_policy', 'files'}
    assert body['base_commit'] == BASE_COMMIT
    assert body['patch_id'] == 'sd-r1-run-ownership-20261020'
    assert body['patch_source'] == 'DreamSoul-AI/RPipe@e33bcc8818f43e1f7ba875f6cd8a447d3378d864'
    assert (vendor / 'ORIGIN').read_text(encoding='utf-8').splitlines()[0] == BASE_COMMIT
    files = body['files']
    original_keys = {'git_blob', 'sha256', 'canonical_sha256'}
    assert all(isinstance(item, dict) and original_keys <= item.keys() for item in files.values())
    original = {name: {key: item[key] for key in original_keys}
                for name, item in files.items() if name != NEW_FILE}
    encoded = json.dumps(original, sort_keys=True, separators=(',', ':')).encode()
    assert len(original) == 102 and hashlib.sha256(encoded).hexdigest() == BASE_RECORD_SHA256
    assert {name for name, item in files.items() if 'local_sha256' in item} == PATCH_FILES
    assert {name for name, item in files.items() if 'original_blob' in item} == PATCH_FILES - {NEW_FILE}
    assert all(files[NEW_FILE][key] is None for key in original_keys)
    actual = {p.relative_to(vendor).as_posix() for p in vendor.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.name != 'ORIGIN'}
    assert actual == set(files), f'vendor file set changed: {actual ^ set(files)}'
    for name, item in files.items():
        extra = ({'local_sha256', 'reason'} if name in PATCH_FILES else set())
        if name in PATCH_FILES - {NEW_FILE}:
            extra.add('original_blob')
            assert item['original_blob'] == f"blobs/{item['git_blob']}.txt"
        assert set(item) == original_keys | extra, f'unexpected source metadata: {name}'
        raw = (vendor / name).read_bytes().replace(b'\r\n', b'\n')
        expected = item.get('local_sha256', item['canonical_sha256'])
        assert hashlib.sha256(raw).hexdigest() == expected, f'unrecorded RPipe change: {name}'
        original = item.get('original_blob')
        if original:
            blob_path = record.parent / original
            assert blob_path.resolve().is_relative_to(record.parent.resolve()), name
            raw = blob_path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == item['sha256'], name
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            assert blob == item['git_blob'], name
    return body


if __name__ == '__main__':
    checked = check()
    print(f"RPipe {checked['base_commit']} + {checked['patch_id']}: {len(checked['files'])} files verified")
