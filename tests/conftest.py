"""Shared helpers: copy the smoke Study into a temp dir and run it through RPipe."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests import _process_tree

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'studies' / 'smoke'

# Child interpreters must verify this checkout, not a different globally editable SD.
os.environ['PYTHONPATH'] = os.pathsep.join((
    str(ROOT / 'src'), str(ROOT), str(ROOT / 'studies' / '_shared' / 'structure')))
if str(ROOT / 'studies' / '_shared' / 'structure') not in sys.path:
    sys.path.insert(0, str(ROOT / 'studies' / '_shared' / 'structure'))


def copy_study(tmp_path, patch=None, name='smoke_copy'):
    """Copy declarations, recipe and flow (no runs) and apply a study.yaml patch."""
    dest = tmp_path / name
    shutil.copytree(SMOKE, dest, ignore=shutil.ignore_patterns(
        'runs', 'shared', 'scripts', 'index.json', 'process.json', 'provenance.json',
        'spec_mtp_*.json', 'activity.json', '__pycache__', 'docs'))
    study = yaml.safe_load((dest / 'study.yaml').read_text(encoding='utf-8'))
    study['study'] = name
    study['fixed']['algorithm']['config']['version'] = name
    if patch:
        patch(study)
    (dest / 'study.yaml').write_text(yaml.safe_dump(study, sort_keys=False), encoding='utf-8')
    return dest


def cli(*args, entry='studies._shared.flow.cli', check=False, timeout=90):
    env = {**os.environ, 'PYTHONUTF8': '1', 'MPLBACKEND': 'Agg'}
    proc = _process_tree.sh([sys.executable, '-m', entry, *map(str, args)],
                          capture_output=True, text=True, encoding='utf8', errors='backslashreplace',
                          check=False, env=env, cwd=ROOT, timeout=timeout)
    if check and proc.returncode != 0:
        raise AssertionError(proc.stdout[-4000:] + proc.stderr[-4000:])
    return proc


@pytest.fixture
def study_copy(tmp_path):
    return lambda patch=None, name='smoke_copy': copy_study(tmp_path, patch, name)


@pytest.fixture(scope='session')
def historical_file(tmp_path_factory):
    """Read immutable historical inputs, never import or execute archived source."""
    import hashlib

    commit = 'b8b20b5ada7fd860894114cbf224ce749af430cf'
    root = tmp_path_factory.mktemp('historical-inputs')

    def materialize(relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('historical member must stay inside the archive')
        command = ['git', '-c', 'safe.directory=' + str(ROOT), '-C', str(ROOT)]
        oid = subprocess.check_output(command + ['rev-parse', commit + ':' + relative.as_posix()],
                                      text=True, timeout=20).strip()
        data = subprocess.check_output(command + ['cat-file', 'blob', oid], timeout=20)
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == oid
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
        assert path.read_bytes() == data, 'historical input was changed: ' + str(relative)
        return path

    return materialize


@pytest.fixture
def saved_decode_config():
    def _config(task='esp_mtp', run_ar=True):
        return {'algorithm': {'config': {'task': task, 'eval': {'run_ar_baseline': run_ar}}}}
    return _config


@pytest.fixture
def saved_decode_assets(saved_decode_config):
    def _assets(tmp_path, rate='1.0', with_sources=True, overall=True, name='a'):
        import csv
        import json
        from studies._shared.structure import sources
        from studies._shared.structure.evidence import digest

        assets = tmp_path / name
        assets.mkdir(parents=True)
        rows = [{'category': 'synthetic', 'exact_match_rate': rate, 'accept_d1': '5'}]
        if overall:
            rows.append({'category': 'overall', 'exact_match_rate': rate, 'accept_d1': '5'})
        with (assets / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        cfg = saved_decode_config()
        saved = sources.manifest()
        saved['config_digest'] = digest(cfg)
        study_dir = assets.parents[2]
        if (study_dir / 'study.yaml').exists():
            from spec_mtp.structure.artifact.provenance import source_files
            from spec_mtp.structure.make.expand import load_study_yaml
            saved['study_files'] = source_files(study_dir, load_study_yaml(study_dir))
        if with_sources:
            (assets / 'spec_mtp_sources.json').write_text(json.dumps(saved), encoding='utf-8')
        inputs = {'settings': {'run': {'seed': 0}, 'eval': {'run_ar_baseline': True, 'device': 'cpu'}, 'spec': {}},
                  'files': {}, 'device': 'cpu', 'prompts': [{'input_ids': [1, 2], 'category': 'synthetic'}]}
        stats = {'new_tokens': 6, 'committed': 5, 'accepted': 5, 'prefill_calls': 1, 'decode_calls': 1,
                 'trace': [{'root_position': 2, 'cache_before': 2, 'accepted_nodes': [1, 2, 3, 4, 5],
                            'stopped': True, 'cache_after': None, 'committed_tokens': [4, 5, 6, 7, 8]}]}
        row = {'category': 'synthetic', **{k: stats[k] for k in ('new_tokens', 'committed', 'accepted', 'decode_calls')},
               'spec_time': 1, 'ar_time': 1, 'exact_match': float(rate) if rate else 1,
               'accept_d1': 5, 'attempt_d1': 5}
        item = {'input_ids': [1, 2], 'output_ids': [1, 2, 3, 4, 5, 6, 7, 8],
                'ar_ids': [1, 2, 3, 4, 5, 6, 7, 8], 'stats': stats}
        import hashlib
        token_hash = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        stats['verification_argmax_audit'] = {'passed': True, 'checked_committed_tokens': 6,
            'input_sha256': token_hash(item['input_ids']), 'output_sha256': token_hash(item['output_ids']),
            'committed': [{'forward': 0, 'row': 1, 'chosen': 3, 'argmax': 3}] +
                [{'forward': 1, 'row': i, 'chosen': token, 'argmax': token} for i, token in enumerate([4,5,6,7,8])]}
        if rate not in ('1.0',''):
            item['ar_ids'][-1] = 99
            row['exact_match'] = 0
        raw = {'task': 'esp_mtp', 'inputs_digest': digest(inputs), 'out': {'rows': [row], 'evidence': [item]}}
        (assets / 'decode_inputs.json').write_text(json.dumps(inputs))
        (assets / 'decode_observations.json').write_text(json.dumps(raw))
        if overall and rate == '1.0':
            from spec_mtp.structure.algorithm.counts import _aggregate
            flat = [{'category': k, **v} for k, v in _aggregate([row]).items()]
            with (assets / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
        return assets
    return _assets


@pytest.fixture
def saved_decode_study(saved_decode_config, saved_decode_assets):
    def _study(tmp_path, runs, axes=None, seeds=(123,)):
        import json
        from types import SimpleNamespace
        from studies._shared.structure import sources
        import yaml

        study = tmp_path / 'study'
        (study / 'runs').mkdir(parents=True)
        (study / 'experiment_config.yaml').write_text(yaml.safe_dump(saved_decode_config()))
        (study / 'study.yaml').write_text(yaml.safe_dump(
            {'study': 'x', 'axes': axes or {'algorithm.config.variant': ['a', 'b']}, 'seeds': list(seeds)}))
        experiments = []
        for variant, status, passed, source_digest in runs:
            run_id = f'run_{variant}'
            experiments.append({'factors': {'algorithm.config.variant': variant},
                                'runs': [{'id': run_id, 'run_dir': run_id, 'seed': 123}]})
            if status is None:
                continue
            (study / 'runs' / run_id).mkdir()
            saved_decode_assets(study / 'runs' / run_id, name='assets')
            (study / 'runs' / run_id / 'config.yaml').write_text(__import__('yaml').safe_dump(saved_decode_config()))
            digest_value = sources.manifest()['digest'] if source_digest == 'd' else source_digest
            (study / 'runs' / run_id / 'result.json').write_text(json.dumps(
                {'status': status, 'spec_mtp': {'passed': passed, 'sources_digest': digest_value}}))
        (study / 'index.json').write_text(json.dumps({'study': 'x', 'experiments': experiments}))
        return SimpleNamespace(scope='study', study_dir=study, state={})
    return _study


@pytest.fixture
def corrupt_saved_argmax():
    def _corrupt_audit(assets, key, value):
        import json

        path = assets / 'decode_observations.json'
        raw = json.loads(path.read_text())
        raw['out']['evidence'][0]['stats']['verification_argmax_audit'][key] = value
        path.write_text(json.dumps(raw))
    return _corrupt_audit
