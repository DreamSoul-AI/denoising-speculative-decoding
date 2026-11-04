"""Validate frozen parameters and convert existing latent evidence to measurements."""
import hashlib
import json
from pathlib import Path

from studies.k_affine_residual.structure.correction import FrozenCorrection, digest
from studies.k_affine_residual.structure.evaluation import aligned_arrays, evaluate_pairs, validate_documents


def read_report(assets):
    assets = Path(assets)
    raw = json.loads((assets / 'latent_pairs.json').read_text(encoding='utf-8'))
    if type(raw.get('seed')) is not int:
        raise ValueError('raw seed must be an integer')
    frozen_path = assets / 'frozen_correction.json'
    frozen = FrozenCorrection(**json.loads(frozen_path.read_text(encoding='utf-8')))
    file_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if (raw['frozen_sha_before'] != frozen.sha256 or raw['frozen_sha_after'] != frozen.sha256
            or raw['frozen_file_sha_before'] != file_sha or raw['frozen_file_sha_after'] != file_sha):
        raise ValueError('frozen correction changed or mismatches raw evidence')
    docs = validate_documents(raw['documents'], raw['positions'])
    fit_docs = [doc for doc in docs if doc['split'] == 'fit']
    body = frozen.body()
    if (body['model_space'] != raw['model_space'] or body['fit_data_sha'] != digest(raw['fit_pairs'])
            or sorted(tuple(row) for row in body['fit_documents'])
            != sorted((doc['id'], doc['content_sha']) for doc in fit_docs)):
        raise ValueError('fit source or model space mismatch')
    aligned_arrays(raw['fit_pairs'], fit_docs, raw['positions'], raw['model_space'])
    report = evaluate_pairs(raw['pairs'], [doc for doc in docs if doc['split'] != 'fit'],
                            raw['positions'], frozen)
    report['scope'] = raw['scope']
    report['model'] = raw['model']
    report['seed'] = raw['seed']
    return raw, report


def run(ctx):
    raw, report = read_report(ctx.layout.assets_dir)
    if type(ctx.control.seed) is not int or raw['seed'] != ctx.control.seed:
        raise ValueError('raw seed does not match current Run seed')
    if ctx.state['execute'].get('raw_sha') != digest(raw):
        raise ValueError('execute/collect raw mismatch')
    ctx.state['k_affine_residual'] = report
    ctx.state['collected']['metrics'].update({
        f'{split}_{arm}_mse': value
        for split, row in report['splits'].items() for arm, value in row['mse'].items()})
