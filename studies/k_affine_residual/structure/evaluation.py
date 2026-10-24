"""Postcommit alignment, document-level three-arm pairing and completeness."""

from __future__ import annotations

from collections import defaultdict

import torch

from .correction import ARMS, apply_correction, digest, matrix, require_sha

SPLITS = ('fit', 'holdout', 'cross_dataset')


def validate_documents(documents, positions):
    if not positions or any(type(p) is not int or p < 1 for p in positions) or len(set(positions)) != len(positions):
        raise ValueError('positions must be unique positive prefix lengths')
    if not documents:
        raise ValueError('documents must be nonempty')
    seen_ids, content_owner, token_owner, prefix_owner = set(), {}, {}, {}
    checked = []
    for doc in documents:
        name, split, tokens = doc['id'], doc['split'], doc['tokens']
        if not isinstance(name, str) or not name or name in seen_ids or split not in SPLITS:
            raise ValueError('invalid/duplicate document ID or split')
        if not tokens or any(type(t) is not int or t < 0 for t in tokens) or max(positions) >= len(tokens):
            raise ValueError('document requires integer tokens and all committed positions')
        token_sha = digest(tokens)
        content_sha = doc.get('content_sha', token_sha)
        require_sha(content_sha, 'document content hash')
        for owners, sha in ((content_owner, content_sha), (token_owner, token_sha)):
            if sha in owners and owners[sha] != split:
                raise ValueError('fit/test document content overlap')
            owners[sha] = split
        for position in positions:
            prefix = digest(tokens[:position])
            if prefix in prefix_owner and prefix_owner[prefix] != split:
                raise ValueError('fit/test exact prefix overlap')
            prefix_owner[prefix] = split
        seen_ids.add(name)
        checked.append({'id': name, 'split': split, 'tokens': list(tokens),
                        'content_sha': content_sha, 'token_sha': token_sha})
    if set(doc['split'] for doc in checked) != set(SPLITS):
        raise ValueError('fit, holdout and cross_dataset documents are required')
    return checked


def pair_arrays(pair, document):
    expected = {'document_id', 'document_sha', 'split', 'position', 'prefix', 'committed_token',
                'prefix_embeddings', 'committed_embedding', 'z_soft', 'z_actual', 'model_space'}
    if set(pair) != expected:
        raise ValueError('pair schema rejects missing fields and future/next-token labels')
    pos = pair['position']
    if type(pos) is not int or pos < 1 or pos >= len(document['tokens']):
        raise ValueError('invalid committed position')
    if (pair['document_id'] != document['id'] or pair['document_sha'] != document['token_sha']
            or pair['split'] != document['split'] or pair['prefix'] != document['tokens'][:pos]
            or type(pair['committed_token']) is not int or pair['committed_token'] != document['tokens'][pos]):
        raise ValueError('postcommit P/y/owner alignment mismatch')
    embeddings = matrix(pair['prefix_embeddings'], 'prefix embeddings')
    ey = matrix([pair['committed_embedding']], 'committed embedding')
    zs, zy = (matrix([pair[name]], name) for name in ('z_soft', 'z_actual'))
    if embeddings.shape[0] != pos or ey.shape[1] != embeddings.shape[1] or zs.shape != zy.shape:
        raise ValueError('pair dimensions disagree')
    require_sha(pair['model_space'], 'pair model space')
    return ey - embeddings.mean(0, keepdim=True), zs, zy


def aligned_arrays(pairs, documents, positions, model_space):
    by_id = {doc['id']: doc for doc in documents}
    expected = {(doc['id'], p) for doc in documents for p in positions}
    seen, xs, zs, ys = set(), [], [], []
    for pair in pairs:
        key = (pair['document_id'], pair['position'])
        if key not in expected or key in seen or pair['model_space'] != model_space:
            raise ValueError('duplicate, foreign or wrong model-space pair')
        x, z, y = pair_arrays(pair, by_id[pair['document_id']])
        xs.append(x)
        zs.append(z)
        ys.append(y)
        seen.add(key)
    if seen != expected:
        raise ValueError('incomplete document/position pairing')
    return torch.cat(xs), torch.cat(zs), torch.cat(ys)


def evaluate_pairs(pairs, documents, positions, frozen):
    body = frozen.body()
    if any(doc['split'] == 'fit' or doc['id'] in dict(body['fit_documents']) for doc in documents):
        raise ValueError('fit/test document ID overlap')
    if {doc['content_sha'] for doc in documents} & {sha for _, sha in body['fit_documents']}:
        raise ValueError('fit/test document content overlap')
    x, z, y = aligned_arrays(pairs, documents, positions, body['model_space'])
    errors = {arm: (apply_correction(frozen, x, z, arm, model_space=body['model_space']) - y)
              .square().mean(1).tolist() for arm in ARMS}
    docs = []
    for document in documents:
        indices = [i for i, pair in enumerate(pairs) if pair['document_id'] == document['id']]
        mse = {arm: sum(errors[arm][i] for i in indices) / len(indices) for arm in ARMS}
        docs.append({'document_id': document['id'], 'split': document['split'], 'positions': len(indices),
                     'mse': mse, 'full_minus_uncorrected': mse['full'] - mse['uncorrected'],
                     'full_minus_bias': mse['full'] - mse['bias']})
    groups = defaultdict(list)
    for doc in docs:
        groups[doc['split']].append(doc)
    splits = {split: {
        'documents': len(rows),
        'mse': {arm: sum(row['mse'][arm] for row in rows) / len(rows) for arm in ARMS},
        'full_minus_uncorrected': sum(row['full_minus_uncorrected'] for row in rows) / len(rows),
        'full_minus_bias': sum(row['full_minus_bias'] for row in rows) / len(rows),
    } for split, rows in groups.items()}
    if set(splits) != {'holdout', 'cross_dataset'}:
        raise ValueError('both evaluation splits are required')
    return {'complete': True, 'arms': list(ARMS), 'documents': docs, 'splits': splits,
            'pairs': len(pairs), 'frozen_sha': frozen.sha256, 'model_space': body['model_space'],
            'scientific_claim': 'postcommit hidden MSE only; no hit-rate or generalization claim'}


def check_matrix(records, seeds):
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('declared seeds must be unique integers')
    by_seed = {}
    for record in records:
        seed = record.get('seed')
        report = record.get('k_affine_residual') or {}
        if type(seed) is not int or seed not in seeds or seed in by_seed:
            raise ValueError('missing, duplicate or undeclared seed')
        if type(report.get('seed')) is not int or report['seed'] != seed:
            raise ValueError('report seed does not match Run seed')
        if record.get('status') != 'succeeded' or report.get('complete') is not True:
            raise ValueError('incomplete or failed Run')
        if report.get('arms') != list(ARMS) or set(report.get('splits', {})) != {'holdout', 'cross_dataset'}:
            raise ValueError('incomplete arm/split matrix')
        require_sha(report.get('frozen_sha'), 'frozen correction hash')
        for split in report['splits'].values():
            if type(split.get('documents')) is not int or split['documents'] < 1:
                raise ValueError('missing documents')
            if set(split.get('mse', {})) != set(ARMS):
                raise ValueError('missing arm measurement')
            values = matrix([list(split['mse'].values())], 'split MSE')
            if (values < 0).any():
                raise ValueError('split MSE must be nonnegative')
        by_seed[seed] = report
    if set(by_seed) != set(seeds):
        raise ValueError('incomplete seed matrix; successful subset is not complete')
    return {'complete': True, 'seeds': sorted(by_seed), 'arms': list(ARMS)}
