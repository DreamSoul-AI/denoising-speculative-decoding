"""Study-local postcommit pairs and frozen affine calibration; no decoder changes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.data.factory import Data

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
ARMS = ('none', 'bias_only', 'full_k')
SPLITS = ('fit', 'holdout', 'cross_dataset')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, body):
    Path(path).write_text(json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def source_map():
    return {str(p.relative_to(REPO)): sha(p) for p in sorted((REPO / 'src/spec_mtp').rglob('*.py'))}


def check_assets(manifest):
    for path, expected in manifest['assets'].items():
        if sha(path) != expected['sha256']:
            raise ValueError(f'read-only input changed: {path}')


def validate_documents(documents, lengths):
    seen, prefixes = {}, {split: set() for split in SPLITS}
    for doc in documents:
        split = doc['split']
        digest = doc['document_sha256']
        if digest in seen:
            raise ValueError('duplicate document across or within partitions')
        seen[digest] = split
        tokens = doc['token_ids']
        if len(tokens) < max(lengths):
            raise ValueError('document shorter than registered positions')
        for length in lengths:
            if length < 2:
                raise ValueError('soft mean requires a nonempty past')
            prefixes[split].add(tuple(tokens[:length]))
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            if prefixes[split] & prefixes[other]:
                raise ValueError('exact prefix crosses fit/evaluation partitions')


def fit_affine(x, residual, ridge):
    """Independent bias baseline and full ridge, with unpenalized intercept."""
    x, residual = np.asarray(x, dtype=np.float64), np.asarray(residual, dtype=np.float64)
    if x.ndim != 2 or residual.shape != x.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(residual).all():
        raise ValueError('fit requires finite paired matrices of equal shape')
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('registered ridge must be positive and finite')
    xm, rm = x.mean(0), residual.mean(0)
    xc, rc = x - xm, residual - rm
    a = np.linalg.solve(xc.T @ xc + ridge * np.eye(x.shape[1]), xc.T @ rc).T
    return {'a': a, 'b': rm - a @ xm, 'bias_only': rm}


def predictions(x, zmask, frozen):
    x, zmask = np.asarray(x, dtype=np.float64), np.asarray(zmask, dtype=np.float64)
    return np.stack((zmask, zmask + frozen['bias_only'], zmask + x @ frozen['a'].T + frozen['b']), axis=1)


def model_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def extract_pair(module, tokens, length):
    """Only tokens[:length] are read; y is its last, already committed token."""
    import torch

    with torch.inference_mode():
        ids = torch.tensor(tokens[:length], dtype=torch.long, device='cpu').unsqueeze(0)
        embedded = module.embed(ids)
        mean = embedded[:, :-1].mean(dim=1)
        soft = torch.cat((embedded[:, :-1], mean.unsqueeze(1)), dim=1)
        positions = torch.arange(length, device='cpu').unsqueeze(0)
        zmask = module.hf.model(inputs_embeds=soft, position_ids=positions, use_cache=False,
                               return_dict=True).last_hidden_state[:, -1]
        ztrue = module.hf.model(inputs_embeds=embedded, position_ids=positions, use_cache=False,
                               return_dict=True).last_hidden_state[:, -1]
        return tuple(t.squeeze(0).cpu().numpy().copy() for t in (embedded[:, -1] - mean, zmask, ztrue))


def check_plan_bytes(blob_bytes, working_bytes, blob_sha256, working_sha256):
    if (hashlib.sha256(blob_bytes).hexdigest() != blob_sha256
            or hashlib.sha256(working_bytes).hexdigest() != working_sha256):
        raise ValueError('registered immutable plan or working bytes changed')


def prepare_binding(ctx):
    registration = read(STUDY / 'docs/registration.json')
    manifest = read(STUDY / 'docs/data_manifest.json')
    expected = {'study.yaml': registration['declaration_sha256'],
                'experiment_config.yaml': registration['base_config_sha256'],
                'docs/data_manifest.json': registration['data_manifest_sha256']}
    for name, digest in expected.items():
        if sha(STUDY / name) != digest:
            raise ValueError(f'registration changed: {name}')
    git = ['git', '-C', str(REPO), '-c', f'safe.directory={REPO}']
    def git_read(*args):
        return subprocess.run(git + list(args), capture_output=True, check=True, timeout=15).stdout
    head = git_read('rev-parse', 'HEAD').decode().strip()
    blobs = {}
    for name, digest in registration['project_plans'].items():
        expected_blob = registration['project_plan_blobs'][name]
        blob = git_read('show', f'HEAD:{name}')
        blob_id = git_read('rev-parse', f'HEAD:{name}').decode().strip()
        original = git_read('rev-parse', f"{expected_blob['commit']}:{name}").decode().strip()
        if blob_id != expected_blob['blob'] or original != blob_id:
            raise ValueError('immutable registered plan blob identity changed')
        check_plan_bytes(blob, (REPO / name).read_bytes(), expected_blob['sha256'], digest)
        blobs[name] = dict(blob=blob_id, blob_sha256=expected_blob['sha256'], working_sha256=digest,
                           registered_plan_commit=expected_blob['commit'])
    if ctx.config['system']['config']['device'] != 'cpu':
        raise ValueError('this registration permits CPU only')
    check_assets(manifest)
    binding = dict(head=head, project_plans=blobs, section=registration['plan_section'],
                   registration_sha256=sha(STUDY / 'docs/registration.json'), spec_mtp_sources=source_map())
    write(ctx.assets_dir / 'k_affine_binding.json', binding)


def build_data(config, assets_dir, seed):
    del config, assets_dir, seed
    manifest = read(STUDY / 'docs/data_manifest.json')
    validate_documents(manifest['documents'], manifest['prefix_lengths'])
    check_assets(manifest)
    return Data(name='cached_documents', source='k_affine_residual', loaders={}, meta=manifest)


class AffineAlgorithm(Algorithm):
    def run(self, data, model, system, tracker):
        import time
        import torch
        import transformers

        del tracker
        started = time.perf_counter()
        module = model.module
        module.eval()
        root = system.assets_dir / 'affine'
        root.mkdir()  # a Run may not overwrite an earlier extraction
        lengths = self.config.setting('prefix_lengths')
        expected = self.config.setting('expected_pairs')
        if lengths != data.meta['prefix_lengths'] or self.config.setting('arms') != list(ARMS):
            raise ValueError('Run differs from the registered positions/arms')
        validate_documents(data.meta['documents'], lengths)
        before = model_digest(module)
        hf = module.hf
        tied = hf.get_input_embeddings().weight.data_ptr() == hf.get_output_embeddings().weight.data_ptr()
        if not tied or module.device.type != 'cpu' or module.dtype != torch.float32:
            raise ValueError('registered tied CPU FP32 model is required')
        actual = {name: getattr(hf.config, name) for name in ('model_type', 'vocab_size', 'hidden_size',
                  'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
                  'max_position_embeddings', 'tie_word_embeddings')}
        if actual != dict(model_type='llama', vocab_size=151936, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=128, tie_word_embeddings=True):
            raise ValueError(f'actual model differs from registration: {actual}')
        identity = dict(unique_parameters=sum(p.numel() for p in hf.parameters()),
                        stored_state_elements=sum(p.numel() for p in hf.state_dict().values()),
                        tied_storage=tied, dtype=str(module.dtype), device=str(module.device),
                        transformers=transformers.__version__, torch=torch.__version__,
                        model_config=hf.config.to_dict(), state_sha256_before=before,
                        torch_threads=torch.get_num_threads(), model_seed=torch.initial_seed())
        write(root / 'model_identity_before.json', identity)
        rows, xs, masks, targets = [], [], [], []
        frozen, freeze_hash = None, None
        for split in SPLITS:
            if split != 'fit' and (frozen is None or sha(root / 'frozen_k.npz') != freeze_hash):
                raise ValueError('heldout extraction requires unchanged frozen calibration')
            start = len(rows)
            for doc in data.meta['documents']:
                if doc['split'] != split:
                    continue
                for length in lengths:
                    x, mask, target = extract_pair(module, doc['token_ids'], length)
                    xs.append(x); masks.append(mask); targets.append(target)
                    rows.append(dict(split=split, document_sha256=doc['document_sha256'],
                                     document_id=doc['document_id'], length=length, position=length-1,
                                     committed_y=doc['token_ids'][length-1], prefix_tokens=doc['token_ids'][:length]))
            if len(rows) - start != expected[split]:
                raise ValueError('registered pair count mismatch')
            if split == 'fit':
                frozen = fit_affine(np.stack(xs), np.stack(targets) - np.stack(masks), self.config.setting('ridge_lambda'))
                np.savez(root / 'frozen_k.npz', **frozen)
                freeze_hash = sha(root / 'frozen_k.npz')
                write(root / 'freeze.json', dict(frozen_k_sha256=freeze_hash, fit_rows=len(rows),
                      fit_document_sha256=[d['document_sha256'] for d in data.meta['documents'] if d['split']=='fit'],
                      evaluation_rows_extracted_at_freeze=0, independently_fitted_bias=True,
                      ridge_lambda=self.config.setting('ridge_lambda')))
        x, zmask, ztrue = np.stack(xs), np.stack(masks), np.stack(targets)
        predicted = predictions(x, zmask, frozen)
        # Full vocab is computed for diagnostics; only its per-row reductions are persisted.
        agreement, kl = np.empty((len(rows), 3)), np.empty((len(rows), 3))
        with torch.inference_mode():
            for i in range(len(rows)):
                truth_logits = hf.lm_head(torch.from_numpy(ztrue[i])).double()
                truth_logp = torch.log_softmax(truth_logits, dim=-1)
                for j in range(3):
                    logits = hf.lm_head(torch.from_numpy(predicted[i, j].astype(np.float32))).double()
                    agreement[i, j] = int(logits.argmax() == truth_logits.argmax())
                    kl[i, j] = float((truth_logp.exp() * (truth_logp - torch.log_softmax(logits, dim=-1))).sum())
        np.savez(root / 'pairs.npz', x=x, zmask=zmask, ztrue=ztrue, predicted=predicted,
                 argmax_agreement=agreement, target_kl=kl)
        write(root / 'rows.json', rows)
        after = model_digest(module)
        if before != after or sha(root / 'frozen_k.npz') != freeze_hash:
            raise ValueError('frozen model or calibration changed')
        identity.update(state_sha256_after=after, frozen_k_sha256=freeze_hash,
                        elapsed_seconds=time.perf_counter()-started,
                        raw_files={p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file()})
        write(root / 'execution.json', identity)
        return dict(mode='eval', elapsed_seconds=identity['elapsed_seconds'])


def analyze(root):
    """Recompute metrics from raw pairs, never treating an improvement as validity."""
    rows, execution, freeze = read(root / 'rows.json'), read(root / 'execution.json'), read(root / 'freeze.json')
    for name, digest in execution['raw_files'].items():
        if sha(root / name) != digest:
            raise ValueError(f'raw artifact changed: {name}')
    if sha(root / 'frozen_k.npz') != freeze['frozen_k_sha256'] or freeze['evaluation_rows_extracted_at_freeze'] != 0:
        raise ValueError('calibration was not frozen before evaluation')
    if execution['state_sha256_before'] != execution['state_sha256_after']:
        raise ValueError('model weights changed')
    with np.load(root / 'pairs.npz', allow_pickle=False) as raw, np.load(root / 'frozen_k.npz', allow_pickle=False) as frozen:
        arrays = {key: raw[key] for key in raw.files}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('nonfinite observation')
        count = len(rows)
        shapes = dict(x=(count,32), zmask=(count,32), ztrue=(count,32), predicted=(count,3,32),
                      argmax_agreement=(count,3), target_kl=(count,3))
        if set(arrays) != set(shapes) or any(arrays[key].shape != shape for key,shape in shapes.items()):
            raise ValueError('raw array shape mismatch')
        fit_indices = [i for i,row in enumerate(rows) if row['split']=='fit']
        refit = fit_affine(arrays['x'][fit_indices], arrays['ztrue'][fit_indices]-arrays['zmask'][fit_indices],
                           freeze['ridge_lambda'])
        if any(not np.array_equal(frozen[key], refit[key]) for key in refit):
            raise ValueError('frozen full K/bias was not independently fitted on fit rows')
        calculated = predictions(arrays['x'], arrays['zmask'], frozen)
        if not np.array_equal(calculated, arrays['predicted']):
            raise ValueError('predictions do not use the frozen K')
        target = arrays['ztrue'][:, None, :].astype(np.float64)
        mse = ((calculated-target)**2).mean(axis=-1)
        denominator = np.linalg.norm(calculated, axis=-1) * np.linalg.norm(target, axis=-1)
        cosine = (calculated*target).sum(axis=-1) / denominator
        result = {'splits': {}, 'execution': execution, 'frozen_k_sha256': freeze['frozen_k_sha256']}
        manifest = read(STUDY / 'docs/data_manifest.json')
        expected_rows = [dict(split=split, document_sha256=doc['document_sha256'], document_id=doc['document_id'],
                              length=length, position=length-1, committed_y=doc['token_ids'][length-1],
                              prefix_tokens=doc['token_ids'][:length])
                         for split in SPLITS for doc in manifest['documents'] if doc['split'] == split
                         for length in manifest['prefix_lengths']]
        if rows != expected_rows:
            raise ValueError('raw rows differ from the registered documents/causal positions')
        for split in SPLITS:
            indices = [i for i, row in enumerate(rows) if row['split'] == split]
            if len(indices) != manifest['pair_counts'][split]:
                raise ValueError('incomplete raw partition')
            documents = sorted({rows[i]['document_sha256'] for i in indices})
            doc_mse = np.stack([mse[[i for i in indices if rows[i]['document_sha256']==doc]].mean(0) for doc in documents])
            result['splits'][split] = dict(pairs=len(indices), documents=len(documents),
                arms={arm:dict(document_mean_mse=float(doc_mse[:,j].mean()),
                               mean_cosine=float(cosine[indices,j].mean()),
                               target_argmax_agreement=float(arrays['argmax_agreement'][indices,j].mean()),
                               target_kl=float(arrays['target_kl'][indices,j].mean())) for j,arm in enumerate(ARMS)},
                paired_full_minus_none=(doc_mse[:,2]-doc_mse[:,0]).tolist(),
                paired_full_minus_bias=(doc_mse[:,2]-doc_mse[:,1]).tolist())
        result['valid'] = True
        return result


def check_run(assets):
    binding = read(assets / 'k_affine_binding.json')
    if binding['spec_mtp_sources'] != source_map():
        raise ValueError('SpecMTP source changed after prepare')
    check_assets(read(STUDY / 'docs/data_manifest.json'))
    return analyze(assets / 'affine')
