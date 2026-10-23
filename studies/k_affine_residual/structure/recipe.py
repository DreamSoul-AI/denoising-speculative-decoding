"""Offline CPU pilot and RPipe v0.3.1 public-registry assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from studies.k_affine_residual.structure.correction import ARMS, PROTOCOL, digest, fit_correction
from studies.k_affine_residual.structure.evaluation import aligned_arrays, validate_documents

SOURCE = 'k_affine_residual'


def model_identity(session):
    state = hashlib.sha256()
    for name, tensor in sorted(session.hf.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite model parameter')
        state.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        state.update(value.view(torch.uint8).numpy().tobytes())
    return digest({'weights': state.hexdigest(), 'config': session.hf.config.to_dict(),
                   'protocol': PROTOCOL, 'representation': 'final normalized hidden at committed position',
                   'torch': torch.__version__, 'model_class': type(session.hf).__name__})


@torch.no_grad()
def extract_pairs(session, documents, positions, model_space):
    if str(session.device) != 'cpu' or session.dtype != torch.float32 or not session.tied_embeddings:
        raise ValueError('pilot requires CPU FP32 and actually tied embeddings')
    pairs = []
    for doc in documents:
        for pos in positions:
            # Read only the prefix plus the already committed y; no next-word target.
            prefix = doc['tokens'][:pos]
            y = doc['tokens'][pos]
            ids = torch.tensor([prefix + [y]], dtype=torch.long, device='cpu')
            embeddings = session.embed(ids)
            soft = embeddings.clone()
            soft[:, -1] = embeddings[:, :-1].mean(1)
            positions_tensor = torch.arange(pos + 1).unsqueeze(0)
            zm = session.forward(soft, positions_tensor, last_hidden_only=True).last_hidden_state[:, -1]
            zy = session.forward(embeddings, positions_tensor, last_hidden_only=True).last_hidden_state[:, -1]
            pairs.append({
                'document_id': doc['id'], 'document_sha': doc['token_sha'], 'split': doc['split'],
                'position': pos, 'prefix': prefix, 'committed_token': y,
                'prefix_embeddings': embeddings[0, :-1].tolist(),
                'committed_embedding': embeddings[0, -1].tolist(),
                'z_soft': zm[0].tolist(), 'z_actual': zy[0].tolist(), 'model_space': model_space,
            })
    return pairs


def run_pilot(session, documents, positions, assets_dir, *, seed, extract=extract_pairs):
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    docs = validate_documents(documents, positions)
    assets = Path(assets_dir)
    frozen_path, raw_path = assets / 'frozen_correction.json', assets / 'latent_pairs.json'
    if frozen_path.exists() or raw_path.exists():
        raise FileExistsError('pilot evidence already exists; use a new Run')
    session.eval()
    identity = model_identity(session)
    fit_docs = [doc for doc in docs if doc['split'] == 'fit']
    fit_pairs = extract(session, fit_docs, positions, identity)
    x, z, y = aligned_arrays(fit_pairs, fit_docs, positions, identity)
    frozen = fit_correction(x, y - z, model_space=identity,
                            fit_documents=[(doc['id'], doc['content_sha']) for doc in fit_docs],
                            fit_data_sha=digest(fit_pairs))
    assets.mkdir(parents=True, exist_ok=True)
    with frozen_path.open('x', encoding='utf-8') as handle:
        json.dump(frozen.record(), handle, sort_keys=True, allow_nan=False)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    before = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if identity != model_identity(session):
        raise ValueError('model changed during fit extraction')
    evaluation = []
    for split in ('holdout', 'cross_dataset'):
        evaluation.extend(extract(session, [doc for doc in docs if doc['split'] == split], positions, identity))
    if before != hashlib.sha256(frozen_path.read_bytes()).hexdigest() or identity != model_identity(session):
        raise ValueError('frozen K or model changed during evaluation')
    raw = {
        'protocol': PROTOCOL, 'arms': list(ARMS), 'seed': seed,
        'documents': docs, 'positions': list(positions), 'fit_pairs': fit_pairs, 'pairs': evaluation,
        'model_space': identity, 'frozen_sha_before': frozen.sha256, 'frozen_sha_after': frozen.sha256,
        'frozen_file_sha_before': before, 'frozen_file_sha_after': before,
        'model': {'class': type(session.hf).__name__, 'parameters': sum(p.numel() for p in session.hf.parameters()),
                  'device': str(session.device), 'dtype': str(session.dtype), 'tied': session.tied_embeddings,
                  'config': session.hf.config.to_dict(), 'torch': torch.__version__},
        'scope': 'synthetic engineering pilot; not the preregistered 160-pair study',
    }
    with raw_path.open('x', encoding='utf-8') as handle:
        json.dump(raw, handle, sort_keys=True, allow_nan=False)
    return {'mode': 'eval', 'task': SOURCE, 'raw_sha': digest(raw)}


def _data_builder(config, assets_dir, seed=None):
    from rpipe.structure.api.data_api import Data

    del assets_dir
    docs = config.config['documents']
    return Data(name=config.name, source=SOURCE, loaders={}, meta={'documents': docs, 'seed': seed})


def _model_builder(config, assets_dir, data_meta=None):
    from rpipe.structure.api.model_api import Model
    from spec_mtp.model import HFCausalLM
    from transformers import LlamaConfig, LlamaForCausalLM

    del assets_dir
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(data_meta['seed'])
        native = LlamaForCausalLM(LlamaConfig(**config.config, attn_implementation='eager')).to('cpu').eval()
    session = HFCausalLM(module=native)
    if not session.tied_embeddings:
        raise ValueError('tiny model must use tied input/output storage')
    return Model(name=config.name, source=SOURCE, module=session, meta={'random_seed': data_meta['seed']})


def register(ctx):
    from rpipe.structure.api.algorithm_api import Algorithm, AlgorithmRegistry
    from rpipe.structure.api.data_api import DataRegistry
    from rpipe.structure.api.model_api import ModelRegistry

    cfg = ctx.config
    if cfg['algorithm'].get('mode') != 'eval' or cfg['algorithm'].get('source') != SOURCE:
        raise ValueError('K pilot requires its own eval Algorithm')
    if any(cfg[name].get('source') != SOURCE for name in ('data', 'model')):
        raise ValueError('K pilot requires explicit local token documents and random tiny model')
    if (cfg['data'].get('name') != 'token_documents' or cfg['model'].get('name') != 'tiny_llama'
            or cfg['algorithm']['config'].get('arms') != list(ARMS)):
        raise ValueError('unknown data/model or incomplete three-arm declaration')
    if cfg.get('system', {}).get('config', {}).get('device') != 'cpu':
        raise ValueError('K pilot is CPU only')
    positions = cfg['algorithm']['config']['positions']
    docs = validate_documents(cfg['data']['config']['documents'], positions)
    model_cfg = cfg['model']['config']
    if model_cfg.get('tie_word_embeddings') is not True:
        raise ValueError('K pilot requires tied embeddings')
    if any(type(model_cfg.get(key)) is not int or model_cfg[key] < 1 for key in (
            'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
            'num_attention_heads', 'num_key_value_heads', 'max_position_embeddings')):
        raise ValueError('tiny model dimensions must be positive integers')
    if any(t >= model_cfg['vocab_size'] for doc in docs for t in doc['tokens']):
        raise ValueError('token exceeds tiny vocabulary')
    if max(positions) + 1 > model_cfg['max_position_embeddings']:
        raise ValueError('positions exceed model context')

    class KResidualAlgorithm(Algorithm):
        def run(self, data, model, system, tracker):
            del tracker
            return run_pilot(model.module, data.meta['documents'], self.config.setting('positions'),
                             system.assets_dir, seed=data.meta['seed'])

    DataRegistry.register('token_documents', SOURCE, _data_builder)
    ModelRegistry.register('tiny_llama', SOURCE, _model_builder)
    AlgorithmRegistry.register('eval', SOURCE, KResidualAlgorithm)
