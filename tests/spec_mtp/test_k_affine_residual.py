"""Small independent checks for the separately registered K calibration Study."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def affine():
    path = Path(__file__).resolve().parents[2] / 'studies/k_affine_residual/structure/affine.py'
    spec = importlib.util.spec_from_file_location('test_k_affine_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_centered_ridge_matches_augmented_normal_equations(affine):
    rng = np.random.default_rng(7)
    x, r = rng.normal(size=(12, 3)) + 2, rng.normal(size=(12, 3)) + 3
    result = affine.fit_affine(x, r, .01)
    augmented = np.column_stack((x, np.ones(len(x))))
    penalty = np.diag([.01, .01, .01, 0.])
    oracle = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ r)
    np.testing.assert_allclose(result['a'], oracle[:-1].T, atol=1e-11)
    np.testing.assert_allclose(result['b'], oracle[-1], atol=1e-11)
    np.testing.assert_array_equal(result['bias_only'], r.mean(0))
    assert not np.allclose(result['b'], result['bias_only'])


def test_zero_residual_and_constant_features(affine):
    x = np.ones((5, 2))
    zero = affine.fit_affine(x, np.zeros_like(x), .001)
    np.testing.assert_array_equal(affine.predictions(x, x, zero), np.repeat(x[:,None], 3, axis=1))
    residual = np.array([[2., 3.]] * 5)
    fit = affine.fit_affine(x, residual, .001)
    np.testing.assert_array_equal(fit['a'], np.zeros((2, 2)))
    np.testing.assert_array_equal(fit['b'], [2., 3.])


@pytest.mark.parametrize('ridge', [0, -1, np.nan, np.inf])
def test_invalid_ridge_is_not_a_silent_fit(affine, ridge):
    with pytest.raises(ValueError, match='ridge'):
        affine.fit_affine(np.ones((2, 2)), np.ones((2, 2)), ridge)


def test_registered_documents_and_prefixes_are_disjoint(affine):
    manifest = affine.read(affine.STUDY / 'docs/data_manifest.json')
    affine.validate_documents(manifest['documents'], manifest['prefix_lengths'])
    assert manifest['pair_counts'] == {'fit': 64, 'holdout': 32, 'cross_dataset': 64}
    duplicate = [dict(doc) for doc in manifest['documents']]
    duplicate[-1]['document_sha256'] = duplicate[0]['document_sha256']
    with pytest.raises(ValueError, match='duplicate document'):
        affine.validate_documents(duplicate, manifest['prefix_lengths'])
    duplicate[-1]['document_sha256'] = 'distinct-doc'
    duplicate[-1]['token_ids'] = duplicate[0]['token_ids']
    with pytest.raises(ValueError, match='prefix crosses'):
        affine.validate_documents(duplicate, manifest['prefix_lengths'])


def test_eval_predictions_cannot_mutate_frozen_fit(affine, tmp_path):
    rng = np.random.default_rng(3)
    fit = affine.fit_affine(rng.normal(size=(10, 2)), rng.normal(size=(10, 2)), .001)
    path = tmp_path/'frozen.npz'
    np.savez(path, **fit)
    before = affine.sha(path)
    with np.load(path, allow_pickle=False) as frozen:
        affine.predictions(rng.normal(size=(50, 2)), rng.normal(size=(50, 2)), frozen)
    assert affine.sha(path) == before


def test_actual_hidden_target_is_same_committed_position_without_future(affine):
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM

    torch.manual_seed(123)
    module = HFCausalLM(tiny_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32,
                                        num_layers=1, num_heads=2, num_kv_heads=1,
                                        max_position_embeddings=32, tie_word_embeddings=True))
    tokens, length = [3, 7, 11, 13, 17, 19], 4
    x, zmask, ztrue = affine.extract_pair(module, tokens, length)
    changed_future = tokens[:length] + [31, 37]
    for left, right in zip((x, zmask, ztrue), affine.extract_pair(module, changed_future, length)):
        np.testing.assert_array_equal(left, right)
    with torch.inference_mode():
        embedded = module.embed(torch.tensor([tokens[:length]]))
        ordinary = module.forward(embedded, torch.arange(length).unsqueeze(0), last_hidden_only=True)
    np.testing.assert_array_equal(ztrue, ordinary.last_hidden_state[0, length-1].numpy())
    np.testing.assert_allclose(x, (embedded[0,-1]-embedded[0,:-1].mean(0)).numpy(), atol=1e-8)
    assert not np.array_equal(zmask, ztrue)
    assert not np.array_equal(ztrue, affine.extract_pair(module, [3, 7, 11, 21], length)[2])


@pytest.fixture
def synthetic_raw(affine, tmp_path):
    """Validation-only arrays: these are never reported as real model observations."""
    manifest = affine.read(affine.STUDY/'docs/data_manifest.json')
    rows = [dict(split=split, document_sha256=doc['document_sha256'], document_id=doc['document_id'],
                 length=length, position=length-1, committed_y=doc['token_ids'][length-1],
                 prefix_tokens=doc['token_ids'][:length])
            for split in affine.SPLITS for doc in manifest['documents'] if doc['split']==split
            for length in manifest['prefix_lengths']]
    rng = np.random.default_rng(9)
    x = rng.normal(size=(160,32)).astype(np.float32)
    mask = rng.normal(size=(160,32)).astype(np.float32)
    target = mask+x
    target[64:] = mask[64:]-x[64:]
    frozen = affine.fit_affine(x[:64], target[:64]-mask[:64], .001)
    predicted = affine.predictions(x, mask, frozen)
    np.savez(tmp_path/'frozen_k.npz', **frozen)
    np.savez(tmp_path/'pairs.npz', x=x, zmask=mask, ztrue=target, predicted=predicted,
             argmax_agreement=np.ones((160,3)), target_kl=np.zeros((160,3)))
    affine.write(tmp_path/'rows.json', rows)
    affine.write(tmp_path/'freeze.json',dict(frozen_k_sha256=affine.sha(tmp_path/'frozen_k.npz'),
                 evaluation_rows_extracted_at_freeze=0,ridge_lambda=.001))
    affine.write(tmp_path/'execution.json',dict(state_sha256_before='same',state_sha256_after='same',
                 raw_files={p.name:affine.sha(p) for p in tmp_path.iterdir()}))
    return tmp_path


def test_validation_accepts_a_complete_negative_research_result(affine, synthetic_raw):
    analysis = affine.analyze(synthetic_raw)
    assert analysis['valid']
    for split in ('holdout', 'cross_dataset'):
        assert all(value > 0 for value in analysis['splits'][split]['paired_full_minus_none'])


@pytest.mark.parametrize('file', ['frozen_k.npz', 'pairs.npz', 'rows.json'])
def test_raw_drift_is_rejected_without_resigning(affine, synthetic_raw, file):
    with (synthetic_raw/file).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError):
        affine.analyze(synthetic_raw)


def test_all_native_phase_exports_are_callable():
    from rpipe.flow.study import run_study_phase
    from types import SimpleNamespace
    from rpipe.structure.make.expand import load_study_yaml
    study = Path(__file__).resolve().parents[2] / 'studies/k_affine_residual'
    # Calling collect without execute must fail on real missing evidence, not a missing package export.
    ctx = SimpleNamespace(layout=SimpleNamespace(assets_dir=study/'nonexistent-test-assets'))
    with pytest.raises(FileNotFoundError, match='k_affine_binding'):
        run_study_phase(study, 'collect', ctx, load_study_yaml(study))
