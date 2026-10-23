"""Postcommit alignment, held-out isolation, freeze order and three-arm contracts."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from spec_mtp.model import HFCausalLM
from studies.k_affine_residual.structure import recipe, evaluation
from studies.k_affine_residual.structure.correction import FrozenCorrection
from studies.k_affine_residual.flow.collect.run import read_report, run as collect
from studies.k_affine_residual.flow.summarize.run import run as summarize


def documents():
    return [{'id': key, 'split': split, 'tokens': tokens} for key, split, tokens in (
        ('fit', 'fit', [1,2,3,4]), ('hold', 'holdout', [5,6,7,8]),
        ('cross', 'cross_dataset', [9,10,11,12]))]


@pytest.fixture
def session():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        native = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, tie_word_embeddings=True, attn_implementation='eager')).eval()
    return HFCausalLM(module=native)


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_freeze_precedes_holdout_extraction_and_real_cpu_flow(session, tmp_path, backend):
    session.hf.set_attn_implementation(backend)
    calls = []
    def observe(model, docs, positions, identity):
        split = docs[0]['split']
        assert (tmp_path/'frozen_correction.json').exists() == (split != 'fit')
        calls.append(split)
        return recipe.extract_pairs(model, docs, positions, identity)
    out = recipe.run_pilot(session, documents(), [1,2], tmp_path, seed=123, extract=observe)
    assert calls == ['fit','holdout','cross_dataset']
    raw, report = read_report(tmp_path)
    assert report['complete'] and report['pairs'] == 4
    assert report['model']['device'] == 'cpu' and report['model']['parameters'] == 728
    ctx = SimpleNamespace(layout=SimpleNamespace(assets_dir=tmp_path), control=SimpleNamespace(seed=123),
        state={'execute':out, 'collected':{'metrics':{}}, 'result_draft':{'status':'succeeded'}})
    collect(ctx); summarize(ctx)
    assert ctx.state['result_draft']['k_affine_residual']['complete']
    assert len(ctx.state['collected']['metrics']) == 6
    with pytest.raises(FileExistsError):
        recipe.run_pilot(session, documents(), [1,2], tmp_path, seed=123)
    body=json.loads((tmp_path/'frozen_correction.json').read_text())
    frozen=FrozenCorrection(**body)
    pair=raw['pairs'][0]
    doc=next(d for d in raw['documents'] if d['id']==pair['document_id'])
    x,_,_=evaluation.pair_arrays(pair,doc)
    expected=torch.tensor(pair['committed_embedding'],dtype=torch.float64)-torch.tensor(
        pair['prefix_embeddings'],dtype=torch.float64).mean(0)
    torch.testing.assert_close(x[0],expected)
    for change in ('prefix','committed_token','next_token'):
        invalid=copy.deepcopy(pair)
        if change=='prefix': invalid['prefix']=[99]
        elif change=='committed_token': invalid['committed_token']+=1
        else: invalid['next_token']=1
        with pytest.raises(ValueError): evaluation.pair_arrays(invalid,doc)
    with pytest.raises(ValueError,match='incomplete'):
        evaluation.evaluate_pairs(raw['pairs'][:-1],raw['documents'][1:],[1,2],frozen)
    with pytest.raises(ValueError,match='overlap'):
        evaluation.evaluate_pairs(raw['fit_pairs'],raw['documents'][:1],[1,2],frozen)
    (tmp_path/'frozen_correction.json').write_text(json.dumps(body)+' ')
    with pytest.raises(ValueError,match='changed'):
        read_report(tmp_path)


@pytest.mark.parametrize('kind',['id','content','prefix','bool'])
def test_reject_data_leakage_and_invalid_positions(kind):
    docs=documents(); positions=[1,2]
    if kind=='id': docs[1]['id']='fit'
    elif kind=='content': docs[1]['tokens']=docs[0]['tokens'][:]
    elif kind=='prefix': docs[1]['tokens'][0]=docs[0]['tokens'][0]
    else: positions=[True]
    with pytest.raises(ValueError): evaluation.validate_documents(docs,positions)


def test_frozen_model_identity_rejects_actual_backend_change(session, tmp_path):
    def change_backend(model, docs, positions, identity):
        if docs[0]['split'] == 'holdout':
            model.hf.set_attn_implementation('sdpa')
        return recipe.extract_pairs(model, docs, positions, identity)
    with pytest.raises(ValueError, match='model changed during evaluation'):
        recipe.run_pilot(session, documents(), [1,2], tmp_path, seed=123, extract=change_backend)
    assert not (tmp_path/'latent_pairs.json').exists()


@pytest.mark.parametrize('raw_seed,control_seed', [(123,999), (True,1), (1,True)])
def test_collect_rejects_wrong_or_boolean_seed_before_state_write(session, tmp_path, raw_seed, control_seed):
    from studies.k_affine_residual.structure.correction import digest
    recipe.run_pilot(session, documents(), [1,2], tmp_path, seed=123)
    raw_path = tmp_path/'latent_pairs.json'
    raw = json.loads(raw_path.read_text())
    raw['seed'] = raw_seed
    raw_path.write_text(json.dumps(raw))
    ctx = SimpleNamespace(layout=SimpleNamespace(assets_dir=tmp_path),
        control=SimpleNamespace(seed=control_seed),
        state={'execute':{'raw_sha':digest(raw)}, 'collected':{'metrics':{}}})
    before = copy.deepcopy(ctx.state)
    with pytest.raises(ValueError, match='seed'):
        collect(ctx)
    assert ctx.state == before


@pytest.mark.parametrize('report_seed', [999, True, None])
def test_summarize_and_process_reject_report_seed_mismatch(report_seed, tmp_path, monkeypatch):
    from importlib import import_module
    process = import_module('studies.k_affine_residual.flow.process.run')
    report={'seed':report_seed, 'complete':True,'arms':['uncorrected','bias','full'],
        'frozen_sha':'a'*64, 'splits':{s:{'documents':1,
            'mse':{'uncorrected':1.,'bias':.8,'full':.5}} for s in ('holdout','cross_dataset')}}
    ctx=SimpleNamespace(control=SimpleNamespace(seed=123),
        state={'k_affine_residual':report,'result_draft':{}})
    with pytest.raises(ValueError, match='seed'):
        summarize(ctx)
    assert ctx.state['result_draft'] == {}
    result_path=tmp_path/'runs'/'run'
    result_path.mkdir(parents=True)
    (result_path/'result.json').write_text(json.dumps({'status':'succeeded','k_affine_residual':report}))
    monkeypatch.setattr(process, 'load_study_yaml', lambda study: {})
    monkeypatch.setattr(process, 'expand_patches', lambda declaration: [{'seed':123}])
    monkeypatch.setattr(process, 'load_index', lambda study:
        {'experiments':[{'runs':[{'id':'run','seed':123}]}]})
    with pytest.raises(ValueError, match='seed'):
        process.run(SimpleNamespace(scope='study',study_dir=tmp_path))


@pytest.mark.parametrize('change',['missing','duplicate','status','arm','nonfinite'])
def test_process_rejects_incomplete_or_corrupt_matrix(change):
    report={'seed':123,'complete':True,'arms':['uncorrected','bias','full'], 'frozen_sha':'a'*64,
        'splits':{s:{'documents':1,'mse':{'uncorrected':1.,'bias':.8,'full':.5}}
                  for s in ('holdout','cross_dataset')}}
    rows=[{'seed':123,'status':'succeeded','k_affine_residual':report}]
    if change=='missing': rows=[]
    elif change=='duplicate': rows*=2
    elif change=='status': rows[0]['status']='failed'
    elif change=='arm': report['arms'].pop()
    else: report['splits']['holdout']['mse']['full']=float('nan')
    with pytest.raises(ValueError): evaluation.check_matrix(rows,[123])
