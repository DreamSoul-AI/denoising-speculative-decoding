"""CPU pure saved-count, trace and comparison contracts; no numerical model."""
import copy
import hashlib
import json
import pytest
from studies.smoke.structure import validation as protocol


def tiny_stats():
    prompt,output=[1],[1,2,3]
    stats={'prefill_calls':1,'decode_calls':1,'accepted':0,'committed':1,'new_tokens':2,
        'trace':[{'root_position':1,'cache_before':1,'accepted_nodes':[],'committed_tokens':[3],'stopped':True,'cache_after':None}],
        'attempt_by_depth':[0,1,0,0],'accept_by_depth':[0,0,0,0],'hit_context_limit':0}
    digest=lambda ids:hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    stats['verification_argmax_audit']={'passed':True,'checked_committed_tokens':2,'input_sha256':digest(prompt),
        'output_sha256':digest(output),'committed':[{'forward':0,'row':0,'chosen':2,'argmax':2},
                                                  {'forward':1,'row':0,'chosen':3,'argmax':3}]}
    return prompt,output,stats


@pytest.mark.parametrize('temperature',[0.,.7,1.])
def test_cell_preserves_native_trace_and_separates_ar_diagnostic_from_argmax(temperature):
    prompt,output,stats=tiny_stats()
    result=protocol.cell_observation(prompt,output,[1,9,9],stats,temperature)
    assert result['exact_match'] is False and result['trace']==stats['trace']
    assert result['argmax_checked_tokens']==(2 if temperature==0 else None)


@pytest.mark.parametrize('fault',['trace','coverage','owner','prefix'])
def test_t0_cell_rejects_trace_or_argmax_contract_damage(fault):
    prompt,output,stats=tiny_stats()
    if fault=='trace':stats['trace'][0]['committed_tokens']=[8]
    elif fault=='coverage':stats['verification_argmax_audit']['checked_committed_tokens']=1
    elif fault=='owner':stats['verification_argmax_audit']['committed'][1]['row']=3
    else:output[0]=9
    with pytest.raises((ValueError,RuntimeError)):protocol.cell_observation(prompt,output,[1,2,3],stats,0.)


@pytest.mark.parametrize('fault',['none','CPU','device','rng','cells','model','missing_model','schema','summary'])
def test_comparison_requires_frozen_cuda_identity_before_any_bitwise_claim(fault):
    cells={f'{variant}|T={temperature}|p={i}':{'new_tokens':32,'exact_match':True,
           'argmax_checked_tokens':32 if temperature==0 else None}
           for variant in protocol.VARIANTS for temperature in protocol.TEMPERATURES for i in range(4)}
    report={'schema':1,'protocol':protocol.PROTOCOL,'identity':{'device':'cuda:0','rng':'cuda percell123+i'},
            'model_digest':'a'*64,'cells':cells,'summary':protocol.summarize_cells(cells)}
    previous=copy.deepcopy(report)
    if fault=='CPU':previous={'device':'cpu','cells':{'fixture':1}}
    elif fault=='device':previous['identity']['device']='cpu'
    elif fault=='rng':previous['identity']['rng']='other'
    elif fault=='cells':previous['cells']['fixture']=2
    elif fault=='model':previous['model_digest']='b'*64
    elif fault=='missing_model':previous.pop('model_digest')
    elif fault=='schema':previous.pop('schema')
    elif fault=='summary':previous.pop('summary')
    if fault=='none':protocol.compare_reports(previous,report)
    else:
        with pytest.raises(ValueError):protocol.compare_reports(previous,report)


@pytest.mark.parametrize('fault',['none','commit','count','device','seed'])
def test_historical_cpu_figure_keeps_original_934dac3_saved_identity(historical_file,fault):
    path=historical_file('studies/smoke/docs/delivery_validation.json')
    data=json.loads(path.read_bytes());before=path.read_bytes()
    if fault=='commit':data['implementation_commit']='0'*40
    elif fault=='count':data['cpu'][0]['accepted']+=1
    elif fault=='device':data['cpu'][0]['device']='cuda:0'
    elif fault=='seed':data['cpu'][0]['seed']=124
    if fault=='none':
        figure=protocol.historical_cpu_figure(data)
        assert [patch.get_height() for patch in figure.axes[0].patches]==[28,31,8,27,31,8]
        figure.clear()
    else:
        with pytest.raises(ValueError):protocol.historical_cpu_figure(data)
    assert path.read_bytes()==before
