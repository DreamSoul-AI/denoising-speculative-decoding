"""Pair new fixed input EMA with previously verified 4B outputs, no new output forwards."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'studies/output_space_4b_recheck_v2'))
from diagnose_output_windows import file_hash
from analyze_results import paired_summary


def main():
    import torch  # noqa: F401 - Windows torch-first
    import yaml
    from studies._shared.structure.sources import manifest,check_frozen
    from studies._shared.structure.validation import verify_evidence
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    study=Path(__file__).resolve().parent
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    decl=yaml.safe_load((study/'study.yaml').read_text(encoding='utf-8'))
    source=manifest()
    assert source['digest']=='670c744e0535dcad'
    check_frozen(study,decl,source)
    pcheck(study,decl)
    gate=read(study/'spec_mtp_study_check.json')
    process=read(study/'process.json')
    assert gate['complete'] and gate['passed'] and process['complete'] and process['spec_mtp']['passed']
    previous=read(study/'docs/preceding_results.json')
    pending=set(previous)
    groups=[]
    generated=forwards=0
    for exp in read(study/'index.json')['experiments']:
        assert len(exp['runs'])==1
        run=exp['runs'][0]
        folder=study/'runs'/run['id']
        assets=folder/'assets'
        cfg=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
        verify_evidence(cfg,assets)
        result=read(folder/'result.json')
        assert result['status']=='succeeded' and result['spec_mtp']['passed']
        spec=cfg['algorithm']['config']['spec']
        assert spec['method']=='ema_velocity' and spec['tree']=={'mode':'static','branches':[1,1],'pruning':False}
        inputs=read(assets/'decode_inputs.json')
        raw=read(assets/'decode_observations.json')['out']
        name='output_space_4b_recheck_v2' if len(raw['rows'])==26 else 'output_space_4b_holdout'
        assert name in pending
        pending.remove(name)
        original=ROOT/'studies'/name/'docs/execution_v1.json'
        assert file_hash(original)==previous[name]['execution_sha256']
        old=read(original)
        assert old['quality_passed'] and inputs['model_digest']==old['model_digest'] and inputs['runtime']==old['runs']['AR']['runtime']
        assert inputs['dtype']=='torch.float32' and inputs['backend']=='eager'
        assert len(raw['rows'])==len(old['paired_rows'])
        rows=[]
        for i,(oldrow,e,r) in enumerate(zip(old['paired_rows'],raw['evidence'],raw['rows'])):
            ar=old['runs']['AR']['out']['evidence'][i]
            assert e['input_ids']==ar['input_ids']==inputs['prompts'][i]['input_ids']
            assert e['output_ids']==e['ar_ids']==ar['output_ids'] and r['exact_match']==1
            assert e['stats']['prefill_time']==e['stats']['decode_time']==r['spec_time']==r['ar_time']==0
            assert not e['stats']['hit_context_limit']
            stats=e['stats']
            cells={**oldrow['cells'],'EMA':{'accepted':stats['accepted'],'model_calls':stats['prefill_calls']+stats['decode_calls'],
                'new_tokens':stats['new_tokens'],'accept_d1':stats['accept_by_depth'][1],'accept_d2':stats['accept_by_depth'][2],
                'audit_ar_calls':r['ar_calls']}}
            rows.append({**oldrow,'cells':cells})
            generated+=2*stats['new_tokens']
            forwards+=cells['EMA']['model_calls']+r['ar_calls']
        pairs=[paired_summary(rows,a,b) for a,b in (('input','EMA'),('EMA','zero'),('EMA','one'))]
        groups.append({'reused_study':name,'n':len(rows),'run_id':run['id'],'spec':spec,
            'model_digest':inputs['model_digest'],'runtime':inputs['runtime'],
            'inputs_sha256':file_hash(assets/'decode_inputs.json'),
            'raw_sha256':file_hash(assets/'decode_observations.json'),'reused_execution_sha256':file_hash(original),
            'rows':rows,'pairs':pairs,'raw':raw,
            'EMA_accepted':sum(r['cells']['EMA']['accepted'] for r in rows),
            'EMA_main_calls':sum(r['cells']['EMA']['model_calls'] for r in rows),
            'both_outputs_gain_over_EMA':all(p['accepted_total_delta']>0 for p in pairs[1:])})
    assert not pending
    groups.sort(key=lambda g:-g['n'])
    body={'quality_passed':True,'source_digest':source['digest'],'groups':groups,
        'new_generated_tokens_including_audits':generated,'new_generation_forwards_including_audits':forwards,
        'new_output_or_independent_AR_forwards':0,'teacher_forwards':0,'warmup_forwards':0,
        'timing_disabled':True,'tool_sha256':file_hash(Path(__file__)),
        'plan_sha256':file_hash(study/'docs/PLAN.md'),'scope':'Fixed historical EMA recipe, identical original input sets/outputs; no retuning or speed inference.'}
    path=study/'docs/execution_v1.json'
    text=json.dumps(body,indent=2,allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text(encoding='utf-8')==text
    else:
        path.write_text(text,encoding='utf-8')
    print(json.dumps({'groups':[{k:g[k] for k in ('reused_study','n','EMA_accepted','EMA_main_calls','pairs','both_outputs_gain_over_EMA')} for g in groups],
        'new_tokens':generated,'forwards':forwards}))


if __name__=='__main__':
    main()
