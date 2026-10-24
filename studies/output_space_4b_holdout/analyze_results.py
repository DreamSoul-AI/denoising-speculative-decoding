"""Recompute paired counts from native 4B Runs; no generation or speed analysis."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from diagnose_output_windows import file_hash


def paired_summary(rows, first, second):
    accepted = [r['cells'][second]['accepted']-r['cells'][first]['accepted'] for r in rows]
    calls = [r['cells'][second]['model_calls']-r['cells'][first]['model_calls'] for r in rows]
    assert rows
    return {'first':first,'second':second,'n':len(rows),'accepted_total_delta':sum(accepted),
        'accepted_mean_delta':statistics.mean(accepted),'accepted_stdev_delta':statistics.stdev(accepted) if len(rows)>1 else 0.,
        'accepted_min_delta':min(accepted),'accepted_max_delta':max(accepted),
        'accepted_positive':sum(x>0 for x in accepted),'accepted_zero':sum(x==0 for x in accepted),
        'accepted_negative':sum(x<0 for x in accepted),'main_model_call_delta':sum(calls)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    import torch  # noqa: F401 - Windows torch-first
    import yaml
    from studies._shared.structure.sources import manifest,check_frozen
    from studies._shared.structure.validation import verify_evidence
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    study=args.study.resolve()
    decl=yaml.safe_load((study/'study.yaml').read_text(encoding='utf-8'))
    live=manifest()
    assert live['digest']=='670c744e0535dcad'
    check_frozen(study,decl,live)
    pcheck(study,decl)
    quality=read(study/'spec_mtp_study_check.json')
    process=read(study/'process.json')
    assert quality['passed'] and quality['complete'] and process['complete'] and process['spec_mtp']['passed']
    prompts=read(study/'docs/prompts_manifest.json')['rendered_prompts']
    index=read(study/'index.json')
    assert len(index['experiments']) == 4
    runs={}
    for exp in index['experiments']:
        assert len(exp['runs']) == 1
        run=exp['runs'][0]
        folder=study/'runs'/run['id']
        assets=folder/'assets'
        cfg=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
        verify_evidence(cfg,assets)
        result=read(folder/'result.json')
        inputs=read(assets/'decode_inputs.json')
        raw=read(assets/'decode_observations.json')['out']
        assert result['status']=='succeeded' and result['spec_mtp']['passed']
        assert inputs['dtype']=='torch.float32' and inputs['backend']=='eager'
        assert cfg['algorithm']['config']['eval']['warmup'] is False
        assert len(prompts)==len(inputs['prompts'])==len(raw['rows'])==len(raw['evidence'])
        for p,q,e in zip(prompts,inputs['prompts'],raw['evidence']):
            assert p['input_ids']==q['input_ids']==e['input_ids']
            assert not e['stats'].get('hit_context_limit',False)
        settings=cfg['algorithm']['config']
        spec=settings.get('spec',{})
        case='AR' if settings['task']=='ar_baseline' else ('input' if spec['method']=='esp' else 'zero' if spec['output']['order']==0 else 'one')
        for e in raw['evidence']:
            if case=='AR':
                assert e['stats']['ar_time']==0
            else:
                assert e['stats']['prefill_time']==e['stats']['decode_time']==0
        runs[case]={'run_id':run['id'],'raw_sha256':file_hash(assets/'decode_observations.json'),
            'inputs_sha256':file_hash(assets/'decode_inputs.json'),'model_digest':inputs['model_digest'],
            'runtime':inputs['runtime'],'spec':spec,'out':raw,'gate':result['spec_mtp']}
    assert set(runs)=={'input','zero','one','AR'}
    assert len({r['model_digest'] for r in runs.values()})==1
    assert all(r['runtime']==runs['AR']['runtime'] for r in runs.values())
    rows=[]
    for i,prompt in enumerate(prompts):
        ar=runs['AR']['out']['evidence'][i]
        cells={}
        for method in ('input','zero','one'):
            e=runs[method]['out']['evidence'][i]
            row=runs[method]['out']['rows'][i]
            assert e['output_ids']==e['ar_ids']==ar['output_ids'] and row['exact_match']==1
            assert row['spec_time']==row['ar_time']==0
            stats=e['stats']
            cells[method]={'accepted':stats['accepted'],'decode_calls':stats['decode_calls'],
                'prefill_calls':stats['prefill_calls'],'model_calls':stats['prefill_calls']+stats['decode_calls'],
                'audit_ar_calls':row['ar_calls'],'new_tokens':stats['new_tokens'],
                'accept_d1':stats['accept_by_depth'][1],'accept_d2':stats['accept_by_depth'][2]}
        rows.append({'question_id':prompt['question_id'],'category':prompt['category'],'generator_seed':123+i,
            'new_tokens':ar['stats']['ar_new_tokens'],'AR_main_model_calls':ar['stats']['ar_calls'],'cells':cells})
    pairs=[paired_summary(rows,a,b) for a,b in (('input','zero'),('input','one'),('zero','one'))]
    generated=sum(r['new_tokens'] for r in rows)*7
    forwards=sum(r['AR_main_model_calls'] for r in rows)+sum(c['model_calls']+c['audit_ar_calls'] for r in rows for c in r['cells'].values())
    totals={m:{k:sum(r['cells'][m][k] for r in rows) for k in ('accepted','model_calls','decode_calls','new_tokens','accept_d1','accept_d2')} for m in ('input','zero','one')}
    body={'quality_passed':True,'source_digest':live['digest'],'model_digest':runs['AR']['model_digest'],
        'question_ids':[r['question_id'] for r in rows],'n':len(rows),'paired_rows':rows,'paired_summaries':pairs,
        'totals':totals,'H1_both_output_orders_gain_over_input':all(p['accepted_total_delta']>0 for p in pairs[:2]),
        'AR_main_model_calls':sum(r['AR_main_model_calls'] for r in rows),'total_generated_tokens_including_audits':generated,
        'total_generation_model_forwards_including_audits':forwards,'warmup_forwards':0,'teacher_forwards':0,
        'decoding_timing_disabled':True,'timing_zeros_are_sentinels_not_measurements':True,
        'runs':runs,'study_quality':quality,'tool_sha256':file_hash(Path(__file__)),
        'plan_sha256':file_hash(study/'docs/PLAN.md'), 'snapshot_sha256':file_hash(study/'docs/modelscope_snapshot.json'),
        'manifest_sha256':file_hash(study/'docs/prompts_manifest.json'),
        'scope':'Matched two-slot chain, fixed prompts; no speed, significance or exact reproduction of unlocated prior output-space4B gains.'}
    target=study/'docs/execution_v1.json'
    text=json.dumps(body,indent=2,allow_nan=False)+'\n'
    if target.exists():
        assert target.read_text(encoding='utf-8')==text
    else:
        target.write_text(text,encoding='utf-8')
    print(json.dumps({'n':len(rows),'totals':totals,'pairs':pairs,'AR_calls':body['AR_main_model_calls'],
        'generated_tokens':generated,'forwards':forwards,'both_orders_gain':body['H1_both_output_orders_gain_over_input']}))


if __name__=='__main__':
    main()
