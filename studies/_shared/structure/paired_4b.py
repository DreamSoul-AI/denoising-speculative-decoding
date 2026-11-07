"""Pure scalar reaggregation of the fixed 4B paired protocols."""
import statistics

if not __debug__:
    raise RuntimeError('saved paired readout requires Python assertions enabled')

def paired_summary(rows, first, second):
    accepted = [r['cells'][second]['accepted']-r['cells'][first]['accepted'] for r in rows]
    calls = [r['cells'][second]['model_calls']-r['cells'][first]['model_calls'] for r in rows]
    assert rows
    return {'first':first,'second':second,'n':len(rows),'accepted_total_delta':sum(accepted),
        'accepted_mean_delta':statistics.mean(accepted),'accepted_stdev_delta':statistics.stdev(accepted) if len(rows)>1 else 0.,
        'accepted_min_delta':min(accepted),'accepted_max_delta':max(accepted),
        'accepted_positive':sum(x>0 for x in accepted),'accepted_zero':sum(x==0 for x in accepted),
        'accepted_negative':sum(x<0 for x in accepted),'main_model_call_delta':sum(calls)}


def recompute_pairs(prompts, runs):
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
    return {'n': len(rows), 'question_ids': [r['question_id'] for r in rows],
        'paired_rows': rows, 'paired_summaries': pairs, 'totals': totals,
        'H1_both_output_orders_gain_over_input': all(p['accepted_total_delta']>0 for p in pairs[:2]),
        'AR_main_model_calls': sum(r['AR_main_model_calls'] for r in rows),
        'total_generated_tokens_including_audits': generated,
        'total_generation_model_forwards_including_audits': forwards}


def recompute_control(old, raw, inputs, spec, run_id, name):
    generated=forwards=0
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
    group = {'reused_study':name,'n':len(rows),'run_id':run_id,'spec':spec,
        'model_digest':inputs['model_digest'],'runtime':inputs['runtime'],
        'rows':rows,'pairs':pairs,'raw':raw,
        'EMA_accepted':sum(r['cells']['EMA']['accepted'] for r in rows),
        'EMA_main_calls':sum(r['cells']['EMA']['model_calls'] for r in rows),
        'both_outputs_gain_over_EMA':all(p['accepted_total_delta']>0 for p in pairs[1:])}
    return group, generated, forwards
