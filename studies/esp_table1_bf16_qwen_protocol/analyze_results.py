"""Read fixed four-cell ESP evidence without choosing the best mask count."""
import hashlib
import json
from pathlib import Path
from studies._shared.structure.evidence import digest

CELLS = ((10,1,'auxiliary',None),(30,1,'primary',1.63),(60,1,'primary',1.74),(60,2,'control',1.74))


def summarize(rows):
    assert rows
    groups = {}
    for row in rows:
        assert row['decode_calls'] >= 0 and row['committed'] >= 0
        groups.setdefault(row['category'], []).append(row)
    def ratio(group):
        calls = sum(r['decode_calls'] for r in group)
        return sum(r['committed'] for r in group)/calls if calls else 0.
    return {'n':len(rows),'exact_count':sum(bool(r['exact']) for r in rows),
            'accepted':sum(r['accepted'] for r in rows),
            'main_calls':sum(r['decode_calls']+1 for r in rows),
            'pooled_tau':ratio(rows),'prompt_mean_tau':sum(r['committed']/max(r['decode_calls'],1) for r in rows)/len(rows),
            'category_mean_pooled_tau':sum(ratio(g) for g in groups.values())/len(groups)}


def ordered_cells(observed):
    result=[]
    for bc,k,role,target in CELLS:
        cell=dict(observed.get((bc,k),{}),BC=bc,num_masks=k,role=role,target=target)
        cell['pending']=(bc,k) not in observed
        for metric in ('pooled_tau','prompt_mean_tau','category_mean_pooled_tau'):
            value=cell.get(metric)
            cell[metric+'_target_delta']=value-target if value is not None and target is not None else None
        result.append(cell)
    return result


def main():
    study=Path(__file__).resolve().parent
    manifest=json.loads((study/'docs/prompts_manifest.json').read_text())
    prompts=manifest['rendered_prompts'];observed={}
    reused=study/'docs/reused_preflight.json'
    if reused.exists():
        reuse=json.loads(reused.read_text());source=study.parents[1]/reuse['source_evidence']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==reuse['evidence_sha256']
        for c in reuse['cells']:
            assert c['n']==len(prompts) and len(c['rows'])==len(prompts)
            assert c['dtype']=='torch.bfloat16' and c['actual_spec']['mask']=={'init':'mean_prompt','lam':.1,'update':True}
            assert all(p['input_ids']==e['input_ids'] for p,e in zip(prompts,c['raw_tokens']))
            cell=dict(c,provenance='reused frozen preflight; not rerun or re-signed')
            cell.update(summarize(c['rows']))
            observed[(c['BC'],c['actual_spec']['num_masks'])]=cell
    index=json.loads((study/'index.json').read_text())
    snapshot=json.loads((study/'docs/modelscope_snapshot.json').read_text())
    for entry in index['experiments']:
        spec=entry['factors']['algorithm.config']['spec'];run_id=entry['runs'][0]['id']
        assets=study/'runs'/run_id/'assets';raw_file=assets/'decode_observations.json'
        if not raw_file.exists():continue
        raw=json.loads(raw_file.read_text());input_file=assets/'decode_inputs.json';inputs=json.loads(input_file.read_text())
        assert raw['inputs_digest']==digest(inputs) and inputs['dtype']=='torch.bfloat16'
        assert inputs['backend']=='eager' and inputs['settings']['eval']['temperature']==0
        assert inputs['settings']['eval']['max_new_tokens']==256 and inputs['settings']['eval']['warmup'] is False
        assert inputs['settings']['spec']==spec and spec['tree']=={'mode':'dynamic','pruning':True}
        assert spec['mask']=={'init':'mean_prompt','lam':.1,'update':True}
        landed={name.replace('\\','/').rsplit('/',1)[-1]:sha for name,sha in inputs['files'].items()}
        assert all(landed[f['path']]==f['sha256'] for f in snapshot['files'])
        policy=json.loads((assets/'numeric_policy.json').read_text())
        assert policy=={'dtype':'torch.bfloat16','bf16_reduced_precision_reduction':True,'deterministic':False}
        out=raw['out'];assert len(out['rows'])==len(out['evidence'])==len(prompts)
        rows=[]
        for p,r,e in zip(prompts,out['rows'],out['evidence']):
            assert p['input_ids']==e['input_ids'] and r['spec_time']==r['ar_time']==0
            assert e['stats']['block_complexity']==spec['block_complexity']
            assert bool(r['exact_match'])==(e['output_ids']==e['ar_ids'])
            rows.append(dict(r,question_id=p['question_id'],exact=bool(r['exact_match'])))
        cell={'run_id':run_id,'actual_spec':spec,'dtype':inputs['dtype'],'model_digest':inputs['model_digest'],
              'runtime':inputs['runtime'],'numeric_policy':policy,'rows':rows,
              'raw_tokens':[{'input_ids':e['input_ids'],'output_ids':e['output_ids'],'ar_ids':e['ar_ids']} for e in out['evidence']],
              'raw_sha256':hashlib.sha256(raw_file.read_bytes()).hexdigest(),
              'inputs_sha256':hashlib.sha256(input_file.read_bytes()).hexdigest(),'provenance':'new native Run'}
        cell.update(summarize(rows));key=(spec['block_complexity'],spec['num_masks']);assert key not in observed
        observed[key]=cell
    cells=ordered_cells(observed)
    assert len({c['model_digest'] for c in cells if not c['pending']})<=1
    body={'scope':f'{len(prompts)} fixed SpecBench prompts; BC60/1 primary and BC60/2 control predeclared',
          'all_cells_observed':all(not c['pending'] for c in cells),
          'quality_passed':all(not c['pending'] and c['exact_count']==len(prompts) for c in cells),
          'no_best_mask_selection':True,'cells':cells}
    filename='protocol_evidence_v1.json' if body['all_cells_observed'] else 'protocol_pending.json'
    path=study/'docs'/filename;text=json.dumps(body,indent=2)+'\n'
    if path.exists() and body['all_cells_observed']:assert path.read_text()==text
    else:path.write_text(text,encoding='utf-8')
    lines=['# ESP Table 1 Qwen3 BF16 protocol report','',f"{len(prompts)} prompts; observed={body['all_cells_observed']}, token quality passed={body['quality_passed']}. BC60/1 is primary and BC60/2 is control, fixed before results.",'',
           '| BC/masks | role | exact/n | pooled τ | τ − target | prompt mean τ | mean − target | category mean τ | category − target |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in cells:
        values=['n/a' if key.endswith('_target_delta') and c['target'] is None else 'pending' if c.get(key) is None else f'{c[key]:.6f}' for key in ('pooled_tau','pooled_tau_target_delta','prompt_mean_tau','prompt_mean_tau_target_delta','category_mean_pooled_tau','category_mean_pooled_tau_target_delta')]
        exact='pending' if c['pending'] else f"{c['exact_count']}/{c['n']}"
        lines.append(f"| {c['BC']}/{c['num_masks']} | {c['role']} | {exact} | "+' | '.join(values)+' |')
    lines+=['','目标BC30=1.63，BC60双方均列相对1.74差值，不按τ选主配置。失配格τ是诊断值，不能算复现成功；预检不代替全480。','',
            '[PLAN](<../../../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-esp-table1-bf16-qwen-protocol-docs-plan>)引用G.2具体说明：Qwen3 BC60单mask/29节点，双mask/19节点为对照；保持动态树剪枝，BF16/T0/256/seed123，无测速/预热。','',
            'Mean mask按Eq5 λ=.1更新，与后续4B用户指定的ESP frozen mean不同。原执行冻结PLAN不回写，已有三格按哈希复用，原生exact门不放宽。',
            '',f'逐prompt计数、全部token及哈希见[{filename}]({filename})。全部480的差值须另列，未运行不能用预检外推。']
    (study/'docs/STUDY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'n':len(prompts),'observed':body['all_cells_observed'],'quality_passed':body['quality_passed'],
                      'cells':[{k:c.get(k) for k in ('BC','num_masks','role','pending','exact_count','pooled_tau','pooled_tau_target_delta')} for c in cells]}))


if __name__=='__main__':
    main()
