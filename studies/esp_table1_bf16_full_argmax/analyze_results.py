"""Read fixed four-cell ESP evidence without choosing the best mask count."""
import hashlib
import json
from pathlib import Path
from studies._shared.structure.evidence import digest
from argmax_audit import token_digest

CELLS = ((10,1,'primary',None),(30,1,'primary',1.63),(60,1,'primary',1.74),(60,2,'control',1.74))


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
            'AR_exact_rate':sum(bool(r['exact']) for r in rows)/len(rows),
            'argmax_checked_tokens':sum(r.get('argmax_checked_tokens',0) for r in rows),
            'argmax_all_passed':all(r.get('argmax_passed') is True for r in rows),
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
        audits=[json.loads(line) for line in (assets/'argmax_audit.jsonl').read_text().splitlines()]
        assert len(audits)==len(prompts)
        rows=[]
        for p,r,e in zip(prompts,out['rows'],out['evidence']):
            assert p['input_ids']==e['input_ids'] and r['spec_time']==r['ar_time']==0
            assert e['stats']['block_complexity']==spec['block_complexity']
            assert bool(r['exact_match'])==(e['output_ids']==e['ar_ids'])
            audit=e['stats']['verification_argmax_audit']
            assert audit==audits[len(rows)] and audit['passed'] is True
            assert audit['input_sha256']==token_digest(e['input_ids']) and audit['output_sha256']==token_digest(e['output_ids'])
            continuation=e['output_ids'][len(e['input_ids']):]
            assert audit['checked_committed_tokens']==len(continuation)==e['stats']['new_tokens']
            assert len(audit['committed'])==len(continuation)
            assert all(a['chosen']==a['argmax']==token for a,token in zip(audit['committed'],continuation))
            first=next((i-len(e['input_ids']) for i,(a,b) in enumerate(zip(e['output_ids'],e['ar_ids'])) if a!=b),min(len(e['output_ids']),len(e['ar_ids']))-len(e['input_ids'])) if e['output_ids']!=e['ar_ids'] else None
            rows.append(dict(r,question_id=p['question_id'],exact=bool(r['exact_match']),
                             first_new_difference_index=first,argmax_passed=True,argmax_checked_tokens=len(continuation)))
        result_file=assets.parent/'result.json'
        native=json.loads(result_file.read_text()) if result_file.exists() else {}
        native_status=native.get('status','pending finalisation')
        gate=native.get('spec_mtp') or {}
        cell={'run_id':run_id,'native_status':native_status,
              'native_validation_policy':gate.get('validation_policy'),'validator_sha256':gate.get('validator_sha256'),
              'gate_revalidation':{k:v for k,v in (gate.get('revalidation') or {}).items() if k in ('previous_result_sha256','previous_status','evidence_sha256','execute_rerun')},
              'argmax_audit_sha256':hashlib.sha256((assets/'argmax_audit.jsonl').read_bytes()).hexdigest(),'actual_spec':spec,'dtype':inputs['dtype'],'model_digest':inputs['model_digest'],
              'runtime':inputs['runtime'],'numeric_policy':policy,'rows':rows,
              'token_audits':[{'question_id':p['question_id'],'output_sha256':hashlib.sha256(json.dumps(e['output_ids']).encode()).hexdigest(),'AR_sha256':hashlib.sha256(json.dumps(e['ar_ids']).encode()).hexdigest(),'first_new_difference_index':next((i-len(e['input_ids']) for i,(a,b) in enumerate(zip(e['output_ids'],e['ar_ids'])) if a!=b), min(len(e['output_ids']),len(e['ar_ids']))-len(e['input_ids'])) if e['output_ids']!=e['ar_ids'] else None} for p,e in zip(prompts,out['evidence'])],
              'raw_sha256':hashlib.sha256(raw_file.read_bytes()).hexdigest(),
              'inputs_sha256':hashlib.sha256(input_file.read_bytes()).hexdigest(),'provenance':'new native Run'}
        cell.update(summarize(rows));key=(spec['block_complexity'],spec['num_masks']);assert key not in observed
        observed[key]=cell
    cells=ordered_cells(observed)
    assert len({c['model_digest'] for c in cells if not c['pending']})<=1
    body={'scope':f'{len(prompts)} fixed SpecBench prompts; BC60/1 primary and BC60/2 control predeclared',
          'all_cells_observed':all(not c['pending'] for c in cells),
          'quality_contract':'tau with full committed-token verification-forward argmax audit; AR diagnostic separately',
          'quality_passed':all(not c['pending'] and c['argmax_all_passed'] for c in cells),
          'AR_all_exact':all(not c['pending'] and c['exact_count']==len(prompts) for c in cells),
          'no_best_mask_selection':True,'cells':cells}
    formal_ready=body['all_cells_observed'] and all(c.get('native_status')=='succeeded' and c.get('native_validation_policy')=='verifier-argmax-v1' for c in cells)
    body['native_finalisation_complete']=formal_ready
    filename='protocol_evidence_v1.json' if formal_ready else 'protocol_pending.json'
    path=study/'docs'/filename;text=json.dumps(body,indent=2)+'\n'
    if path.exists() and formal_ready:assert path.read_text()==text
    else:path.write_text(text,encoding='utf-8')
    reference=f'[{filename}]({filename})' if formal_ready else f'本机临时文件 `{filename}`（未入Git）'
    lines=['# ESP Table 1 Qwen3 BF16 protocol report','',f"{len(prompts)} prompts; completed={sum(not c['pending'] for c in cells)}/4; 全四格验收={'passed' if body['quality_passed'] else 'pending'}. 已完成格的τ/argmax有效性单列，AR失配不否定该格。BC60/1 is primary and BC60/2 is control, fixed before results.",'',
           '| BC/masks | role | τ/argmax结果 | native状态 | AR exact/n (rate) | argmax checked/pass | pooled τ | τ − target | prompt mean τ | mean − target | category mean τ | category − target |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in cells:
        values=['n/a' if key.endswith('_target_delta') and c['target'] is None else 'pending' if c.get(key) is None else f'{c[key]:.6f}' for key in ('pooled_tau','pooled_tau_target_delta','prompt_mean_tau','prompt_mean_tau_target_delta','category_mean_pooled_tau','category_mean_pooled_tau_target_delta')]
        exact='pending' if c['pending'] else f"{c['exact_count']}/{c['n']} ({c['AR_exact_rate']:.2%})"
        audit='pending' if c['pending'] else f"{c['argmax_checked_tokens']}/{c['argmax_all_passed']}"
        valid='pending' if c['pending'] else 'valid' if c['argmax_all_passed'] else 'invalid'
        lines.append(f"| {c['BC']}/{c['num_masks']} | {c['role']} | {valid} | {c.get('native_status','pending')} | {exact} | {audit} | "+' | '.join(values)+' |')
    lines+=['','目标BC30=1.63，BC60双方均列相对1.74差值，不按τ选主配置。AR独立审计失配单列，不能宣称与独立AR一致；不阻止τ计算。τ验收要求全480、全部提交token的验证前向argmax自检通过，不以预检替代结果。','',
            '聚合定义：pooled τ=Σcommitted/Σdecode_calls；prompt mean τ为逐条committed/max(decode_calls,1)的等权均值；category mean τ先按原始category各算pooled τ，再对类别等权。类别样本数不等，不能把该权重称为论文已确认的任务权重；分组见[prompt清单](prompts_manifest.json)。','',
            '计数定义：committed为解码阶段实际提交数，包含bonus，经过EOS/预算裁剪，不含prefill首token。argmax审计覆盖完整新输出（含prefill首token），因此全480条的argmax checked比Σcommitted多480；实际model calls=Σdecode_calls+480。accepted为候选接受数，不能替代committed或τ。','',
            '[PLAN](<../../../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-esp-table1-bf16-full-argmax-docs-plan>)引用G.2具体说明：Qwen3 BC60单mask/29节点，双mask/19节点为对照；保持动态树剪枝，BF16/T0/256/seed123，无测速/预热。','',
            'Mean mask按Eq5 λ=.1更新，与后续4B用户指定的ESP frozen mean不同。四格全部新Run，旧冻结PLAN不回写；旧AR门失败result归档；仅重新验收后记录真实native状态及新validator SHA，不重跑execute。新增argmax必须全过；独立AR差异不阻塞native summarize/process。',
            '',f'逐prompt计数、AR首分叉位置和自检覆盖及哈希见{reference}。完整原始tokens/argmax选择记录/Run仅在本机保留，未运行格不能用预检外推。']
    if formal_ready and body['quality_passed']:
        from plot_results import write_plot
        write_plot(body,study/'docs/figures/tau_acceptance.png')
        matched=all(round(c['pooled_tau'],2)==c['target'] for c in cells if c['role']=='primary' and c['target'] is not None)
        lines+=['',f"结论：主配置pooled τ按两位小数与论文目标{'一致' if matched else '不一致'}。数学自检与原生生命周期通过不等于论文τ数值复现成功；三种聚合和双mask对照全部保留，不事后择优。",
                '', '[原生process验收证据](native-process-revalidation-v1.json)记录四格成功状态及旧process归档哈希。',
                '', '![Full480 tau and accepted draft counts](figures/tau_acceptance.png)',
                '',f'图：同一四格480条，三种τ聚合及真实accepted draft总数（EOS/预算裁剪前）；黑菱形为论文目标。来源[{filename}]({filename})，AR一致率另列，不含测速。']
    (study/'docs/STUDY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'n':len(prompts),'observed':body['all_cells_observed'],'quality_passed':body['quality_passed'],
                      'cells':[{k:c.get(k) for k in ('BC','num_masks','role','pending','exact_count','pooled_tau','pooled_tau_target_delta')} for c in cells]}))


if __name__=='__main__':
    main()
