"""Bounded fit/dev/eval of Jason's frozen frontier equation with raw auditing."""

from dataclasses import asdict
import json
from pathlib import Path
import time
import sys

import torch

from spec_mtp.audit import audit_run, verify_model_files
from spec_mtp.model import HFCausalLM
from spec_mtp.structure.algorithm.decode.loop import generate
from spec_mtp.structure.algorithm.mask.frontier import FrontierMaskProvider
from spec_mtp.structure.api.decode_api import DecodeRequest, Prompt
from studies.trajectory_observation.structure.input_state import (
    encode_inputs, file_hash, write_json, pooled, ProvenanceSink,
)


def ridge(features, residual, regularization):
    x,y=features.double(),residual.double()
    penalty=x.new_full((x.shape[1],),regularization);penalty[-1]=0
    return torch.linalg.solve(x.T@x+torch.diag(penalty),x.T@y).float()


def coverage(probe, truth, features, gain, head, banned):
    hits=0
    for start in range(0,len(probe),4):
        stop=start+4
        target=(truth[start:stop]@head.T).argmax(-1)
        logits=(probe[start:stop]+features[start:stop]@gain)@head.T
        logits.scatter_(1,banned[start:stop,None],float('-inf'))
        hits+=int((logits.topk(14,dim=-1).indices==target[:,None]).any(-1).sum())
    return hits/len(probe)


def source_bridge(model, item, spec, eos, root, gain_path, expected):
    """Run the unchanged Jason decoder on identical IDs and a frozen gain."""
    import spec.decoding as source
    from core.config import _convert
    cfg=dict(spec)
    if gain_path is not None:
        cfg['frontier_gain']={'enabled':True,'path':str(gain_path),'online':False}
    choices=[];select=source.select_token
    def capture(logits,temperature,generator):
        token=select(logits,temperature,generator)
        choices.append((token,int(logits.argmax().item())))
        return token
    source.select_token=capture
    decoder=source.SpecDecoder(model,_convert(cfg),{'max_new_tokens':64,'temperature':0.,
        'stop_on_eos':True,'dump_states':True},eos)
    try:
        output,stats=decoder.generate(torch.tensor([item['input_ids']],device=model.device))
        tokens=output[0,len(item['input_ids']):].tolist();state=stats.pop('states')
        retained=[choices[0]];cursor=1;remaining=len(tokens)-1;accepted=0
        for count in state['acc'].tolist():
            n=min(count+1,remaining);retained.extend(choices[cursor:cursor+n]);accepted+=min(count,n)
            cursor+=count+1;remaining-=n
        assert remaining==0 and [x[0] for x in retained]==tokens and all(x==y for x,y in retained)
        result={'tokens':tokens,'A':accepted,'C':stats['decode_calls'],'S':len(tokens)-1,
                'retained_audit':retained,'passed':tokens==expected['tokens'] and accepted==expected['A']
                    and stats['decode_calls']==expected['C']}
        write_json(root/('source-bridge-'+('none' if gain_path is None else 'full')+'.json'),result)
        if not result['passed']:raise ValueError('unchanged source bridge differs from host adaptation')
    finally:
        source.select_token=select;decoder.recorder.close();decoder.gain.close()


def run(config):
    from transformers import AutoTokenizer
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.json',config)
    started=time.monotonic();model=None
    try:
        def check_source():
            members=json.loads(Path(config['source_manifest']).read_text(encoding='utf-8'))
            if any(file_hash(Path(config['source_root'])/p)!=h for p,h in members.items()):
                raise ValueError('frozen frontier source drift')
        check_source()
        source_dir=Path(config['jason_source'])
        sys.path.insert(0,str(source_dir/'src'))
        identity={p.relative_to(source_dir).as_posix():file_hash(p) for p in source_dir.rglob('*.py')}
        write_json(root/'jason-source-identity.json',identity)
        model_members=json.loads(Path(config['model_manifest']).read_text(encoding='utf-8'))
        verify_model_files(config['model'],model_members)
        tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
        inputs=encode_inputs(tokenizer,json.loads(Path(config['inputs']).read_text(encoding='utf-8')))
        write_json(root/'inputs.json',inputs)
        torch.set_num_threads(2)
        model=HFCausalLM(config['model'],torch_dtype='bfloat16',attn_implementation='eager',device_map='cuda:0')
        write_json(root/'model.json',model.describe())
        eos=model.hf.generation_config.eos_token_id;eos=tuple(eos if isinstance(eos,list) else [eos])
        sources={k:file_hash(config[k]) for k in ('source_manifest','model_manifest','inputs')}
        sources['config']=file_hash(root/'config.json')
        spec={'method':'esp','num_masks':1,'impl':'efficient','tree':{'mode':'static','branches':[14],'pruning':True}}
        all_rows=[]
        def execute(split,arm,gain=None):
            pairs=[];positions=[]
            for item in inputs[split]:
                if time.monotonic()-started>config.get('timeout',1400):raise TimeoutError('frontier batch deadline')
                provider=FrontierMaskProvider(model.embedding_table,model.hf.get_output_embeddings().weight,gain)
                path=root/f'{split}-{item["qid"]}-{arm}.jsonl'
                arm_sources={**sources,'gain':'none' if gain is None else file_hash(root/(arm+'.pt'))}
                req=DecodeRequest(Prompt(item['qid'],tuple(item['input_ids']),item['category']),spec,
                    max_new_tokens=32 if split!='eval' else 64,eos_ids=eos,run_id=root.name,sources=arm_sources)
                sink=ProvenanceSink(path,model_members,model.describe())
                try:result=generate(req,model,provider,sink)
                finally:sink.close()
                verdict=audit_run(path,arm_sources,expected_model_sources=model_members)
                if not all((verdict.source_integrity,verdict.argmax_coverage,verdict.argmax_correctness)):
                    raise ValueError('frontier T0 failed')
                pairs.extend(provider.pairs);positions.extend(provider.pair_positions)
                row={'split':split,'arm':arm,'qid':item['qid'],'category':item['category'],
                     **{k:result.stats['acceptance_metrics'][k] for k in ('A','C','S')},
                     'N':len(result.committed),'tokens':list(result.committed),'raw':path.name,'audit':asdict(verdict),
                     'sources':arm_sources}
                all_rows.append(row)
                with (root/'rows.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({k:row[k] for k in ('split','arm','qid','A','C','S')}),flush=True)
            if split!='eval':
                torch.save(pairs,root/(split+'-pairs.pt'));write_json(root/(split+'-pair-positions.json'),positions)
            return pairs
        fit=execute('fit','none');dev=execute('dev','none')
        if not fit or not dev:raise ValueError('no calibration pairs')
        def tensors(rows):return [torch.stack([r[i] for r in rows]).cuda() for i in range(3)]
        probe,truth,x=tensors(fit);vp,vt,vx=tensors(dev)
        # Features include the unpenalized intercept. E(b)=placeholder+u;
        # recover banned IDs from input embeddings for source cov@14 selection.
        # Positions' bonus IDs are selected from the raw commit stream below.
        banned=[]
        for item in inputs['dev']:
            raw=[json.loads(l) for l in (root/f'dev-{item["qid"]}-none.jsonl').read_text(encoding='utf-8').splitlines()]
            end=raw[-1];tokens=end['committed'];offset=0
            for tr in end['stats']['trace']:
                banned.append(tokens[offset]);offset+=len(tr['committed_tokens'])
        banned=torch.tensor(banned,device='cuda')
        if len(banned)!=len(vp):raise ValueError('dev bonus alignment mismatch')
        head=model.hf.get_output_embeddings().weight.detach().float()
        scores={};best_score=-1;best=None
        for lam in (.1,1.,10.):
            gain=ridge(x,truth-probe,lam)
            score=coverage(vp,vt,vx,gain,head,banned);scores[str(lam)]=score
            if score>best_score:best_score=score;best=gain;chosen=lam
        bias=torch.zeros_like(best);bias[-1]=(truth-probe).mean(0)
        torch.save({'K':best.cpu(),'ridge':chosen},root/'full.pt');torch.save({'K':bias.cpu()},root/'bias.pt')
        gain_hashes={a:file_hash(root/(a+'.pt')) for a in ('full','bias')}
        write_json(root/'fit.json',{'fit_pairs':len(fit),'dev_pairs':len(dev),'scores':scores,'ridge':chosen,
                                  'gain_sha256':gain_hashes,'equation':'z_probe + [E(bonus)-placeholder;1] @ K'})
        del head,probe,truth,x,vp,vt,vx,fit,dev,gain
        torch.cuda.empty_cache()
        verify_model_files(config['model'],model_members);check_source()
        for arm,gain in [('none',None),('bias',bias),('full',best)]:execute('eval',arm,gain)
        for arm,path in [('none',None),('full',root/'full.pt')]:
            expected=next(r for r in all_rows if r['split']=='eval' and r['arm']==arm)
            source_bridge(model,inputs['eval'][0],spec,eos,root,path,expected)
        if any(file_hash(source_dir/p)!=h for p,h in identity.items()):raise ValueError('Jason source drift')
        if any(file_hash(root/(a+'.pt'))!=h for a,h in gain_hashes.items()):raise ValueError('frozen gain drift')
        verify_model_files(config['model'],model_members);check_source()
        write_json(root/'summary.json',{'complete':True,'arms':{a:pooled([r for r in all_rows if r['split']=='eval' and r['arm']==a]) for a in ('none','bias','full')},
                                      'audited_tokens':sum(r['N'] for r in all_rows)})
    except BaseException as e:
        write_json(root/'failure.json',{'type':type(e).__name__,'error':str(e)});raise
    finally:
        if model is not None:model.close()
