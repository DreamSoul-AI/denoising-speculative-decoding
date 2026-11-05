"""Execute only the current tiny CUDA engineering protocol into a new external root."""
import argparse
import json
from pathlib import Path
import sys

from spec_mtp.structure.artifact.provenance import file_sha256
from studies.smoke.structure import validation as protocol
from studies._shared.structure.sources import package_files
from studies._shared.flow.write.run import publish_new_completion


def source_snapshot():
    files = [Path(__file__),Path(protocol.__file__),Path(publish_new_completion.__code__.co_filename)]
    return {'package':package_files(),'tools':{str(p.resolve()):file_sha256(p) for p in files}}


def run(args):
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    if not __debug__: raise RuntimeError('smoke engineering requires Python assertions enabled')
    if args.device != 'cuda:0': raise ValueError('smoke probe requires explicit cuda:0; no CPU fallback')
    if not isinstance(args.version,str) or not args.version.strip(): raise ValueError('a new engineering version is required')
    if (args.compare is None) != (args.compare_sha256 is None): raise ValueError('compare requires its actual SHA256')
    aliases={'plan':Path(args.plan).absolute()}
    if args.compare is not None: aliases['compare']=Path(args.compare).absolute()
    inputs={name:path.resolve(strict=True) for name,path in aliases.items()}
    hashes={name:file_sha256(path) for name,path in inputs.items()}
    for name in inputs:
        if hashes[name] != getattr(args,name+'_sha256'): raise ValueError(name+' differs from its explicit SHA256')
    previous=json.loads(inputs['compare'].read_bytes()) if 'compare' in inputs else None
    if previous is not None and previous.get('protocol') != protocol.PROTOCOL:
        raise ValueError('old CPU or incompatible smoke protocols cannot compare to the CUDA engineering probe')
    if previous is not None:
        if previous.get('schema') != 1 or previous.get('summary') != protocol.summarize_cells(previous.get('cells')):
            raise ValueError('comparison requires a complete schema1 report with reconciled summary')
        if inputs['compare'].name != 'probe.json': raise ValueError('comparison requires the completed probe.json member')
        folder=inputs['compare'].parent
        for role,member in (('compare_complete','probe_complete.json'),('compare_header','header.json'),('compare_cells','cells.jsonl')):
            aliases[role]=folder/member
            inputs[role]=aliases[role].resolve(strict=True)
            inputs[role].relative_to(folder)
            hashes[role]=file_sha256(inputs[role])
        marker=json.loads(inputs['compare_complete'].read_bytes())
        header=json.loads(inputs['compare_header'].read_bytes())
        if (marker.get('schema') != 1 or marker.get('protocol') != protocol.PROTOCOL
                or marker.get('version') != previous.get('version') or marker.get('summary') != previous.get('summary')
                or marker.get('probe_sha256') != hashes['compare'] or marker.get('header_sha256') != hashes['compare_header']
                or marker.get('cells_sha256') != hashes['compare_cells']
                or any(header.get(key) != previous.get(key) for key in ('schema','protocol','version','identity','sources'))):
            raise ValueError('comparison requires the completed probe/header/cells SHA256 binding')
    checkout=Path(__file__).resolve().parents[4]
    output=validate_study_root(validate_output_root(args.output_root,checkout,*(p.parent for p in inputs.values())))
    if output.exists() or output.is_symlink(): raise FileExistsError('smoke requires a new external output root')
    sources=source_snapshot()
    def verify():
        if (any(aliases[name].resolve(strict=True) != path or file_sha256(path) != hashes[name] for name,path in inputs.items())
                or source_snapshot() != sources): raise ValueError('smoke inputs or current sources changed')
    import torch
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no CPU fallback')
    device=torch.device(args.device)
    identity={'device':str(device),'device_name':torch.cuda.get_device_name(device),'dtype':'torch.float32','backend':'eager',
        'torch':torch.__version__,'cuda':torch.version.cuda,'model_seed':123,'prompt_seed':0,'cell_seed_base':123,
        'prompt_rng':'torch.Generator(cpu), four length32 integer fixtures; transfer to CUDA',
        'model_rng':'torch.manual_seed(123); construct on CUDA','decode_rng':'torch.Generator(cuda:0), fresh seed123+i per cell',
        'model_config':protocol.MODEL_CONFIG,'variants':protocol.VARIANTS,'temperatures':list(protocol.TEMPERATURES),
        'max_new_tokens':32,'stop_on_eos':False,'plan_sha256':hashes['plan'],
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}
    torch.manual_seed(123)
    generator=torch.Generator(device='cpu').manual_seed(0)
    prompts=[torch.randint(0,256,(1,32),generator=generator,device='cpu').to(device) for _ in range(4)]
    identity['input_ids']=[p[0].tolist() for p in prompts]
    if previous is not None and previous.get('identity') != identity: raise ValueError('incompatible smoke device/config/RNG/input identity')
    verify();output.mkdir(parents=True,exist_ok=False)
    header={'schema':1,'protocol':protocol.PROTOCOL,'version':args.version,'identity':identity,'sources':sources,
        'inputs':{name:{'path':str(path),'sha256':hashes[name]} for name,path in inputs.items()},
        'scope':'new CUDA engineering observations; no old CPU bitwise baseline or sampling distribution claim',
        'trace_origin':'native SpecDecoder.stats.trace; no independent source-line/cache observer'}
    (output/'header.json').write_text(json.dumps(header,indent=2)+'\n',encoding='utf-8')
    from spec_mtp.structure.system.logger import Logger
    logger=Logger(output/'assets');logger.info('smoke CUDA device='+str(device)+' name='+identity['device_name'])
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder,ar_generate
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    from studies._shared.structure.config import _convert
    from studies._shared.structure.evidence import model_digest
    model=None;cells={}
    try:
        with torch.device(device): model=HFCausalLM(tiny_config=protocol.MODEL_CONFIG,attn_implementation='eager')
        if model.device != device or model.dtype != torch.float32: raise ValueError('smoke model is not actual CUDA FP32')
        logger.info('smoke model device='+str(model.device)+' dtype='+str(model.dtype)+' input_device='+str(prompts[0].device))
        header['model_digest']=model_digest(model)
        if previous is not None and previous.get('model_digest') != header['model_digest']:
            raise ValueError('incompatible smoke actual model identity')
        with (output/'cells.jsonl').open('x',encoding='utf-8') as stream:
            for variant,spec in protocol.VARIANTS.items():
                for temperature in protocol.TEMPERATURES:
                    cfg=_convert({'max_new_tokens':32,'temperature':temperature,'stop_on_eos':False})
                    for i,prompt in enumerate(prompts):
                        decoder=SpecDecoder(model,_convert(spec),cfg,generator=torch.Generator(device=device).manual_seed(123+i),
                            mask_policy=build_mask_provider(_convert(spec),model.embedding_table))
                        generated,stats=decoder.generate(prompt)
                        ar,_=ar_generate(model,prompt,cfg,generator=torch.Generator(device=device).manual_seed(123+i))
                        key=f'{variant}|T={temperature}|p={i}'
                        cells[key]=protocol.cell_observation(prompt[0].tolist(),generated[0].tolist(),ar[0].tolist(),stats,temperature)
                        stream.write(json.dumps({'cell':key,**cells[key]})+'\n');stream.flush()
        summary=protocol.summarize_cells(cells)
        if header['model_digest'] != model_digest(model): raise ValueError('smoke model changed during validation')
        report={**header,'cells':cells,'summary':summary}
        if previous is not None: protocol.compare_reports(previous,report)
    finally:
        primary=sys.exc_info()[1]
        if model is not None:
            try: model.close()
            except BaseException as cleanup:
                if primary is None: raise
                setattr(primary,'cleanup_errors',(*getattr(primary,'cleanup_errors',()),cleanup))
    verify()
    (output/'probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    verify()
    complete={'schema':1,'protocol':protocol.PROTOCOL,'version':args.version,'summary':summary,
        'header_sha256':file_sha256(output/'header.json'),'probe_sha256':file_sha256(output/'probe.json'),
        'cells_sha256':file_sha256(output/'cells.jsonl'),'readout_only':False,'legacy_CPU_comparison':False}
    publish_new_completion(output/'probe_complete.json',json.dumps(complete,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','output-root'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('plan-sha256','device','version'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--compare',type=Path);parser.add_argument('--compare-sha256')
    args=parser.parse_args(argv);run(args);return 0


if __name__=='__main__':raise SystemExit(main())
