
from spec_mtp.masks import build_mask_provider
import json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import psutil,torch,yaml
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'studies/qwen3_4b_bf16_bias_norm_scan'))
import norm_control
from spec_mtp.model import HFCausalLM
import spec_mtp.decode as decoding
from studies._shared.structure.config import _convert
from studies._shared.structure.validation import _check_argmax
from transformers import AutoModelForCausalLM
from rpipe.structure.system.runtime import apply_runtime
from rpipe.structure.system.config import SystemConfig
out=ROOT/'.tmp/sd-memory-20261016'
manifest=json.loads((ROOT/'studies/qwen3_4b_bf16_five_methods/docs/prompts_manifest.json').read_text())
prompt=max(manifest['rendered_prompts'],key=lambda p:len(p['input_ids']))
assert prompt['question_id']==295 and len(prompt['input_ids'])==1500
assert psutil.virtual_memory().available >= 8060927256 + psutil.Process().memory_info().rss + 2147483648
norm_control.check_refs();torch.set_num_threads(1)
apply_runtime(123,SystemConfig.from_mapping({'config':{'deterministic':False}}))
original=AutoModelForCausalLM.from_pretrained
torch.cuda.reset_peak_memory_stats()
with patch.object(AutoModelForCausalLM,'from_pretrained',side_effect=lambda *a,**kw:original(*a,**dict(kw,device_map='cuda'))):
 model=HFCausalLM(r'D:\models\Qwen3-4B',local_files_only=True,torch_dtype='bfloat16',attn_implementation='eager')
assert model.dtype==torch.bfloat16 and model.device.type=='cuda'
assert model.embedding_table.data_ptr()==model.hf.get_output_embeddings().weight.data_ptr()
decl=yaml.safe_load((ROOT/'studies/qwen3_4b_bf16_bias_norm_scan/study.yaml').read_text())
setting=decl['axes']['algorithm.config'][1]
assert setting['direction']=='template' and setting['spec']['ema']['step_scale']==2
builder=decoding.build_mask_provider
info={}
def controlled(*a,**kw):return norm_control.observe_initial_bias(builder(*a,**kw),'template',info)
ids=torch.tensor([prompt['input_ids']],device='cuda')
with torch.inference_mode(),patch.object(decoding,'build_mask_provider',controlled),patch.object(decoding,'time',SimpleNamespace(perf_counter=lambda:0.)):
 result,stats=decoding.SpecDecoder(model,_convert(setting['spec']),_convert({'temperature':0.,'max_new_tokens':256,'stop_on_eos':False}),eos_ids=(), mask_policy=build_mask_provider(_convert(setting['spec']), model.embedding_table)).generate(ids)
assert stats['new_tokens']==256 and stats['prefill_time']==stats['decode_time']==0
checked=_check_argmax({'input_ids':prompt['input_ids'],'output_ids':result[0].tolist(),'stats':stats})
record={'complete':True,'capacity_only':True,'no_timing':True,'dtype':str(model.dtype),'question_id':295,'input_tokens':1500,'generated_tokens':256,'argmax_checked_tokens':checked,'eos_disabled_for_capacity_bound_only':True,'formal_tau_reported':False,'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'bias':info}
(out/'capacity_decode.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
