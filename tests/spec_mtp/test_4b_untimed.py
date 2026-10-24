
from spec_mtp.masks import build_mask_provider
from types import SimpleNamespace
import time
from unittest.mock import patch
import pytest
import torch
from test_window_observer import ROOT, model as model, spec
import spec_mtp.decode as decoding
from studies._shared.structure.config import _convert


@pytest.mark.parametrize('order',[None,0,1])
def test_untimed_decoder_preserves_tokens_semantics_and_calls(model, order):
    prompt=torch.tensor([[1,2,3,4]])
    cfg=_convert({'max_new_tokens':16,'temperature':0.,'stop_on_eos':False})
    original=decoding.time
    expected,stats=decoding.SpecDecoder(model,spec(order),cfg, mask_policy=build_mask_provider(spec(order), model.embedding_table)).generate(prompt)
    with patch.object(decoding,'time',SimpleNamespace(perf_counter=lambda:0.)):
        actual,untimed=decoding.SpecDecoder(model,spec(order),cfg, mask_policy=build_mask_provider(spec(order), model.embedding_table)).generate(prompt)
        assert untimed['prefill_time'] == untimed['decode_time'] == 0
        reference,ar_stats=decoding.ar_generate(model,prompt,cfg)
        assert ar_stats['ar_time'] == 0
    assert torch.equal(actual,expected) and torch.equal(actual,reference)
    assert {k:v for k,v in stats.items() if k not in ('prefill_time','decode_time')} == {
        k:v for k,v in untimed.items() if k not in ('prefill_time','decode_time')}
    assert decoding.time is original is time and time.perf_counter() > 0


@pytest.mark.parametrize('failure',[False,True])
def test_study_wrapper_restores_clock_on_success_and_failure(monkeypatch, failure):
    monkeypatch.syspath_prepend(str(ROOT/'studies/output_space_4b_recheck'))
    import untimed_4b
    original=decoding.time
    def parent(*args):
        assert decoding.time.perf_counter() == 0 and decoding.time is not time
        if failure:
            raise RuntimeError('injected failure')
        return {'ok':True}
    monkeypatch.setattr(untimed_4b.SpecMtpEvalAlgorithm,'run',parent)
    algo=untimed_4b.UntimedEval.__new__(untimed_4b.UntimedEval)
    algo.config=SimpleNamespace(setting=lambda *args:{'warmup':False})
    if failure:
        with pytest.raises(RuntimeError,match='injected failure'):
            algo.run(None,None,None,None)
    else:
        assert algo.run(None,None,None,None) == {'ok':True}
    assert decoding.time is original


def test_direct_cuda_materialization_preserves_dtype_kwargs_and_restores_loader(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'studies/output_space_4b_recheck'))
    import untimed_4b
    from transformers import AutoModelForCausalLM
    seen={}
    def original(*args,**kwargs):
        seen.update(kwargs)
        return 'model'
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',original)
    def factory(*args):
        return AutoModelForCausalLM.from_pretrained('cache',dtype=torch.float32,local_files_only=True)
    monkeypatch.setattr(untimed_4b,'_build_hf',factory)
    assert untimed_4b.build_cuda(None,None) == 'model'
    assert seen == {'dtype':torch.float32,'local_files_only':True,'device_map':'cuda'}
    assert AutoModelForCausalLM.from_pretrained is original
