"""Frontier correction identity and delayed same-position labels."""
import torch
import pytest

from spec_mtp.model import HFCausalLM
from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
from spec_mtp.structure.algorithm.mask.providers import ESPMaskProvider
from spec_mtp.structure.algorithm.mask.frontier import FrontierMaskProvider


def test_zero_gain_matches_esp_and_labels_only_the_next_root():
    torch.manual_seed(2)
    model=HFCausalLM(tiny_config={'vocab_size':31,'hidden_size':16,'intermediate_size':32,
        'num_layers':1,'num_heads':2,'num_kv_heads':1})
    spec={'num_masks':1,'impl':'efficient','tree':{'mode':'static','branches':[14],'pruning':True}}
    prompt=torch.tensor([[1,2,3]]);evaluation={'max_new_tokens':12,'stop_on_eos':False}
    baseline=ESPMaskProvider(1,model.embedding_table)
    p=FrontierMaskProvider(model.embedding_table,model.hf.get_output_embeddings().weight,torch.zeros(17,16))
    a,sa=SpecDecoder(model,spec,evaluation,mask_policy=baseline).generate(prompt)
    b,sb=SpecDecoder(model,spec,evaluation,mask_policy=p).generate(prompt)
    assert torch.equal(a,b) and sa['retained_accepted']==sb['retained_accepted']
    assert len(p.pairs)==sb['decode_calls']
    assert sb['verification_argmax_audit']['passed']


def test_frontier_feature_is_same_position_bonus_minus_actual_placeholder():
    table=torch.eye(3);K=torch.arange(12).reshape(4,3).float()/10
    p=FrontierMaskProvider(table,table,K);p.init_from_prompt(table[:2])
    h=torch.tensor([1.,2.,3.]);m=torch.tensor([.2,.3,.4]);b=table[2]
    z=p.frontier_logits(h,m,b,torch.zeros(3));u=torch.cat([b-m,torch.ones(1)])
    torch.testing.assert_close(z,h+u@K)
    true=torch.tensor([4.,5.,6.]);p.label_root(true)
    assert len(p.pairs)==1 and p.pending is None
    torch.testing.assert_close(p.pairs[0][1],true)
    torch.testing.assert_close(p.pairs[0][2],u)
    p.frontier_logits(h,m,b,torch.zeros(3),position=7)
    with pytest.raises(ValueError,match='pending probe position'):
        p.label_root(true,position=8)
    p.label_root(true,position=7)
    assert p.pair_positions[-1]==(7,7)


def test_source_ridge_leaves_intercept_unpenalized():
    from studies.trajectory_observation.structure.frontier_matrix import ridge
    x=torch.zeros(6,4);x[:,-1]=1
    y=torch.tensor([3.,-2.]).expand(6,2)
    gain=ridge(x,y,10.)
    torch.testing.assert_close(gain[:-1],torch.zeros(3,2))
    torch.testing.assert_close(gain[-1],torch.tensor([3.,-2.]))


def test_source_coverage_never_accepts_a_pruned_repeat():
    from studies.trajectory_observation.structure.frontier_matrix import coverage
    h=torch.eye(17)[2:3];x=torch.ones(1,18);gain=torch.zeros(18,17)
    assert coverage(h,h,x,gain,torch.eye(17),torch.tensor([2]))==0
    assert coverage(h,h,x,gain,torch.eye(17),torch.tensor([3]))==1
