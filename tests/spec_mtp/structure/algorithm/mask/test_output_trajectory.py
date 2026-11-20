"""Untied readout, positional distance and unavailable bonus observation contracts."""

import pytest
import torch

from spec_mtp.model import HFCausalLM
from spec_mtp.structure.algorithm.mask.output_trajectory import OutputTrajectoryMaskProvider
from spec_mtp.structure.algorithm.mask import build_mask_provider
from spec_mtp.structure.algorithm.decode.tree import DraftTree
from spec_mtp.structure.algorithm.decode.loop import SpecDecoder, ar_generate


def test_output_readout_uses_actual_untied_head_and_is_pure():
    table = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    head = torch.tensor([[0., 1.], [1., 0.], [0., -1.]])
    p = OutputTrajectoryMaskProvider(2, table, head, kind='holt', alpha=.5, beta=.5)
    p.init_from_prompt(table[:2])
    p.init_hidden(torch.tensor([[1., 0.], [1., 0.]]))
    p.observe_position(torch.tensor([0., 1.]), 2, 1)
    tree = DraftTree(1, tokens=[2, 0], depths=[1, 1], parents=[0, 0])
    M,V,P = p.state.M.clone(),p.state.V.clone(),p.observed_position
    masks = p.tree_masks(tree, 3)
    for owner,depth in enumerate([0]+tree.depths):
        for slot in range(2):
            h=M+(3+depth+slot-P)*V
            h=p.scale*h/h.norm()
            torch.testing.assert_close(masks[owner,slot],torch.softmax(h@head.T,dim=-1)@table)
    p.tree_masks(tree,3).zero_()
    assert torch.equal(p.state.M,M) and torch.equal(p.state.V,V)
    assert p.observed_position==P and len(p.ledger)==2
    with pytest.raises(ValueError,match='next real committed'):
        p.observe_position(torch.ones(2),4,2)


@pytest.mark.parametrize('kind',['order0','difference','holt','kalman'])
@pytest.mark.parametrize('budget',[1,2,9])
def test_untied_decoder_output_ledger_and_complete_argmax(kind,budget):
    torch.manual_seed(8)
    model=HFCausalLM(tiny_config={'vocab_size':31,'hidden_size':16,'intermediate_size':32,
        'num_layers':1,'num_heads':2,'num_kv_heads':1,'tie_word_embeddings':False})
    spec={'method':'output_trajectory','num_masks':1,'impl':'efficient',
          'tree':{'mode':'static','branches':[14],'pruning':True},'trajectory':{'kind':kind}}
    provider=build_mask_provider(spec,model.embedding_table,output_head=model.hf.get_output_embeddings().weight)
    prompt=torch.tensor([[1,2,3]])
    evaluation={'max_new_tokens':budget,'stop_on_eos':False}
    out,stats=SpecDecoder(model,spec,evaluation,mask_policy=provider).generate(prompt)
    reference,_=ar_generate(model,prompt,evaluation)
    assert torch.equal(out,reference)
    assert stats['verification_argmax_audit']['checked_committed_tokens']==budget
    positions=[r['position'] for r in provider.ledger[1:]]
    assert positions==list(range(3,provider.observed_position+1))
    for trace in stats['trace']:
        observed=trace['output_hidden']
        assert observed['observed_block_rows']==[0]+trace['retained_accepted_nodes']
        assert observed['last_committed_position']-observed['last_observed_position'] in (0,1)
    assert provider.state.committed_steps==len(positions)
    assert stats['output_observation_ledger']==provider.ledger
