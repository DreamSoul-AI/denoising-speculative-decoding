import pytest
from test_window_observer import ROOT


def test_4b_pairing_retains_gains_losses_ties_and_main_calls(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'studies/output_space_4b_recheck'))
    from analyze_results import paired_summary
    rows=[{'cells':{'input':{'accepted':a,'model_calls':c},'zero':{'accepted':b,'model_calls':d}}}
          for a,b,c,d in ((3,5,8,6),(4,2,6,8),(1,1,4,4))]
    result=paired_summary(rows,'input','zero')
    assert result['accepted_total_delta']==0 and result['main_model_call_delta']==0
    assert (result['accepted_positive'],result['accepted_zero'],result['accepted_negative'])==(1,1,1)
    assert result['accepted_min_delta']==-2 and result['accepted_max_delta']==2 and result['n']==3


def test_missing_pair_is_rejected_not_imputed_zero(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'studies/output_space_4b_recheck'))
    from analyze_results import paired_summary
    with pytest.raises(KeyError):
        paired_summary([{'cells':{'input':{'accepted':1,'model_calls':3}}}],'input','zero')
