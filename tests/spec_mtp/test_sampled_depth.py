import pytest
from test_window_observer import ROOT


@pytest.mark.parametrize('first_a,first_b,target,second,eos,nodes_a,nodes_b,expected', [
    (10,10,10,20,[],[1],[1,2],True),
    (10,11,10,20,[],[1],[1,2],False),
    (10,10,10,20,[],[],[1,2],False),
    (10,10,10,None,[],[1],[1,2],False),
    (10,10,10,20,[10],[1],[1,2],False),
    (10,10,10,20,[],[1,2],[1,2],False),
    (10,10,11,20,[],[1],[1,2],False),
])
def test_sampled_second_depth_legality(monkeypatch, first_a, first_b, target, second, eos, nodes_a, nodes_b, expected):
    monkeypatch.syspath_prepend(str(ROOT))
    from studies.output_space_cross_task.analyze_sampled_depth import legal_second_flip
    assert legal_second_flip(first_a,first_b,target,second,eos,nodes_a,nodes_b) is expected
