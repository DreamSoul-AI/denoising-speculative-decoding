"""Changed-temperature rare hits never establish original-condition reachability."""

import pytest
from test_window_observer import ROOT


def summary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from diagnose_rare_stress import event_summary
    return event_summary


def test_first_hit_uses_seed_identity_not_draw_offset(monkeypatch):
    result = summary(monkeypatch)([0, 11, 0, 11], [10000,10001,10002,10003], 11, .1)
    assert result['hits'] == 2 and result['first_hit_seed'] == 10001
    assert result['hit_seeds'] == [10001,10003]
    assert result['Clopper_Pearson_95pct'][0] < .5 < result['Clopper_Pearson_95pct'][1]


@pytest.mark.parametrize('draws,k,endpoint', [([0,0],0,0.),([11,11],2,1.)])
def test_zero_and_all_hit_interval_endpoints(monkeypatch, draws, k, endpoint):
    result = summary(monkeypatch)(draws,[10000,10001],11,.5)
    assert result['hits'] == k
    assert result['Clopper_Pearson_95pct'][0 if k == 0 else 1] == endpoint
    if k == 0:
        assert result['first_hit_seed'] is None


@pytest.mark.parametrize('draws,seeds,p', [([],[],.1),([11],[],.1),([11],[1],-1),([11],[1],float('nan'))])
def test_event_summary_rejects_invalid_observations(monkeypatch, draws, seeds, p):
    with pytest.raises(ValueError):
        summary(monkeypatch)(draws,seeds,11,p)


def test_stress_detection_does_not_raise_original_condition_power(monkeypatch):
    original = summary(monkeypatch)([0]*16,list(range(16)),11,1e-11)
    stress = summary(monkeypatch)([11]+[0]*15,list(range(16)),11,3e-5)
    assert stress['hits'] == 1 and original['hits'] == 0
    assert original['ideal_Bernoulli_probability_at_least_one'] < 2e-10
    assert original['Clopper_Pearson_95pct'][1] > 1e-11


@pytest.mark.parametrize('delta,expected', [(0,'falsified'),(-1,'falsified'),(2,'preserved'),(None,'infeasible_not_scored')])
def test_prompt_sign_prediction_includes_zero_and_infeasible(monkeypatch, delta, expected):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    monkeypatch.syspath_prepend(str(ROOT))
    from studies.output_space_cross_task.analyze_continuation import bounded_sign_prediction
    a = {'qid':242,'seed':126}
    if delta is not None:
        a['accepted_delta'] = delta
    results = bounded_sign_prediction([a,{'qid':244,'seed':126,'accepted_delta':-2}])
    assert results[0]['verdict'] == expected and results[1]['verdict'] == 'preserved'
