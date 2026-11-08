"""cpu pure spies: continuation policy does not mutate norm workers."""


def test_continuation_wait_policy_is_injected_without_changing_the_norm_module(monkeypatch):
    from studies._shared.structure.algorithm import NormEval, ContinuationEval
    events = []
    norm_wait = lambda assets, initial=False: events.append(('norm', initial)) or 'norm'
    paired_wait = lambda assets, initial=False: events.append(('paired', initial)) or 'paired'
    monkeypatch.setattr(NormEval, 'run', lambda self, *a: self.admission_wait('fixture-assets', initial=False))
    plain = object.__new__(NormEval)
    followup = object.__new__(ContinuationEval)
    plain.admission_wait, followup.admission_wait = norm_wait, paired_wait
    assert followup.run(None, None, None, None) == 'paired'
    assert plain.run(None, None, None, None) == 'norm'
    assert events == [('paired', False), ('norm', False)]
    assert plain.control_kind == 'norm' and followup.control_kind == 'continuation'
