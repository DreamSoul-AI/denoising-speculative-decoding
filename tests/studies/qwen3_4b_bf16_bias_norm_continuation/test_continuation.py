"""Continuation selects its own resource waiter without mutating norm workers."""


def test_continuation_wait_policy_is_injected_without_changing_the_norm_module(monkeypatch):
    from studies.qwen3_4b_bf16_bias_norm_continuation import continuation
    norm = continuation.norm_control
    original = norm.wait_for_admission
    events = []
    norm_wait = lambda assets, initial=False: events.append(('norm', initial)) or 'norm'
    paired_wait = lambda assets, initial=False: events.append(('paired', initial)) or 'paired'
    monkeypatch.setattr(norm.NormEval, 'admission_wait', staticmethod(norm_wait))
    monkeypatch.setattr(continuation.ContinuationEval, 'admission_wait', staticmethod(paired_wait))
    monkeypatch.setattr(continuation, 'check_refs', lambda: None)
    monkeypatch.setattr(norm.NormEval, 'run', lambda self, *a: self.admission_wait('fixture-assets', initial=False))
    plain = object.__new__(norm.NormEval)
    followup = object.__new__(continuation.ContinuationEval)
    assert followup.run(None, None, None, None) == 'paired'
    assert plain.run(None, None, None, None) == 'norm'
    assert events == [('paired', False), ('norm', False)]
    assert norm.wait_for_admission is original
