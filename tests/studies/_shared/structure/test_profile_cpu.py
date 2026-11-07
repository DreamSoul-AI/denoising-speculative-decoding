"""Host attribution labels and the original cProfile self-time formulas."""

from types import SimpleNamespace

from studies._shared.structure import profile_cpu


def test_component_buckets():
    assert profile_cpu.component('/x/spec_mtp/structure/algorithm/tree_attention.py') == 'tree attention'
    assert profile_cpu.component('/x/site-packages/torch/nn/modules/linear.py').startswith('model forward')
    assert profile_cpu.component('~', '<built-in method torch.cat>') == 'torch C calls (host self time)'
    assert profile_cpu.component('~', '<built-in method builtins.len>') == 'other builtins'
    assert profile_cpu.component('/x/y.py') == 'other Python'


def test_profile_statistics_use_self_time_instead_of_cumulative_time(monkeypatch):
    from studies._shared.structure import evaluate

    out = {'rows': [{'spec_time': 2., 'ar_time': 6., 'new_tokens': 10, 'decode_calls': 4}],
           'summaries': {'overall': {'tau': 2.5, 'exact_match_rate': 1.}},
           'evidence': [{'stats': {'verification_argmax_audit': {'passed': True}}}]}
    monkeypatch.setattr(evaluate, 'evaluate', lambda *args: out)
    stats = {('/src/tree_attention.py', 1, 'mask'): (1, 2, 2., 100., {}),
             ('/src/other.py', 2, 'helper'): (1, 3, 3., 200., {})}
    monkeypatch.setattr(profile_cpu.pstats, 'Stats', lambda *args: SimpleNamespace(stats=stats))
    args = SimpleNamespace(new_tokens=2, timing_repeats=1, top=2)
    result = profile_cpu.profile_variant('esp_static', args, object(), [], 'cuda:0')
    assert result['self_time_share'] == {'other Python': .6, 'tree attention': .4}
    assert result['spec_tok_per_s'] == 5. and result['ar_tok_per_s'] == 1.7
    assert result['spec_tok_per_s_unprofiled'] == 5.
    assert result['top_functions'][0]['self_s'] == 3. and result['top_functions'][0]['cum_s'] == 200.


def test_saved_profile_without_device_identity_retains_unknown_execution_scope():
    legacy = {'created': 'saved', 'machine': 'saved machine', 'torch': 'saved torch', 'threads': 1,
              'model': 'saved model', 'workload': 'saved workload', 'results': []}
    text = profile_cpu.markdown(legacy)
    assert 'Saved cProfile (execution scope unknown)' in text
    assert 'Rate scope: unknown for this saved report' in text
    assert 'Host cProfile of CUDA workload' not in text
