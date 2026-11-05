
import importlib.util

from pathlib import Path

import sys

import pytest



ROOT = Path(__file__).resolve().parents[2]

def test_tau_pairs_use_ratios_and_category_pooling(monkeypatch):
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    existing_readout = sys.modules.get('analyze_results')
    spec = importlib.util.spec_from_file_location('full_direction_analysis', ROOT / 'studies/_shared/structure/analyze_4b_full_directions.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert sys.modules.get('analyze_results') is existing_readout
    def row(qid, category, committed, decode_calls):
        return {'question_id': qid, 'category': category, 'accepted': committed // 2,
                'model_calls': decode_calls + 1, 'new_tokens': committed + 1,
                'committed': committed, 'decode_calls': decode_calls}
    first = [row(1, 'a', 1, 2), row(2, 'a', 6, 4), row(3, 'b', 2, 2)]
    second = [row(1, 'a', 2, 4), row(2, 'a', 4, 4), row(3, 'b', 4, 2)]
    assert m.compare(first, second)['tau'] == {'win': 1, 'tie': 1, 'loss': 1}
    assert m.compare(second, first)['tau'] == {'win': 1, 'tie': 1, 'loss': 1}
    assert m.category_mean(first) == pytest.approx(13 / 12)
    assert m.category_mean(first) != sum(r['committed'] / r['decode_calls'] for r in first) / 3
    for error in ('Foreign GPU work appeared: 42', 'Nonpreemptive GPU admission deferred'):
        m.require_resource_failure({'native_status': 'failed'}, error)
        for status in ('succeeded', 'running', 'pending'):
            with pytest.raises(AssertionError, match='Never replace'):
                m.require_resource_failure({'native_status': status}, error)
    with pytest.raises(AssertionError, match='argmax mismatch'):
        m.require_resource_failure({'native_status': 'failed'}, 'argmax mismatch')
