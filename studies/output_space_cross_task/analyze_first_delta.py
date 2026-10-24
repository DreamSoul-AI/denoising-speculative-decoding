"""Test first local acceptance direction against final sampled-path net change.

    python studies/output_space_cross_task/analyze_first_delta.py
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from diagnose_output_windows import file_hash, TRACE_KEYS


def path_accounting(left, right, left_accepted, right_accepted):
    a, b = ([{k: t[k] for k in TRACE_KEYS} for t in trace] for trace in (left, right))
    assert sum(len(t['accepted_nodes']) for t in a) == left_accepted
    assert sum(len(t['accepted_nodes']) for t in b) == right_accepted
    am, bm = ({t['root_position']: t for t in trace} for trace in (a, b))
    assert len(am) == len(a) and len(bm) == len(b)
    assert [t['root_position'] for t in a] == sorted(am) and [t['root_position'] for t in b] == sorted(bm)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None:
        assert a == b and left_accepted == right_accepted
        first_root, first_delta = None, 0
    else:
        assert a[first]['root_position'] == b[first]['root_position']
        first_root = a[first]['root_position']
        first_delta = len(b[first]['accepted_nodes']) - len(a[first]['accepted_nodes'])
    common = sorted(am.keys() & bm.keys())
    changes = [{'root': p, 'delta': len(bm[p]['accepted_nodes']) - len(am[p]['accepted_nodes'])}
               for p in common if len(bm[p]['accepted_nodes']) != len(am[p]['accepted_nodes'])]
    a_only, b_only = sorted(am.keys() - bm.keys()), sorted(bm.keys() - am.keys())
    common_delta = sum(c['delta'] for c in changes)
    exclusive_delta = sum(len(bm[p]['accepted_nodes']) for p in b_only) - sum(len(am[p]['accepted_nodes']) for p in a_only)
    net = right_accepted - left_accepted
    assert common_delta + exclusive_delta == net
    sign = lambda v: (v > 0) - (v < 0)
    return {'first_divergence_root': first_root, 'first_accepted_delta': first_delta, 'final_accepted_delta': net,
        'direction_prediction_preserved': sign(first_delta) == sign(net), 'common_root_changes': changes,
        'common_root_delta': common_delta, 'exclusive_root_delta': exclusive_delta,
        'self_only_roots': a_only, 'graft_only_roots': b_only,
        'first_self': None if first is None else a[first], 'first_graft': None if first is None else b[first]}


def main():
    from studies._shared.structure.sources import manifest
    study = Path(__file__).resolve().parent
    paths = {'tool': Path(__file__), 'plan': study / 'docs/FIRST_DELTA_PLAN.md',
        'summary': study / 'docs/sampling_expansion_analysis_v1.json',
        'new': ROOT / '.tmp/sampling-expansion-20261015-v1/diagnosis.json',
        'old': ROOT / '.tmp/sampling-paths-20261015-v1/diagnosis.json'}
    binding = {k: file_hash(p) for k, p in paths.items()}
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    new, old, summary = (read(paths[k]) for k in ('new', 'old', 'summary'))
    assert all(r['quality_passed'] for r in (new, old, summary))
    assert manifest()['digest'] == new['source_digest'] == '670c744e0535dcad'
    rows = []
    for development, report in ((False, new), (True, old)):
        for unit in report['units']:
            if unit['selection']['intervention_root'] is None:
                rows.append({'qid': unit['qid'], 'seed': unit['seed'], 'development_old_unit': development,
                             'status': 'infeasible_not_scored'})
                continue
            cases = {c['label']: c['replay'] for c in unit['cases']}
            self, graft = cases['self'], cases['graft']
            facts = path_accounting(self['trace'], graft['trace'], self['stats']['accepted'], graft['stats']['accepted'])
            rows.append({'qid': unit['qid'], 'seed': unit['seed'], 'development_old_unit': development, **facts})
    rows.sort(key=lambda r: (r['qid'], r['seed']))
    counters = [r for r in rows if not r['development_old_unit'] and r.get('direction_prediction_preserved') is False]
    assert all(file_hash(p) == binding[k] for k, p in paths.items())
    result = {'source_binding': binding, 'quality_passed': True,
        'H7_first_local_sign_predicts_final_sign_in_new_units': 'falsified' if counters else 'not_falsified',
        'new_counterexamples': [{'qid': r['qid'], 'seed': r['seed'], 'first_root': r['first_divergence_root'],
            'first_delta': r['first_accepted_delta'], 'final_delta': r['final_accepted_delta']} for r in counters],
        'rows': rows, 'new_model_forwards': 0, 'new_sampling_draws': 0,
        'scope': 'Explanatory trace accounting on fixed completed units, not a tensor causal decomposition.'}
    target = study / 'docs/first_delta_analysis_v1.json'
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == text
    else:
        target.write_text(text, encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('H7_first_local_sign_predicts_final_sign_in_new_units', 'new_counterexamples')}))


if __name__ == '__main__':
    main()
