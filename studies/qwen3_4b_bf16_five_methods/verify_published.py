"""CPU reaggregation of published counts; no native replay or model/token audit."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

from analyze_results import CASES, METHODS, compare_rows, read, summarize


def verify(path):
    path = Path(path)
    body = read(path)
    assert body['complete'] and body['argmax_is_gate_AR_is_diagnostic'] and body['timing_disabled']
    assert set(body['cells']) == set(CASES)
    reference = [(r['question_id'], r['category']) for r in body['cells']['AR']['rows']]
    assert len(reference) == len(set(reference)) == 480
    for name, cell in body['cells'].items():
        assert cell['native_status'] == 'succeeded'
        assert [(r['question_id'], r['category']) for r in cell['rows']] == reference
        assert summarize(cell['rows'], name == 'AR') == cell['totals']
        if name != 'AR':
            assert all(r['argmax_checked_tokens'] == r['new_tokens'] == r['committed'] + 1
                       and r['model_calls'] == r['decode_calls'] + 1 for r in cell['rows'])
    expected = [(f'{bc}/{a}', f'{bc}/{b}') for bc in (30, 60) for a, b in combinations(METHODS, 2)]
    assert [(p['first'], p['second']) for p in body['pairs']] == expected
    for pair in body['pairs']:
        actual = compare_rows(body['cells'][pair['first']]['rows'], body['cells'][pair['second']]['rows'])
        assert actual == {k: v for k, v in pair.items() if k not in ('first', 'second')}
    return {'scope': __doc__, 'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cells_checked': len(body['cells']), 'pairs_checked': len(body['pairs']),
            'cells': {name: cell['totals'] for name, cell in body['cells'].items()},
            'pairs': [{k: v for k, v in pair.items() if k != 'rows'} for pair in body['pairs']]}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).parent / 'docs/paired_evidence_v1.json'), indent=2, allow_nan=False))
