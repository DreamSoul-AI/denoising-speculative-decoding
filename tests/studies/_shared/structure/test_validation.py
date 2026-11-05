"""Pure declaration products and saved JSON correctness gates; no model execution."""

import csv
import hashlib
import json
import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from studies._shared.structure import validation

_VALIDATION_PROPERTY_SETTINGS = settings(max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture])
_FACTOR_NAMES = st.text(st.characters(codec='ascii', categories=['L', 'N']), min_size=1, max_size=8)


@_VALIDATION_PROPERTY_SETTINGS
@given(st.dictionaries(_FACTOR_NAMES, st.lists(st.integers(), min_size=1, max_size=4, unique=True), max_size=3))
def test_declared_factors_is_the_full_product(axes):
    factors = validation.declared_factors({'axes': axes})
    assert len(factors) == math.prod(len(values) for values in axes.values())
    assert len({json.dumps(factor, sort_keys=True) for factor in factors}) == len(factors)
    assert all(set(factor) == set(axes) and all(factor[key] in axes[key] for key in factor) for factor in factors)


@_VALIDATION_PROPERTY_SETTINGS
@given(st.floats(min_value=0, max_value=1, allow_nan=False))
def test_ar_rate_never_substitutes_for_missing_argmax_audit(tmp_path_factory, rate):
    assets = tmp_path_factory.mktemp('saved-audit-property')
    with (assets/'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['category', 'exact_match_rate'])
        writer.writeheader()
        writer.writerow({'category': 'overall', 'exact_match_rate': repr(rate)})
    (assets/'spec_mtp_sources.json').write_text('{"digest":"synthetic-saved-source"}', encoding='utf-8')
    (assets/'decode_inputs.json').write_text(json.dumps({'prompts': [{'input_ids': [1]}]}), encoding='utf-8')
    token_hash = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    stats = {'new_tokens': 1, 'accepted': 0, 'committed': 0, 'prefill_calls': 1, 'decode_calls': 0, 'trace': [],
        'verification_argmax_audit': {'passed': True, 'checked_committed_tokens': 1,
            'input_sha256': token_hash([1]), 'output_sha256': token_hash([1, 2]),
            'committed': [{'forward': 0, 'row': 0, 'chosen': 2, 'argmax': 2}]}}
    raw = {'out': {'rows': [{'exact_match': True}],
                   'evidence': [{'input_ids': [1], 'output_ids': [1, 2], 'ar_ids': [1, 2], 'stats': stats}]}}
    path = assets/'decode_observations.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    config = {'algorithm': {'config': {'task': 'esp_mtp', 'eval': {}}}}
    admitted = validation.evaluate_run(config, assets)
    assert admitted['passed'] and admitted['argmax_checked_tokens'] == 1
    del stats['verification_argmax_audit']
    path.write_text(json.dumps(raw), encoding='utf-8')
    rejected = validation.evaluate_run(config, assets)
    assert not rejected['passed'] and any('argmax' in reason for reason in rejected['reasons'])
