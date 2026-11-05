"""Pure declaration products and saved JSON correctness gates; no model execution."""

import csv
import hashlib
import json
import math

import pytest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from spec_mtp.structure.artifact.observation import SpecMtpGateError
from studies._shared.structure import sources, validation
from studies._shared.structure.evidence import digest

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


def test_exact_match_one_passes(tmp_path, saved_decode_config, saved_decode_assets):
    report = validation.evaluate_run(saved_decode_config(), saved_decode_assets(tmp_path))
    assert report['passed'] and report['exact_match_rate'] == 1.0 and report['sources_digest'] == sources.manifest()['digest']


@pytest.mark.parametrize('kwargs, reason', [
    ({'overall': False}, 'no single overall row'),
    ({'with_sources': False}, 'spec_mtp_sources'),
])
def test_gate_failures(tmp_path, kwargs, reason, saved_decode_config, saved_decode_assets):
    report = validation.evaluate_run(saved_decode_config(), saved_decode_assets(tmp_path, **kwargs))
    assert not report['passed']
    assert any(reason in item for item in report['reasons'])


def test_unaudited_esp_run_fails(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    path = assets / 'decode_observations.json'
    raw = json.loads(path.read_text())
    del raw['out']['evidence'][0]['stats']['verification_argmax_audit']
    path.write_text(json.dumps(raw))
    report = validation.evaluate_run(saved_decode_config(), assets)
    assert not report['passed'] and 'argmax' in report['reasons'][0]


def test_ar_baseline_needs_summary(tmp_path, saved_decode_config, saved_decode_assets):
    empty = tmp_path / 'empty'
    empty.mkdir()
    (empty / 'spec_mtp_sources.json').write_text('{"digest": "d"}')
    assert validation.evaluate_run(saved_decode_config('ar_baseline'), empty)['reasons'] == ['missing summary.csv']
    assert validation.evaluate_run(saved_decode_config('ar_baseline'), saved_decode_assets(tmp_path))['passed']


def test_alignment_probe_needs_its_csv(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    assert validation.evaluate_run(saved_decode_config('alignment_probe'), assets)['reasons'] == ['missing alignment_metrics.csv']
    (assets / 'alignment_metrics.csv').write_text('layer\n1\n')
    assert validation.evaluate_run(saved_decode_config('alignment_probe'), assets)['passed']


def test_empty_rate_is_missing(tmp_path, saved_decode_config, saved_decode_assets):
    report = validation.evaluate_run(saved_decode_config(), saved_decode_assets(tmp_path, rate=''))
    assert report['passed'] and report['reported_exact_match_rate'] is None and report['exact_match_rate']==1


def test_two_overall_rows_are_rejected(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    with (assets / 'summary.csv').open('a', newline='', encoding='utf-8') as handle:
        handle.write('overall,1.0,5\n')
    assert 'summary.csv has no single overall row' in validation.evaluate_run(saved_decode_config(), assets)['reasons']


def test_task_name_is_case_insensitive(tmp_path, saved_decode_config, saved_decode_assets):
    assert validation.evaluate_run(saved_decode_config('ESP_MTP'), saved_decode_assets(tmp_path))['task'] == 'esp_mtp'


def test_declared_factors_without_axes():
    assert validation.declared_factors({}) == [{}]


def test_legacy_axis_seed_is_not_an_experiment_factor():
    assert validation.declared_factors({'axes': {'seed': [1, 2], 'x': ['a']}}) == [{'x': 'a'}]


def test_cuda_device_alias_does_not_look_like_config_drift(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    path = assets / 'decode_inputs.json'
    inputs = json.loads(path.read_text())
    inputs['device'] = 'cuda:0'
    inputs['settings']['eval']['device'] = 'cuda'
    path.write_text(json.dumps(inputs))
    raw_path = assets / 'decode_observations.json'
    raw = json.loads(raw_path.read_text())
    raw['inputs_digest'] = digest(inputs)
    raw_path.write_text(json.dumps(raw))
    validation.verify_evidence(saved_decode_config(), assets)


def test_declared_factors_follow_mapping_axis_override_contract():
    base = {'algorithm': {'mode': 'eval', 'config': {
        'version': 'base-v1', 'eval': {'max_new_tokens': 100}, 'mask': 'frozen_mean'}}}
    study = {'fixed': {'algorithm': {'config': {'version': 'fixed-v2'}}},
             'axes': {'algorithm.config': [
                 {'task': 'esp_mtp', 'mask': 'gamma_last'}, {'task': 'ar_baseline'}]},
             'seeds': [123, 456]}
    expected = [
        {'algorithm.config': {'version': 'base-v1', 'eval': {'max_new_tokens': 100},
                              'mask': 'gamma_last', 'task': 'esp_mtp'}},
        {'algorithm.config': {'version': 'base-v1', 'eval': {'max_new_tokens': 100},
                              'mask': 'frozen_mean', 'task': 'ar_baseline'}},
    ]
    # A whole algorithm.config axis replaces that fixed subtree, then merges
    # into the base. Keep the official make semantics, rather than inventing
    # recursive base/fixed/axis composition here.
    assert validation.declared_factors(study, base) == expected


def test_declared_factors_bind_inherited_conditions():
    # Same declared patch, different frozen base: effective factors must differ.
    study = {'axes': {'algorithm.config': [{'task': 'ar_baseline'}]}, 'seeds': [123]}
    first = {'algorithm': {'config': {'eval': {'max_new_tokens': 100}}}}
    second = {'algorithm': {'config': {'eval': {'max_new_tokens': 50}}}}
    assert validation.declared_factors(study, first) != validation.declared_factors(study, second)


@pytest.mark.parametrize('key,value', [('passed',False), ('checked_committed_tokens',5),
    ('input_sha256','bad'), ('output_sha256','bad'), ('committed',[])])
def test_argmax_audit_required_even_when_ar_exact(tmp_path,key,value, saved_decode_config, saved_decode_assets, corrupt_saved_argmax):
    assets = saved_decode_assets(tmp_path)
    corrupt_saved_argmax(assets,key,value)
    assert not validation.evaluate_run(saved_decode_config(),assets)['passed']
    with pytest.raises(SpecMtpGateError,match='argmax'):
        validation.verify_evidence(saved_decode_config(),assets)


@pytest.mark.parametrize('key,value', [('chosen',99),('argmax',99),('forward',3),('row',3)])
def test_argmax_token_and_owner_tampering_fails(tmp_path,key,value, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    path = assets / 'decode_observations.json'
    raw = json.loads(path.read_text())
    raw['out']['evidence'][0]['stats']['verification_argmax_audit']['committed'][0][key] = value
    path.write_text(json.dumps(raw))
    assert not validation.evaluate_run(saved_decode_config(),assets)['passed']


def test_jsonl_argmax_audit_must_match_raw(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    (assets/'argmax_audit.jsonl').write_text('{}\n')
    assert not validation.evaluate_run(saved_decode_config(),assets)['passed']
