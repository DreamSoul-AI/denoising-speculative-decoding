"""Property tests (hypothesis) for the pure helpers written for the RPipe dependency."""

import csv
import json
import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from studies._shared.flow import cli
from studies._shared.flow import checks
from studies._shared.structure.config import apply_variant
from studies._shared.structure.paper_variants import PAPER_VARIANTS
from studies._shared.structure.smoke_variants import VARIANTS
from studies._shared.structure import sources

PROFILE = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
names = st.text(st.characters(codec='ascii', categories=['L', 'N']), min_size=1, max_size=8)
digests = st.text('0123456789abcdef', min_size=4, max_size=8)
file_maps = st.dictionaries(names.map(lambda s: f'{s}.py'), digests, max_size=6)


# ------------------------------------------------------------ sources

@PROFILE
@given(file_maps)
def test_digest_ignores_insertion_order(files):
    assert sources.files_digest(files) == sources.files_digest(dict(reversed(list(files.items()))))


@PROFILE
@given(file_maps, file_maps)
def test_changed_files_is_the_symmetric_difference_of_entries(before, now):
    changed = sources.changed_files({'files': before}, {'files': now})
    assert changed == sorted(changed)
    assert set(changed) == {k for k in set(before) | set(now) if before.get(k) != now.get(k)}
    assert (changed == []) == (before == now)
    assert sources.changed_files({'files': now}, {'files': before}) == changed


@PROFILE
@given(file_maps, file_maps)
def test_digest_changes_iff_files_change(a, b):
    assert (sources.files_digest(a) == sources.files_digest(b)) == (a == b)


# ------------------------------------------------------------ factors

@PROFILE
@given(st.dictionaries(names, st.lists(st.integers(), min_size=1, max_size=4, unique=True), max_size=3))
def test_declared_factors_is_the_full_product(axes):
    factors = checks.declared_factors({'axes': axes})
    assert len(factors) == math.prod(len(v) for v in axes.values())
    assert len({json.dumps(f, sort_keys=True) for f in factors}) == len(factors)
    assert all(set(f) == set(axes) and all(f[k] in axes[k] for k in f) for f in factors)


# ------------------------------------------------------------ gate

@PROFILE
@given(st.floats(min_value=0, max_value=1, allow_nan=False))
def test_ar_rate_never_substitutes_for_missing_argmax_audit(tmp_path_factory, rate):
    assets = tmp_path_factory.mktemp('a')
    with (assets / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['category', 'exact_match_rate'])
        writer.writeheader()
        writer.writerow({'category': 'overall', 'exact_match_rate': repr(rate)})
    (assets / 'spec_mtp_sources.json').write_text('{"digest": "d"}')
    config = {'algorithm': {'config': {'task': 'esp_mtp', 'eval': {}}}}
    assert not checks.evaluate_run(config, assets)['passed']


# ------------------------------------------------------------ variants

presets = st.sampled_from(sorted({**VARIANTS, **PAPER_VARIANTS}))


@PROFILE
@given(presets, st.dictionaries(st.sampled_from(['impl', 'method', 'extra']), st.text(max_size=4), max_size=3))
def test_apply_variant_overrides_preset_keys_and_is_idempotent(name, base):
    preset = {**VARIANTS, **PAPER_VARIANTS}[name]
    once = apply_variant(base, name)
    assert apply_variant(dict(once), name) == once
    for key in preset:
        assert key in once
    for key in set(base) - set(preset):
        assert once[key] == base[key]


@PROFILE
@given(presets)
def test_apply_variant_is_case_insensitive(name):
    assert apply_variant({}, name.upper()) == apply_variant({}, name)


@PROFILE
@given(names.filter(lambda s: s.lower() not in {**VARIANTS, **PAPER_VARIANTS}))
def test_unknown_variant_raises(name):
    with pytest.raises(ValueError, match='unknown spec variant'):
        apply_variant({}, name)


def test_empty_variant_returns_input():
    base = {'impl': 'naive'}
    assert apply_variant(base, None) is base and apply_variant(base, '') is base


# ------------------------------------------------------------ cli parsing

flags = st.lists(st.sampled_from(['--remake', '--dry-run', '-v', '--phases=write']), max_size=3)


@PROFILE
@given(st.sampled_from(['make', 'run', 'launch', 'status', 'collect', 'study']), flags)
def test_command_and_study_arg(tmp_path_factory, command, extra):
    study = tmp_path_factory.mktemp('s')
    (study / 'study.yaml').write_text('study: s\n')
    argv = [command, *extra, str(study)] if command != 'study' else ['study', 'run', *extra, str(study)]
    assert cli._command(argv) == ('run' if command == 'study' else command)
    assert cli._study_arg(argv) == study


def test_command_edge_cases(tmp_path):
    assert cli._command([]) is None
    assert cli._command(['study']) == 'study'
    assert cli._study_arg(['make', str(tmp_path)]) is None  # no study.yaml
    assert cli._study_arg(['make', '--flag']) is None
    assert cli.write_lock_for([]) is None


def test_main_forwards_to_rpipe(monkeypatch, tmp_path, capsys):
    calls = []
    import rpipe.flow.cli as rpipe_cli

    monkeypatch.setattr(rpipe_cli, 'main', lambda argv: calls.append(argv) or None)
    assert cli.main(['status', str(tmp_path)]) == 0
    study = tmp_path / 's'
    study.mkdir()
    (study / 'study.yaml').write_text('study: s\n')
    assert cli.main(['make', str(study)]) == 0
    assert calls == [['status', str(tmp_path)], ['make', str(study)]]
    assert 'spec_mtp sources:' in capsys.readouterr().out
    assert (study / sources.LOCK_NAME).is_file()
