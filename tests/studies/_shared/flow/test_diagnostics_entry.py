"""Diagnostic command input/output and resource contracts without scientific execution."""

import contextlib
import hashlib
import json
from types import SimpleNamespace

import pytest

from studies._shared.flow import diagnostics as command


@pytest.fixture
def diagnostic_case(tmp_path, monkeypatch):
    import torch
    from spec_mtp.structure.model import hf_causal
    from spec_mtp.structure.data import prompts

    calls = []
    plan = tmp_path / 'plan.md'
    plan.write_bytes(b'# Bounded diagnostic engineering fixture\n')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'variants': ['esp_static']}))
    source = {'git': {'head': 'a' * 40, 'package_dirty': False},
              'checkout_git': {'head': 'a' * 40, 'package_dirty': False}, 'files': {'fixture.py': 'b' * 64}}
    monkeypatch.setattr(command, '_source', lambda: source)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'IO fixture; no CUDA calculation')
    monkeypatch.setattr(torch.random, 'fork_rng', lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch, 'manual_seed', lambda *args: None)
    class Model:
        dtype = 'float32'
        def __init__(self, **config):
            self.hf = self
            calls.append(('model', config))
        def to(self, device):
            self.device = device
        def close(self):
            calls.append(('closed',))
    class Ids:
        def to(self, device):
            return self
    monkeypatch.setattr(hf_causal, 'HFCausalLM', Model)
    monkeypatch.setattr(prompts, 'read_local_prompts', lambda *args: [{'input_ids': Ids()}])
    def fingerprint(models, prompts, device, *, presets):
        calls.append(('fingerprint', str(device)))
        row = {'new_tokens': 1, 'accepted': 0, 'committed': 1, 'decode_calls': 1, 'hit_context_limit': 0,
               'tau': 1., 'attempt_by_depth': [1, 0, 0], 'accept_by_depth': [0, 0, 0], 'tokens': [1], 'equals_full_ar': True}
        cells = {f'{variant}|pos={position}|T={temperature}|p={index}': dict(row) for variant in presets
                 for position, temperatures in [(1024, [0., .7, 1.]), (64, [0., 1.])]
                 for temperature in temperatures for index in range(3)}
        return {'cells': cells, 'count': len(cells), 'sha256': hashlib.sha256(json.dumps(cells, sort_keys=True).encode()).hexdigest()}
    monkeypatch.setattr(command.fingerprint, 'fingerprint', fingerprint)
    def profile(*values):
        calls.append(('profile', str(values[-1])))
        return {'variant': values[0], 'exact_match_rate': 1., 'tau': 1., 'decode_calls': 1,
                'spec_tok_per_s': 1., 'ar_tok_per_s': 1., 'self_time_share': {'other Python': 1.}, 'top_functions': []}
    monkeypatch.setattr(command.profile_cpu, 'profile_variant', profile)
    args = SimpleNamespace(command='fingerprint', config=config, plan=plan,
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(), device='cuda:0', output=tmp_path / 'new-run', compare=None)
    return SimpleNamespace(args=args, source=source, calls=calls, model_class=Model)


@pytest.mark.parametrize('kind', ['fingerprint', 'profile'])
def test_diagnostics_bind_inputs_and_publish_only_after_models_close(diagnostic_case, kind):
    case, args = diagnostic_case, diagnostic_case.args
    args.command = kind
    if kind == 'profile':
        args.config.write_text(json.dumps({**command.PROFILE_DEFAULTS, 'variants': ['esp_static'], 'hidden': 32, 'layers': 1}))
    assert command.run(args) == 0
    header = json.loads((args.output / 'header.json').read_text())
    assert header['identity']['device'] == 'cuda:0' and header['plan']['sha256'] == args.plan_sha256
    assert case.calls[-1] == ('closed',)
    assert (args.output / 'protocol_complete.json').is_file()
    if kind == 'profile':
        report = json.loads((args.output / 'profile.json').read_text())
        assert 'host self time' in report['measurement'] and 'not CUDA kernel time' in report['measurement']
        assert 'not CUDA kernel time' in (args.output / 'profile.md').read_text()


@pytest.mark.parametrize('fault', ['plan', 'config', 'existing', 'checkout_dirty', 'unknown', 'cpu', 'variant', 'profile_negative', 'profile_model'])
def test_diagnostics_invalid_binding_fails_before_model_or_output(diagnostic_case, fault):
    case, args = diagnostic_case, diagnostic_case.args
    if fault == 'plan':
        args.plan_sha256 = '0' * 64
    elif fault == 'config':
        args.config.write_text('{}')
    elif fault == 'existing':
        args.output.mkdir()
        (args.output / 'keep').write_bytes(b'old evidence')
    elif fault == 'checkout_dirty':
        case.source['checkout_git']['package_dirty'] = True
    elif fault == 'unknown':
        case.source['git'] = {}
    elif fault == 'cpu':
        args.device = 'cpu'
    elif fault == 'variant':
        args.config.write_text(json.dumps({'variants': ['unknown']}))
    else:
        args.command = 'profile'
        args.config.write_text(json.dumps({**command.PROFILE_DEFAULTS,
            **({'new_tokens': -1} if fault == 'profile_negative' else {'hidden': 35})}))
    with pytest.raises((ValueError, FileExistsError)):
        command.run(args)
    assert not case.calls
    assert not args.output.exists() if fault != 'existing' else (args.output / 'keep').read_bytes() == b'old evidence'


@pytest.mark.parametrize('damage', ['unknown', 'device', 'workload', 'source', 'cells'])
def test_fingerprint_comparison_rejects_unknown_or_incomparable_report_identity(diagnostic_case, damage):
    case, args = diagnostic_case, diagnostic_case.args
    command.run(args)
    baseline = args.output / 'fingerprint.json'
    old = json.loads(baseline.read_text())
    if damage == 'unknown':
        old.pop('identity')
    elif damage == 'device':
        old['identity']['device'] = 'cpu'
    elif damage == 'workload':
        old['identity']['workload']['model_seed'] = 99
    elif damage == 'source':
        old['source']['checkout_git']['package_dirty'] = True
    else:
        old['cells'].pop(next(iter(old['cells'])))
    baseline.write_text(json.dumps(old))
    case.calls.clear()
    args.compare, args.output = baseline, args.output.parent / 'comparison'
    with pytest.raises(ValueError, match='comparison requires'):
        command.run(args)
    assert not case.calls and not args.output.exists()


def test_fingerprint_comparison_binds_baseline_bytes_and_keeps_field_diff_formula(diagnostic_case):
    args = diagnostic_case.args
    command.run(args)
    baseline = args.output / 'fingerprint.json'
    args.compare, args.output = baseline, args.output.parent / 'comparison'
    assert command.run(args) == 0
    header = json.loads((args.output / 'header.json').read_text())
    assert header['input_sources'][str(baseline)] == hashlib.sha256(baseline.read_bytes()).hexdigest()


@pytest.mark.parametrize('fault', ['source_drift', 'close', 'primary_and_cleanup'])
def test_diagnostic_partial_failures_preserve_primary_and_never_publish_completion(diagnostic_case, monkeypatch, fault):
    from spec_mtp.structure.system.logger import Logger

    case, args = diagnostic_case, diagnostic_case.args
    original = command.fingerprint.fingerprint
    primary = ValueError('primary computation failure')
    def compute(*values, **options):
        if fault == 'primary_and_cleanup':
            raise primary
        report = original(*values, **options)
        if fault == 'source_drift':
            args.plan.write_bytes(b'changed')
        return report
    monkeypatch.setattr(command.fingerprint, 'fingerprint', compute)
    if fault != 'source_drift':
        def close(self):
            case.calls.append(('closed',))
            raise OSError('close failed')
        monkeypatch.setattr(case.model_class, 'close', close)
    if fault == 'primary_and_cleanup':
        def log(*values):
            raise OSError('log failed')
        monkeypatch.setattr(Logger, 'exception', log)
    with pytest.raises((ValueError, OSError)) as error:
        command.run(args)
    if fault == 'primary_and_cleanup':
        assert error.value is primary and len(primary.cleanup_errors) == 3
    assert len([call for call in case.calls if call[0] == 'closed']) == 2
    assert not (args.output / 'protocol_complete.json').exists()


@pytest.mark.parametrize('missing', ['--config', '--plan', '--plan-sha256', '--device', '--output'])
def test_diagnostics_cli_requires_explicit_execution_bindings(missing):
    values = ['fingerprint']
    for flag in ('--config', '--plan', '--plan-sha256', '--device', '--output'):
        if flag != missing:
            values.extend((flag, 'fixture'))
    with pytest.raises(SystemExit) as error:
        command.main(values)
    assert error.value.code == 2
