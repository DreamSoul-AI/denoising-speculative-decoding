"""Native CLI engineering checks: CUDA numeric execution and CPU admission IO.

The old CPU 27/31/8 acceptance counts retain their historical identity. These
new CUDA fixtures use CUDA model initialization and make no bitwise comparison
with that CPU baseline; native full-token T0 argmax is the correctness gate.
"""

import csv
import hashlib
import json

import pytest
import torch
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from studies._shared.flow import cli as shared_cli
from studies._shared.structure import sources as shared_sources

from tests.conftest import cli


_CUDA_RECIPE = '''"""CUDA-only model-construction fixture; native admission remains first."""
import torch
from studies._shared.structure.recipe import register as native_register
from studies._shared.structure.model import _build_tiny, _build_hf
from spec_mtp.structure.model.factory import ModelRegistry


def _cuda(builder, *args, **kwargs):
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    with torch.device('cuda:0'):
        model = builder(*args, **kwargs)
    assert model.module.device == torch.device('cuda:0')
    print('native fixture model device=' + str(model.module.device) + ' name=' + torch.cuda.get_device_name(0), flush=True)
    return model


def register(ctx):
    native_register(ctx)
    ModelRegistry.register('tiny_llama', 'spec_mtp', lambda *a, **kw: _cuda(_build_tiny, *a, **kw))
    ModelRegistry.register('hf_causal', 'spec_mtp', lambda *a, **kw: _cuda(_build_hf, *a, **kw))
'''


@pytest.fixture
def cuda_study_copy(study_copy, record_property):
    assert torch.cuda.is_available(), 'native smoke requires CUDA; no CPU fallback'
    device_name = torch.cuda.get_device_name(0)
    print('native smoke device=cuda:0 name=' + device_name, flush=True)
    record_property('cuda_device', device_name)

    def copy(patch=None, name='cuda_smoke'):
        def configure(study):
            study['description'] = 'New CUDA native CLI engineering fixture; old CPU counts are historical only'
            study['fixed']['system']['config']['device'] = 'cuda:0'
            study['fixed']['algorithm']['config']['eval']['device'] = 'cuda:0'
            if patch:
                patch(study)
        root = study_copy(configure, name='cuda-engineering-' + name)
        recipe = root / 'recipe.py'
        recipe.write_text(_CUDA_RECIPE, encoding='utf-8')
        return root
    return copy


def _overall(study, run_id):
    with (study / 'runs' / run_id / 'assets' / 'summary.csv').open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row['category'] == 'overall']
    assert len(rows) == 1
    return rows[0]


def _check_run_identity(study, run_id):
    root = study / 'runs' / run_id
    inputs = json.loads((root / 'assets/decode_inputs.json').read_text(encoding='utf-8'))
    config = yaml.safe_load((root / 'config.yaml').read_text(encoding='utf-8'))
    assert inputs['device'] == inputs['settings']['eval']['device'] == 'cuda:0'
    assert inputs['backend'] == 'eager'
    assert inputs['settings']['run']['seed'] == config['seed'] == 123
    assert inputs['settings']['data']['seed'] == 0
    assert len(inputs['prompts']) == 4 and all(len(p['input_ids']) == 32 for p in inputs['prompts'])
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'succeeded' and result['spec_mtp']['passed'] is True
    assert result['spec_mtp']['sources']['digest'] == result['spec_mtp']['sources_digest']
    assert result['provenance']
    saved_sources = json.loads((root/'assets/spec_mtp_sources.json').read_text(encoding='utf-8'))
    assert saved_sources['study_files']['recipe.py'] == hashlib.sha256((study/'recipe.py').read_bytes()).hexdigest()
    return inputs, result


def _check_smoke(study):
    index = json.loads((study / 'index.json').read_text(encoding='utf-8'))
    seen = set()
    identities = []
    for experiment in index['experiments']:
        variant = experiment['factors']['algorithm.config.variant']
        run_id = experiment['runs'][0]['id']
        inputs, result = _check_run_identity(study, run_id)
        assert inputs['dtype'] == 'torch.float32'
        assert result['spec_mtp']['argmax_all_passed']
        assert result['spec_mtp']['argmax_checked_prompts'] == 4
        assert result['spec_mtp']['argmax_checked_tokens'] == 128
        assert type(result['spec_mtp']['exact_match_rate']) is float
        overall = _overall(study, run_id)
        assert int(float(overall['prompts'])) == 4
        assert int(float(overall['accept_d1'])) >= 0
        identities.append((inputs['model_digest'], inputs['prompts']))
        seen.add(variant)
    assert seen == {'esp_static', 'esp_dynamic', 'ema_velocity'}
    assert all(identity == identities[0] for identity in identities)
    check = json.loads((study / 'spec_mtp_study_check.json').read_text(encoding='utf-8'))
    assert check['complete'] and check['passed'] and len(check['sources_digests']) == 1


def test_smoke_study_matches_plain_generation(cuda_study_copy):
    """New CUDA identity through the native six-stage chain; AR remains diagnostic."""
    study = cuda_study_copy(name='fresh_smoke')
    cli('run', study, check=True)
    _check_smoke(study)


def test_rpipe_entry_runs_the_same_chain(cuda_study_copy):
    study = cuda_study_copy()
    cli('run', study, entry='spec_mtp.flow.study_cli', check=True)
    _check_smoke(study)


def test_make_then_launch_scripts_path(cuda_study_copy):
    study = cuda_study_copy()
    cli('make', study, check=True)
    assert (study / 'spec_mtp_sources.json').is_file()
    assert (study / 'scripts' / 'launch.sh').is_file()
    proc = cli('launch', study, '--console', 'shared')
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    _check_smoke(study)


def test_context_limit_run_passes_on_reached_budget(cuda_study_copy):
    """Option A: context limit 56 < 32 + 32; AR is audited on the budget the speculative pass reached."""
    def patch(study):
        study['fixed']['model']['config']['tiny']['max_position_embeddings'] = 56
        study['axes'] = {'algorithm.config.variant': ['esp_static']}
    study = cuda_study_copy(patch, name='ctx_limit')
    cli('run', study, check=True)
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    _, result = _check_run_identity(study, run_id)
    assert result['spec_mtp']['argmax_all_passed']
    raw = json.loads((study/'runs'/run_id/'assets/decode_observations.json').read_text())
    assert all(item['stats']['hit_context_limit'] for item in raw['out']['evidence'])
    assert all(len(item['ar_ids']) == len(item['output_ids']) for item in raw['out']['evidence'])


def test_argmax_audit_passes_without_independent_ar(cuda_study_copy):
    def patch(study):
        study['fixed']['algorithm']['config']['eval']['run_ar_baseline'] = False
        study['axes'] = {'algorithm.config.variant': ['esp_static']}
    study = cuda_study_copy(patch, name='without_ar')
    proc = cli('run', study)
    assert proc.returncode == 0, proc.stderr
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    result = json.loads((study / 'runs' / run_id / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'succeeded' and result['spec_mtp']['argmax_all_passed']
    assert all(d['exact'] is None for d in result['spec_mtp']['AR_diagnostics'])
    _check_run_identity(study, run_id)


def test_bad_config_fails_before_construction(study_copy):
    def patch(study):
        study['fixed']['model']['config']['attn_implementation'] = 'flash_attention_2'
        study['axes'] = {'algorithm.config.variant': ['esp_static']}
    study = study_copy(patch, name='bad_attn')
    proc = cli('run', study)
    assert proc.returncode != 0
    index = json.loads((study / 'index.json').read_text(encoding='utf-8'))
    run_id = index['experiments'][0]['runs'][0]['id']
    result = json.loads((study / 'runs' / run_id / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'failed' and 'eager or sdpa' in result['error']
    assert not (study / 'runs' / run_id / 'assets' / 'summary.csv').exists()


def test_freeze_rejects_spec_mtp_change_after_make(study_copy):
    study = study_copy(lambda s: s.update(freeze=True), name='frozen')
    cli('make', study, check=True)
    lock = study / 'spec_mtp_sources.json'
    body = json.loads(lock.read_text(encoding='utf-8'))
    body['files']['spec_mtp/structure/algorithm/tree.py'] = '0' * 64  # simulate a source edit after make
    lock.write_text(json.dumps(body), encoding='utf-8')
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    proc = cli('run-one', study, run_id)
    assert proc.returncode != 0
    assert not (study / 'runs' / run_id / 'result.json').exists()
    assert 'tree.py' in proc.stderr


def test_freeze_rejects_study_change_after_make(study_copy):
    """RPipe's own freeze still covers the Study files (recipe, flow, declarations)."""
    study = study_copy(lambda s: s.update(freeze=True), name='frozen_study')
    cli('make', study, check=True)
    hook = study / 'flow' / 'write' / 'run.py'
    hook.write_text(hook.read_text(encoding='utf-8') + '\n# edited after make\n', encoding='utf-8')
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    proc = cli('run-one', study, run_id)
    assert proc.returncode != 0
    assert 'provenance changed' in (proc.stdout + proc.stderr)


def test_alignment_probe_writes_metrics_and_passes_the_gate(cuda_study_copy):
    """The probe's artifact is alignment_metrics.csv; the gate needs it (was dropped before 2026-10-12)."""
    def patch(study):
        study['fixed']['algorithm']['config']['task'] = 'alignment_probe'
        study['fixed']['algorithm']['config']['eval']['probe_steps_per_prompt'] = 4
        study['axes'] = {'algorithm.config.variant': ['esp_static']}
    study = cuda_study_copy(patch, name='probe')
    cli('run', study, check=True)
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    with (study / 'runs' / run_id / 'assets' / 'alignment_metrics.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r['layer']) for r in rows] == [1, 2]
    assert all(int(r['n_accepted']) + int(r['n_rejected']) > 0 for r in rows)
    result = json.loads((study / 'runs' / run_id / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'succeeded' and result['spec_mtp']['passed']
    _check_run_identity(study, run_id)


def test_ar_baseline_runs_the_same_stage_chain(cuda_study_copy):
    def patch(study):
        study['fixed']['algorithm']['config']['task'] = 'ar_baseline'
        study['axes'] = {'algorithm.config.variant': ['esp_static']}
    study = cuda_study_copy(patch, name='ar_stage_chain')
    cli('run', study, check=True)
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    root = study / 'runs' / run_id
    result = json.loads((root / 'result.json').read_text())
    raw = json.loads((root / 'assets/decode_observations.json').read_text())
    assert result['status'] == 'succeeded' and result['spec_mtp']['passed']
    _check_run_identity(study, run_id)
    assert len(raw['out']['evidence']) == 4
    assert all(len(item['output_ids']) == 64 for item in raw['out']['evidence'])


def test_ar_diagnostic_mismatch_does_not_block_native_success(cuda_study_copy):
    study = cuda_study_copy(name='raw_mismatch')
    hook = study / 'flow/collect/run.py'
    hook.write_text("""from studies._shared.flow.collect import run as collect

def run(ctx):
    collect(ctx)
    import json
    from pathlib import Path
    path = Path(ctx.layout.assets_dir) / 'decode_observations.json'
    raw = json.loads(path.read_text())
    raw['out']['evidence'][0]['ar_ids'] = list(raw['out']['evidence'][0]['output_ids'])
    if raw['out']['rows'][0]['exact_match']:
        raw['out']['evidence'][0]['ar_ids'][-1] = -1
    path.write_text(json.dumps(raw))
""")
    proc = cli('run', study)
    assert proc.returncode == 0, proc.stderr
    index = json.loads((study / 'index.json').read_text())
    assert len(index['experiments']) == 3
    for experiment in index['experiments']:
        root = study / 'runs' / experiment['runs'][0]['id']
        assert (root / 'result.json').is_file()
        result = json.loads((root / 'result.json').read_text())
        assert result['status'] == 'succeeded' and result['spec_mtp']['argmax_all_passed']
        _check_run_identity(study, root.name)
        raw = json.loads((root/'assets/decode_observations.json').read_text())
        assert result['spec_mtp']['AR_diagnostics'][0]['exact'] is not bool(raw['out']['rows'][0]['exact_match'])
        assert result['spec_mtp']['AR_reported_exact_mismatches']==[0]


def test_source_preflight_preserves_existing_result(study_copy):
    study = study_copy(name='preserve_previous_result')
    cli('make', study, check=True)
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    result = study / 'runs' / run_id / 'result.json'
    original = b'{"status":"succeeded","previous_evidence":"do not replace"}'
    result.write_bytes(original)
    lock = study / 'spec_mtp_sources.json'
    saved = json.loads(lock.read_text())
    saved['files']['spec_mtp/new.py'] = 'old'
    lock.write_text(json.dumps(saved))
    for command in ('run', 'launch', 'run-one', 'process'):
        proc = cli(command, study, *([run_id] if command == 'run-one' else []))
        assert proc.returncode != 0 and 'sources changed' in proc.stderr
        assert result.read_bytes() == original


def test_nested_config_axis_validates_effective_factors_and_rejects_subsets(cuda_study_copy):
    def patch(study):
        study['axes'] = {'algorithm.config': [
            {'task': 'esp_mtp', 'mask': 'frozen_mean'},
            {'task': 'esp_mtp', 'mask': 'gamma_last'},
        ]}
    study = cuda_study_copy(patch, name='nested_config_axis')
    cli('make', study, '--num-gpus', '1', check=True)
    cli('launch', study, '--num-gpus', '1', '--console', 'shared', check=True)
    output = study.parent / 'nested-report/numbers.md'
    cli('report', study, '--out', output, check=True)
    process = json.loads((study / 'process.json').read_text())
    assert process['complete'] and process['spec_mtp']['expected_factors_match']
    index = json.loads((study / 'index.json').read_text())
    assert all('eval' in exp['factors']['algorithm.config'] for exp in index['experiments'])
    for experiment in index['experiments']:
        for run in experiment['runs']:
            _check_run_identity(study, run['id'])
    previous = output.read_bytes()
    index['experiments'] = index['experiments'][:1]
    (study / 'index.json').write_text(json.dumps(index))
    result = cli('process', study)
    assert result.returncode != 0 and 'incomplete' in result.stdout + result.stderr
    result = cli('report', study, '--out', study.parent / 'rejected-report/numbers.md')
    assert result.returncode != 0 and output.read_bytes() == previous


def test_output_hidden_real_cli_and_tampered_owner_evidence(cuda_study_copy):
    import yaml
    from studies._shared.structure.validation import verify_evidence, SpecMtpGateError
    import pytest

    def patch(study):
        study['fixed']['model']['config']['tiny']['tie_word_embeddings'] = True
        study['axes'] = {'algorithm.config.spec': [{
            'method': 'output_hidden', 'num_masks': 2, 'impl': 'efficient',
            'tree': {'mode': 'static', 'branches': [1, 1], 'pruning': False},
            'output': {'order': 1, 'beta': .9, 'beta_prime': .99},
        }]}
    study = cuda_study_copy(patch, name='output_hidden_cli')
    cli('make', study, '--num-gpus', '1', check=True)
    cli('launch', study, '--num-gpus', '1', '--console', 'shared', check=True)
    cli('report', study, '--out', study.parent / 'output-hidden-report/numbers.md', check=True)
    process = json.loads((study / 'process.json').read_text())
    assert process['complete'] and process['spec_mtp']['passed']
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    run = study / 'runs' / run_id
    _check_run_identity(study, run_id)
    raw_path = run / 'assets/decode_observations.json'
    raw = json.loads(raw_path.read_text())
    raw['out']['evidence'][0]['stats']['trace'][0]['output_hidden']['owner_count'] = 1
    raw_path.write_text(json.dumps(raw))
    with pytest.raises(SpecMtpGateError, match='owner/observation'):
        verify_evidence(yaml.safe_load((run / 'config.yaml').read_text()), run / 'assets')


def test_bf16_five_methods_and_independent_ar_native_cli(cuda_study_copy, tmp_path):
    from copy import deepcopy
    import pytest
    import torch
    import yaml
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from studies._shared.structure.validation import verify_evidence, SpecMtpGateError

    saved = tmp_path / 'bf16_local_hf'
    torch.manual_seed(7)
    with torch.device('cuda:0'):
        model = HFCausalLM(tiny_config={'vocab_size': 256, 'hidden_size': 32, 'intermediate_size': 64,
        'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 256,
        'tie_word_embeddings': True})
    try:
        assert model.device == torch.device('cuda:0')
        model.hf.to(torch.bfloat16).save_pretrained(saved)
    finally:
        model.close()
    common = {'num_masks': 2, 'impl': 'naive', 'block_complexity': 30,
              'tree': {'mode': 'dynamic', 'pruning': True}}
    variants = [
        {**common, 'method': 'esp', 'mask': {'init': 'mean_prompt', 'update': False}},
        {**common, 'method': 'esp', 'mask': {'init': 'mean_prompt', 'lam': .1, 'update': True}},
        {**common, 'method': 'ema_velocity', 'ema': {'level': 'order0', 'beta': .9999,
            'lam': .1, 'step_scale': 1., 'normalize': False, 'extrapolate_ema': False}},
        *[{**common, 'method': 'output_hidden', 'output': {'order': order, 'beta': .9, 'beta_prime': .99}}
          for order in (0, 1)],
    ]
    def patch(study):
        study['fixed']['model'] = {'name': 'hf_causal', 'source': 'spec_mtp', 'config': {
            'name_or_path': str(saved), 'local_files_only': True,
            'torch_dtype': 'bfloat16', 'attn_implementation': 'eager'}}
        base = deepcopy(study['fixed']['algorithm']['config'])
        base['eval'].update(run_ar_baseline=False, warmup=False, max_new_tokens=12)
        study['axes'] = {'algorithm.config': [dict(deepcopy(base), spec=v) for v in variants]
                        + [dict(deepcopy(base), task='ar_baseline')]}
    study = cuda_study_copy(patch, name='bf16_five_methods_cli')
    cli('make', study, '--num-gpus', '1', check=True)
    cli('launch', study, '--num-gpus', '1', '--console', 'shared', check=True)
    check = json.loads((study / 'spec_mtp_study_check.json').read_text())
    assert check['complete'] and check['passed'] and len(check['runs']) == 6
    output_run = None
    for entry in json.loads((study / 'index.json').read_text())['experiments']:
        run = study / 'runs' / entry['runs'][0]['id']
        result = json.loads((run / 'result.json').read_text())
        inputs = json.loads((run / 'assets/decode_inputs.json').read_text())
        assert inputs['dtype'] == 'torch.bfloat16' and inputs['device'] == 'cuda:0' and result['status'] == 'succeeded'
        _check_run_identity(study, run.name)
        if result['spec_mtp']['task'] == 'esp_mtp':
            assert result['spec_mtp']['argmax_all_passed']
            assert all(d['exact'] is None for d in result['spec_mtp']['AR_diagnostics'])
            if inputs['settings']['spec']['method'] == 'output_hidden':
                output_run = run
    assert output_run is not None
    raw_path = output_run / 'assets/decode_observations.json'
    original = raw_path.read_bytes()
    config = yaml.safe_load((output_run / 'config.yaml').read_text())
    for key, value, error in [('owner_count', 1, 'owner/observation'),
                              ('mask_norm_min', 0., 'mapping norm')]:
        raw = json.loads(original)
        raw['out']['evidence'][0]['stats']['trace'][0]['output_hidden'][key] = value
        raw_path.write_text(json.dumps(raw))
        with pytest.raises(SpecMtpGateError, match=error):
            verify_evidence(config, output_run / 'assets')
    raw_path.write_bytes(original)


_CLI_PROPERTY_SETTINGS = settings(max_examples=60, deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture])
_CLI_PROPERTY_FLAGS = st.lists(st.sampled_from(['--remake', '--dry-run', '-v', '--phases=write']), max_size=3)


@_CLI_PROPERTY_SETTINGS
@given(st.sampled_from(['make', 'run', 'launch', 'status', 'collect', 'study']), _CLI_PROPERTY_FLAGS)
def test_command_and_study_arg(tmp_path_factory, command, extra):
    study = tmp_path_factory.mktemp('cli-property')
    (study/'study.yaml').write_text('study: s\n', encoding='utf-8')
    argv = [command, *extra, str(study)] if command != 'study' else ['study', 'run', *extra, str(study)]
    assert shared_cli._command(argv) == ('run' if command == 'study' else command)
    assert shared_cli._study_arg(argv) == study


def test_command_edge_cases(tmp_path):
    assert shared_cli._command([]) is None
    assert shared_cli._command(['study']) == 'study'
    assert shared_cli._study_arg(['make', str(tmp_path)]) is None
    assert shared_cli._study_arg(['make', '--flag']) is None
    assert shared_cli.write_lock_for([]) is None


def test_main_forwards_to_rpipe(monkeypatch, tmp_path, capsys):
    """The historical test name now checks the repository's own host CLI."""
    from spec_mtp.flow import study_cli as host_cli
    calls = []
    monkeypatch.setattr(host_cli, 'main', lambda argv: calls.append(argv) or None)
    assert shared_cli.main(['status', str(tmp_path)]) == 0
    study = tmp_path/'s'; study.mkdir()
    (study/'study.yaml').write_text('study: s\n', encoding='utf-8')
    assert shared_cli.main(['make', str(study)]) == 0
    assert calls == [['status', str(tmp_path)], ['make', str(study)]]
    assert 'spec_mtp sources:' in capsys.readouterr().out
    assert (study/shared_sources.LOCK_NAME).is_file()


def test_cli_lock_only_for_make_run_launch(tmp_path):
    study = tmp_path / 's'
    study.mkdir()
    (study / 'study.yaml').write_text('study: s\n')
    assert shared_cli.write_lock_for(['status', str(study)]) is None
    assert shared_cli.write_lock_for(['make', str(study)]) == study / shared_sources.LOCK_NAME
    first = json.loads((study / shared_sources.LOCK_NAME).read_text())['created_at']
    (study / 'scripts').mkdir()
    (study / 'scripts' / 'jobs.json').write_text('{}')
    assert shared_cli.write_lock_for(['launch', str(study)]) is None  # keep the make-time lock
    assert json.loads((study / shared_sources.LOCK_NAME).read_text())['created_at'] == first
    assert shared_cli.write_lock_for(['launch', str(study), '--remake']) is not None
    assert shared_cli.write_lock_for(['study', 'run', str(study)]) is None  # run must not refresh freeze lock


def test_run_does_not_create_lock_when_freeze(tmp_path):
    study = tmp_path / 'frozen'
    study.mkdir()
    (study / 'study.yaml').write_text('study: s\nfreeze: true\n', encoding='utf-8')
    assert shared_cli.write_lock_for(['run', str(study)]) is None
    assert not (study / shared_sources.LOCK_NAME).exists()


def test_run_writes_lock_when_missing_and_unfrozen(tmp_path):
    study = tmp_path / 'open'
    study.mkdir()
    (study / 'study.yaml').write_text('study: s\n', encoding='utf-8')
    assert shared_cli.write_lock_for(['run', str(study)]) == study / shared_sources.LOCK_NAME


def test_spec_cli_report_keeps_upstream_identity_and_specialized_evidence_gates(cuda_study_copy, monkeypatch, record_property):
    monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
    source = cuda_study_copy(lambda s: s.update(axes={'algorithm.config.variant': ['esp_static']}),
                             name='report_cuda_readonly')
    recipe = source / 'recipe.py'
    recipe_text = recipe.read_text(encoding='utf-8')
    assert recipe_text.count('    return model\n') == 1
    recipe.write_text(recipe_text.replace('    return model\n', '''    seen_forward = False

    def observe_forward(module, inputs, output):
        nonlocal seen_forward
        assert inputs[0].device == inputs[1].device == output.logits.device == torch.device('cuda:0')
        if not seen_forward:
            print('report fixture actual forward device=' + str(output.logits.device)
                  + ' name=' + torch.cuda.get_device_name(output.logits.device), flush=True)
            seen_forward = True

    model.module.register_forward_hook(observe_forward)
    return model
'''), encoding='utf-8')
    before_source = ({p.relative_to(source).as_posix(): p.read_bytes()
                      for p in source.rglob('*') if p.is_file()},
                     {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_dir()})
    study = source.parent / 'report-runtime'
    cli('make', source, '--run-root', study, '--num-gpus', '1', check=True)
    launched = cli('launch', study, '--num-gpus', '1', '--console', 'shared', check=True)
    marker = 'report fixture actual forward device=cuda:0 name=' + torch.cuda.get_device_name(0)
    assert marker in launched.stdout + launched.stderr
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    inputs, completed = _check_run_identity(study, run_id)
    assert inputs['device'] == 'cuda:0' and completed['spec_mtp']['argmax_all_passed']
    record_property('report_forward_device', inputs['device'])
    record_property('report_worker_completed', completed['status'])
    path = study.parent / 'export/numbers.md'
    process_path = study / 'process.json'
    initial = process_path.read_bytes()
    assert cli('report', study).returncode != 0
    cli('report', study, '--out', path, check=True)
    assert process_path.read_bytes() == initial
    previous = path.read_bytes()
    process = json.loads(process_path.read_text())
    process['experiments'][0]['runs'][0]['seed'] = -1
    process_path.write_text(json.dumps(process))
    stale_process = process_path.read_bytes()
    result = cli('report', study, '--out', study.parent / 'other/numbers.md')
    assert process_path.read_bytes() == stale_process
    assert result.returncode != 0 and 'current index' in result.stdout + result.stderr
    assert path.read_bytes() == previous
    cli('process', study, check=True)
    run_id = json.loads((study / 'index.json').read_text())['experiments'][0]['runs'][0]['id']
    (study / 'runs' / run_id / 'assets/decode_observations.json').unlink()
    initial = process_path.read_bytes()
    result = cli('report', study, '--out', study.parent / 'rejected/numbers.md')
    assert result.returncode != 0 and process_path.read_bytes() == initial
    assert path.read_bytes() == previous
    assert before_source == ({p.relative_to(source).as_posix(): p.read_bytes()
                              for p in source.rglob('*') if p.is_file()},
                             {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_dir()})
