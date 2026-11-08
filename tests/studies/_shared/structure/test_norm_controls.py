"""cpu pure IO/spies: source identity, Run-root binding and control admission."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from spec_mtp.structure.algorithm.factory import AlgorithmRegistry
from spec_mtp.structure.model.factory import ModelRegistry
from spec_mtp.structure.artifact.layout import materialize_study
from studies._shared.structure import recipe, validation

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize('fault', ['empty', 'malformed', 'escape', 'mismatch', 'plan'])
def test_norm_control_original_refs_and_plan_reject_before_model(tmp_path, monkeypatch, fault):
    from spec_mtp.structure.artifact import provenance
    checkout = tmp_path/'checkout'; checkout.mkdir()
    study = checkout/'studies/norm'; (study/'docs').mkdir(parents=True)
    source = checkout/'original.py'; source.write_bytes(b'original')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    files = {'original.py': expected}
    if fault == 'empty': files = {}
    elif fault == 'malformed': files = {'original.py': 'bad'}
    elif fault == 'escape': files = {'../outside.py': expected}
    elif fault == 'mismatch': files = {'original.py': '0'*64}
    refs = {'files': files, 'project_plan': {'commit': '1'*40, 'path': 'plan.md', 'sha256': '2'*64}}
    (study/'docs/source_refs.json').write_text(json.dumps(refs), encoding='utf-8')
    def historical(*args):
        assert fault == 'plan'
        raise ValueError('original plan differs')
    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    with pytest.raises((ValueError, AssertionError)):
        validation.check_control_refs(study, checkout, kind='norm')


@pytest.mark.parametrize('kind', ['norm', 'paired', 'continuation'])
def test_norm_control_snapshot_roots_policy_and_original_matrix(tmp_path, monkeypatch, kind):
    checkout = tmp_path/'checkout'
    pairing = checkout/'studies/qwen3_4b_bf16_five_methods'
    names = {'norm': 'qwen3_4b_bf16_bias_norm_scan', 'paired': 'qwen3_4b_bf16_paired_revalidation',
             'continuation': 'qwen3_4b_bf16_bias_norm_continuation'}
    source = checkout/'studies'/names[kind]
    for study in (pairing, source):
        study.mkdir(parents=True)
        (study/'recipe.py').write_text('pass\n', encoding='utf-8')
        declaration = yaml.safe_load((ROOT/'studies'/study.name/'study.yaml').read_text(encoding='utf-8'))
        declaration['provenance'] = {'include': []}
        (study/'study.yaml').write_text(yaml.safe_dump(declaration), encoding='utf-8')
        (study/'experiment_config.yaml').write_bytes((ROOT/'studies'/study.name/'experiment_config.yaml').read_bytes())
    runtime = materialize_study(source, tmp_path/'random-external-leaf')
    classes, builders = [], []
    monkeypatch.setattr(AlgorithmRegistry, 'register', lambda mode, source, cls: classes.append(cls))
    monkeypatch.setattr(ModelRegistry, 'register', lambda name, source, builder: builders.append(builder))
    assets = runtime/'runs/example/assets'
    recipe.register_control(runtime, admission_assets=assets, kind=kind)
    selected = classes[-1]
    assert selected.study_root == runtime and selected.execution_root == checkout and selected.pairing_root == pairing
    assert builders[-1].keywords['admission_assets'] == assets
    waiter = selected.admission_wait
    assert waiter.keywords['capacity'] == checkout/'studies/qwen3_4b_bf16_bias_norm_scan/docs/capacity_calibration_v1.json'
    assert (waiter.keywords['observer'] is None) == (kind == 'norm')
    if kind == 'paired':
        actual = validation.paired_matrix_settings(runtime, pairing)
        assert actual == validation.paired_matrix_settings(source, pairing) and len(actual) == 11
    (source/'recipe.py').write_text('source drift\n', encoding='utf-8')
    with pytest.raises(ValueError):
        recipe.register_control(runtime, admission_assets=assets, kind=kind)
    with pytest.raises(ValueError):
        validation.check_control_refs(runtime, checkout, kind=kind)
    assert len(classes) == len(builders) == 1


@pytest.mark.parametrize('kind', ['norm', 'paired', 'continuation'])
@pytest.mark.parametrize('failure', [False, True])
def test_norm_control_native_gate_precedes_registration_without_import_side_effects(tmp_path, monkeypatch, kind, failure):
    from spec_mtp.structure.make.recipe import load_recipe
    names = {'norm': 'qwen3_4b_bf16_bias_norm_scan', 'paired': 'qwen3_4b_bf16_paired_revalidation',
             'continuation': 'qwen3_4b_bf16_bias_norm_continuation'}
    source = ROOT/'studies'/names[kind]
    events = []
    def gate(ctx):
        events.append('native')
        if failure: raise RuntimeError('old frozen identity drift')
    monkeypatch.setattr(recipe, 'register', gate)
    monkeypatch.setattr(recipe, 'register_control', lambda *args, **kwargs: events.append((args, kwargs)))
    loaded = load_recipe(source, {'recipe': 'recipe.py'})
    assert events == []
    ctx = SimpleNamespace(study_dir=source, assets_dir=tmp_path/'run-assets')
    if failure:
        with pytest.raises(RuntimeError, match='identity drift'): loaded.register(ctx)
        assert events == ['native']
    else:
        loaded.register(ctx)
        assert events == ['native', ((source,), {'admission_assets': ctx.assets_dir, 'kind': kind})]


@pytest.mark.parametrize('kind', ['norm', 'paired', 'continuation'])
def test_norm_control_source_failure_never_reaches_admission_or_loader(tmp_path, monkeypatch, kind):
    from studies._shared.structure import model
    events = []
    def gate(*args, **kwargs):
        events.append(('refs', kwargs['kind']))
        raise ValueError('unknown original source')
    monkeypatch.setattr(validation, 'check_control_refs', gate)
    monkeypatch.setattr(model, 'build_cuda', lambda *args: pytest.fail('source failure reached model'))
    with pytest.raises(ValueError, match='unknown original source'):
        recipe._build_control_cuda(None, tmp_path/'model-cache', study_root=tmp_path, execution_root=tmp_path,
            admission_assets=tmp_path/'run-assets', kind=kind,
            waiter=lambda *args, **kwargs: pytest.fail('source failure reached admission'))
    assert events == [('refs', kind)]


def test_continuation_checks_only_own_and_norm_original_source_chain(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance
    checkout = tmp_path/'checkout'; checkout.mkdir()
    (checkout/'input.py').write_bytes(b'original')
    expected = hashlib.sha256(b'original').hexdigest()
    seen = []
    monkeypatch.setattr(provenance, 'read_historical_bytes', lambda repo, commit, member, sha: seen.append(member) or b'plan')
    continuation = checkout/'studies/continuation'
    norm = checkout/'studies/qwen3_4b_bf16_bias_norm_scan'
    for study, plan in ((continuation, 'continuation-plan.md'), (norm, 'norm-plan.md')):
        (study/'docs').mkdir(parents=True)
        (study/'docs/source_refs.json').write_text(json.dumps({'files': {'input.py': expected},
            'project_plan': {'commit': '1'*40, 'path': plan, 'sha256': '2'*64}}), encoding='utf-8')
    validation.check_control_refs(continuation, checkout, kind='continuation')
    assert seen == ['continuation-plan.md', 'norm-plan.md']


@pytest.mark.parametrize('kind', ['norm', 'direction'])
def test_reused_control_keeps_two_prompt_diagnostics_as_independent_snapshots(tmp_path, monkeypatch, kind):
    from studies._shared.structure import algorithm, geometry_4b, resource_admission, evaluate
    import spec_mtp.structure.algorithm.decode.loop as decoding
    from transformers import AutoTokenizer
    pairing = tmp_path/'pairing'; (pairing/'docs').mkdir(parents=True)
    assets = tmp_path/'assets'; assets.mkdir()
    prompts = [{'question_id': i, 'input_ids': [i]} for i in range(480)]
    (pairing/'docs/prompts_manifest.json').write_text(json.dumps({'rendered_prompts': prompts, 'data_sha256': 'fixture'}))
    (tmp_path/'datasets/spec_bench').mkdir(parents=True)
    (tmp_path/'datasets/spec_bench/question.jsonl').write_text('\n'.join(json.dumps({'question_id': i, 'turns': [str(i)]}) for i in range(480)))
    monkeypatch.setattr(validation, 'check_control_refs', lambda *a, **kw: None)
    monkeypatch.setattr(validation, 'check_direction_refs', lambda *a: None)
    from spec_mtp.structure.artifact import provenance
    monkeypatch.setattr(provenance, 'file_sha256', lambda path: 'fixture')
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: None)
    monkeypatch.setattr(geometry_4b, 'content_start_of', lambda tokenizer, text, ids: int(text)+1)
    monkeypatch.setattr(resource_admission, 'direction_admission', lambda *a, **kw: {'passed': True})
    def control(provider, mode, start, info):
        provider.init = lambda value: info.update(prompt=value, content_start=start() if callable(start) else start)
        return provider
    monkeypatch.setattr(geometry_4b, 'direction_control', control)
    monkeypatch.setattr(geometry_4b, 'observe_initial_bias', lambda provider, mode, info: control(provider, mode, None, info))
    monkeypatch.setattr(evaluate, 'build_mask_provider', lambda: SimpleNamespace(step_scale=.5))
    provider = None
    def generate(decoder, prompt):
        provider.init(prompt[0].tolist()[0])
        stats = {'verification_argmax_audit': {'passed': True, 'checked_committed_tokens': 1},
            'new_tokens': 1, 'prefill_time': 0, 'decode_time': 0, 'accepted': 0, 'committed': 1,
            'decode_calls': 1, 'attempt_by_depth': [0, 1], 'accept_by_depth': [0, 0], 'trace': []}
        return [SimpleNamespace(tolist=lambda: [9])], stats
    monkeypatch.setattr(decoding.SpecDecoder, 'generate', generate)
    def base_run(self, *args):
        nonlocal provider
        provider = evaluate.build_mask_provider()
        for index in range(2):
            decoding.SpecDecoder.generate(None, [SimpleNamespace(tolist=lambda index=index: [index])])
        raise RuntimeError('bounded two-prompt stop')
    monkeypatch.setattr(algorithm.PairedUntimedEval, 'run', base_run)
    cls = algorithm.NormEval if kind == 'norm' else algorithm.DirectionEval
    instance = object.__new__(cls)
    instance.config = SimpleNamespace(setting=lambda key: {'direction': 'template' if kind == 'norm' else 'content',
        'spec': {'ema': {'step_scale': .5}}}[key])
    instance.study_root = instance.execution_root = tmp_path
    instance.pairing_root = pairing
    instance.admission_wait = lambda *a, **kw: {'passed': True}
    model = SimpleNamespace(meta={'norm_scan_admission': {}, 'direction_admission': {}})
    with pytest.raises(RuntimeError, match='bounded two-prompt stop'):
        instance.run(None, model, SimpleNamespace(assets_dir=assets), None)
    name = 'bias' if kind == 'norm' else 'direction'
    report = json.loads((assets/('norm_progress.json' if kind == 'norm' else 'direction_progress.json')).read_text())
    rows = report['rows']
    assert [row[name]['prompt'] for row in rows] == [0, 1]
    if kind == 'direction': assert [row[name]['content_start'] for row in rows] == [1, 2]


def test_content_direction_reuses_one_cuda_provider_with_distinct_prompt_offsets():
    import torch
    from spec_mtp.structure.algorithm.mask import EMAVelocityMaskProvider
    from studies._shared.structure.geometry_4b import direction_control
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    print('control reuse device=cuda:0 name='+torch.cuda.get_device_name(0), flush=True)
    embeds = torch.arange(144, device='cuda:0', dtype=torch.float32).reshape(9, 16).to(torch.bfloat16)
    provider = EMAVelocityMaskProvider(2, embeds, beta=.9999, level='order0', lam=.1)
    index, starts, info, rows = [0], [1, 3], {}, []
    direction_control(provider, 'content', lambda: starts[index[0]], info)
    for i, prompt in enumerate((embeds[:7], embeds[1:9])):
        index[0] = i; info.clear()
        provider.init_from_prompt(prompt)
        expected = EMAVelocityMaskProvider(2, embeds, beta=.9999, level='order0', lam=.1)
        expected.init_from_prompt(prompt[starts[i]:])
        assert torch.equal(provider._velocity, expected._velocity)
        full = EMAVelocityMaskProvider(2, embeds, beta=.9999, level='order0', lam=.1)
        full.init_from_prompt(prompt)
        assert torch.equal(provider._level, full._level)
        rows.append(dict(info))
    assert [row['content_start'] for row in rows] == starts


@pytest.mark.parametrize('kind', ['norm', 'paired', 'continuation', 'direction'])
def test_control_source_drift_during_admission_rejects_before_loading(tmp_path, monkeypatch, kind):
    from studies._shared.structure import model, resource_admission
    events = []
    def check(*args, **kwargs):
        events.append('refs')
        if events.count('refs') == 2: raise ValueError('source changed while waiting')
    monkeypatch.setattr(validation, 'check_control_refs', check)
    monkeypatch.setattr(validation, 'check_direction_refs', check)
    monkeypatch.setattr(model, 'build_cuda', lambda *a: pytest.fail('source drift reached model loader'))
    def wait(*args, **kwargs):
        events.append('wait')
        return {'passed': True}
    monkeypatch.setattr(resource_admission, 'direction_admission', wait)
    with pytest.raises(ValueError, match='source changed while waiting'):
        if kind == 'direction':
            recipe._build_direction_cuda(None, tmp_path/'model-cache', study_root=tmp_path, execution_root=tmp_path,
                admission_assets=tmp_path/'run-assets', recovery=True)
        else:
            recipe._build_control_cuda(None, tmp_path/'model-cache', study_root=tmp_path, execution_root=tmp_path,
                admission_assets=tmp_path/'run-assets', kind=kind, waiter=wait)
    assert events == ['refs', 'wait', 'refs']


@pytest.mark.parametrize('kind', ['norm', 'paired'])
def test_native_control_reader_uses_verified_source_checkout_for_arbitrary_external_run_root(tmp_path, monkeypatch, kind):
    import importlib
    name = 'qwen3_4b_bf16_bias_norm_scan' if kind == 'norm' else 'qwen3_4b_bf16_paired_revalidation'
    checkout = tmp_path/'checkout'; source = checkout/'studies'/name
    source.mkdir(parents=True)
    (source/'study.yaml').write_text('study: synthetic\nrecipe: recipe.py\n')
    (source/'recipe.py').write_text('pass\n')
    runtime = materialize_study(source, tmp_path/'unrelated-runtime-leaf')
    (runtime/'docs').mkdir()
    (runtime/'docs/source_refs.json').write_text('{}')
    monkeypatch.chdir(checkout)
    reader = importlib.import_module('studies.'+name+'.flow.process.run')
    session = SimpleNamespace(output=tmp_path/'new-output', bind_declarations=lambda *a: None,
        sha=lambda *a: 'fixture', read_json=lambda *a: None)
    def check(study, execution, **kwargs):
        assert study == runtime and execution == checkout
        raise ValueError('verified original source gate reached')
    monkeypatch.setattr(validation, 'check_control_refs', check)
    if kind == 'paired':
        monkeypatch.setattr(reader, 'check_control_refs', check)
        monkeypatch.setattr(reader, 'PublishedReadout', lambda *a: session)
    else:
        monkeypatch.setattr(reader, '_native_session', lambda *a: session)
    with pytest.raises(ValueError, match='verified original source gate reached'):
        if kind == 'paired': reader.readout(session.output, study_dir=runtime)
        else: reader.native_readout(checkout, runtime, session.output)
    assert not session.output.exists()
