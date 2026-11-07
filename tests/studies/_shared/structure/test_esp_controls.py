"""ESP metadata restoration and tiny CUDA model-local shape engineering contracts."""
import json
from types import SimpleNamespace

import pytest
import torch

from studies._shared.structure import esp_controls as protocol


@pytest.mark.parametrize('shape', [None, 0, 1, 'true'])
def test_esp_registration_rejects_non_boolean_shape_before_changing_registry(monkeypatch, shape):
    monkeypatch.setattr(protocol.AlgorithmRegistry, 'register', lambda *a: pytest.fail('invalid policy registered'))
    with pytest.raises(ValueError, match='explicit boolean'):
        protocol.register_controls(shape=shape)


def test_esp_numerics_rejects_non_bf16_before_backend_or_asset_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'use_deterministic_algorithms', lambda *a, **kw: pytest.fail('invalid dtype changed backend'))
    adapter = object.__new__(protocol.NumericsEval)
    adapter.config = SimpleNamespace(setting=lambda *a: {'deterministic': False, 'bf16_reduced_precision_reduction': True})
    with pytest.raises(AssertionError):
        adapter.run(None, SimpleNamespace(module=SimpleNamespace(dtype=torch.float32)),
                    SimpleNamespace(assets_dir=tmp_path), None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('reduced,deterministic', [(True,False),(False,False),(True,True),(False,True)])
@pytest.mark.parametrize('failure', [False,True])
def test_bf16_controls_restore_and_record(monkeypatch, tmp_path, reduced, deterministic, failure):
    numerics = protocol
    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    old_det = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    def run(*args):
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is reduced
        assert torch.are_deterministic_algorithms_enabled() is deterministic
        if failure:
            raise RuntimeError('injected native failure')
        return {'ok':True}
    monkeypatch.setattr(numerics.UntimedEval, 'run', run)
    algo = numerics.NumericsEval.__new__(numerics.NumericsEval)
    controls = {'bf16_reduced_precision_reduction':reduced,'deterministic':deterministic}
    algo.config = SimpleNamespace(setting=lambda _: controls)
    args = (None,SimpleNamespace(module=SimpleNamespace(dtype=torch.bfloat16)),SimpleNamespace(assets_dir=tmp_path),None)
    if failure:
        with pytest.raises(RuntimeError,match='injected native failure'):
            algo.run(*args)
    else:
        assert algo.run(*args)=={'ok':True}
    assert json.loads((tmp_path/'numerics_controls.json').read_text())==dict(controls,dtype='torch.bfloat16')
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is old_reduced
    assert torch.are_deterministic_algorithms_enabled() is old_det
    assert torch.is_deterministic_algorithms_warn_only_enabled() is old_warn


@pytest.mark.parametrize('shape', [(1,4,16),(2,3,16),(16,),(0,16)])
def test_rowwise_bf16_is_partition_and_permutation_invariant(monkeypatch, shape):
    shape_control = protocol
    assert torch.cuda.is_available(), 'tiny shape engineering correctness requires actual CUDA'
    device = torch.device('cuda:0')
    torch.manual_seed(123)
    layer = torch.nn.Linear(16,7,bias=True,dtype=torch.bfloat16,device=device)
    before = {k:v.clone() for k,v in layer.state_dict().items()}
    x = torch.randn(shape,dtype=torch.bfloat16,device=device)
    print(f'tiny shape engineering tensor.device={x.device} name={torch.cuda.get_device_name(x.device)}', flush=True)
    actual = shape_control.rowwise_linear(layer.forward,x)
    flat = x.reshape(-1,16)
    if len(flat):
        perm = torch.arange(len(flat)-1,-1,-1,device=device)
        shuffled = shape_control.rowwise_linear(layer.forward,flat[perm])[perm]
        assert torch.equal(actual.reshape(-1,7),shuffled)
        prefix = shape_control.rowwise_linear(layer.forward,flat[:1])
        assert torch.equal(actual.reshape(-1,7)[:1],prefix)
    assert actual.dtype==torch.bfloat16 and actual.shape==(*shape[:-1],7)
    assert all(torch.equal(layer.state_dict()[k],v) for k,v in before.items())
    torch.cuda.synchronize(device)

@pytest.mark.parametrize('enabled', [False,True])
@pytest.mark.parametrize('failure', [False,True])
def test_shape_patch_is_local_and_restores_on_failure(monkeypatch,tmp_path,enabled,failure):
    shape_control = protocol
    assert torch.cuda.is_available(), 'tiny patch engineering correctness requires actual CUDA'
    device = torch.device('cuda:0')
    net = torch.nn.Sequential(torch.nn.Linear(16,8,dtype=torch.bfloat16,device=device),torch.nn.SiLU(),torch.nn.Linear(8,4,dtype=torch.bfloat16,device=device))
    layers = [m for m in net.modules() if isinstance(m,torch.nn.Linear)]
    before = {k:v.clone() for k,v in net.state_dict().items()}
    def parent(*args):
        assert all(('forward' in m.__dict__)==enabled for m in layers)
        inputs = torch.ones(1,2,16,dtype=torch.bfloat16,device=device)
        print(f'tiny local patch tensor.device={inputs.device} name={torch.cuda.get_device_name(inputs.device)}', flush=True)
        assert net(inputs).dtype==torch.bfloat16
        if failure:
            raise RuntimeError('injected shape failure')
        return {'ok':True}
    monkeypatch.setattr(shape_control.NumericsEval,'run',parent)
    algo = shape_control.ShapeEval.__new__(shape_control.ShapeEval)
    algo.config = SimpleNamespace(setting=lambda _: {'linear_rows':enabled})
    args = (None,SimpleNamespace(module=SimpleNamespace(hf=net,dtype=torch.bfloat16)),SimpleNamespace(assets_dir=tmp_path),None)
    if failure:
        with pytest.raises(RuntimeError,match='injected shape failure'):
            algo.run(*args)
    else:
        assert algo.run(*args)=={'ok':True}
    assert all('forward' not in m.__dict__ for m in layers)
    assert all(torch.equal(net.state_dict()[k],v) for k,v in before.items())
    torch.cuda.synchronize(device)


@pytest.mark.parametrize('kind', ['policy', 'argmax', 'revalidation'])
@pytest.mark.parametrize('fault', ['none', 'source_drift', 'bad_marker', 'outside_assets'])
def test_esp_registration_binds_snapshot_source_and_run_assets_without_loading(tmp_path, monkeypatch, kind, fault):
    from spec_mtp.structure.artifact.layout import materialize_study
    from spec_mtp.structure.model.factory import ModelRegistry
    from studies._shared.structure import recipe, model
    checkout = tmp_path/'checkout'; source = checkout/'studies/esp'
    source.mkdir(parents=True)
    (source/'study.yaml').write_text('study: synthetic\nrecipe: recipe.py\n')
    (source/'recipe.py').write_text('pass\n')
    runtime = materialize_study(source, tmp_path/'unrelated-external-name')
    classes, builders = [], []
    monkeypatch.setattr(protocol.AlgorithmRegistry, 'register', lambda *args: classes.append(args[-1]))
    monkeypatch.setattr(ModelRegistry, 'register', lambda *args: builders.append(args[-1]))
    assets = runtime/'runs/a/assets'
    if fault == 'source_drift': (source/'recipe.py').write_text('changed source\n')
    elif fault == 'bad_marker': (runtime/'study_source.json').write_text('{}')
    elif fault == 'outside_assets': assets = tmp_path/'model-cache'
    if fault != 'none':
        with pytest.raises(ValueError): recipe.register_esp(runtime, admission_assets=assets, kind=kind)
        assert classes == builders == []
    else:
        recipe.register_esp(runtime, admission_assets=assets, kind=kind)
        selected = classes[0]
        assert selected.study_root == runtime and selected.execution_root == checkout
        assert selected.baseline_root == checkout/'studies/esp_table1_bf16_full_argmax'
        assert selected.audit_argmax is (kind != 'policy')
        if kind == 'revalidation':
            assert builders[0].keywords['admission_assets'] == assets
        else: assert builders == [model.build_cuda]


@pytest.mark.parametrize('name,kind', [('esp_table1_bf16_full','policy'), ('esp_table1_bf16_qwen_protocol','policy'),
    ('esp_table1_bf16_full_argmax','argmax'), ('esp_table1_nonofficial_revalidation','revalidation')])
@pytest.mark.parametrize('failure', [False, True])
def test_full_esp_recipe_native_gate_precedes_all_overrides_and_import_is_inert(tmp_path, monkeypatch, name, kind, failure):
    from pathlib import Path
    from spec_mtp.structure.make.recipe import load_recipe
    from studies._shared.structure import recipe
    source = Path(__file__).resolve().parents[4]/'studies'/name
    events = []
    def gate(ctx):
        events.append('native')
        if failure: raise RuntimeError('original source gate refused')
    monkeypatch.setattr(recipe, 'register', gate)
    monkeypatch.setattr(recipe, 'register_esp', lambda *a, **kw: events.append((a, kw)))
    loaded = load_recipe(source, {'recipe':'recipe.py'})
    assert events == []
    ctx = SimpleNamespace(study_dir=source, assets_dir=tmp_path/'run-assets')
    if failure:
        with pytest.raises(RuntimeError, match='original source'): loaded.register(ctx)
        assert events == ['native']
    else:
        loaded.register(ctx)
        assert events == ['native', ((source,), {'admission_assets':ctx.assets_dir,'kind':kind})]


@pytest.mark.parametrize('failure', ['none', 'before_wait', 'after_wait'])
def test_esp_revalidation_source_gate_and_wait_precede_loader_with_separate_asset_roots(tmp_path, monkeypatch, failure):
    from studies._shared.structure import recipe, model, validation, resource_admission
    events = []
    def check(*a):
        events.append('refs')
        if (failure == 'before_wait' and len(events) == 1) or (failure == 'after_wait' and len(events) == 3):
            raise ValueError('source identity drift')
    monkeypatch.setattr(validation, 'check_source_refs', check)
    assets, shared = tmp_path/'run-assets', tmp_path/'model-cache'
    monkeypatch.setattr(resource_admission, 'wait_for_admission', lambda target, **kw: events.append(('wait',target,kw['initial'])) or {'passed':True})
    monkeypatch.setattr(model, 'build_cuda', lambda cfg, target, meta: events.append(('model',target)) or SimpleNamespace(meta={}))
    if failure != 'none':
        with pytest.raises(ValueError, match='source identity drift'):
            recipe._build_esp_revalidation_cuda(None, shared, study_root=tmp_path, execution_root=tmp_path, admission_assets=assets)
        assert not any(isinstance(event,tuple) and event[0]=='model' for event in events)
    else:
        result = recipe._build_esp_revalidation_cuda(None, shared, study_root=tmp_path, execution_root=tmp_path, admission_assets=assets)
        assert result.meta['initial_resource_admission']['passed']
        assert events == ['refs',('wait',assets,True),'refs',('model',shared)]


@pytest.mark.parametrize('audit', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_esp_numeric_policy_keeps_backend_and_scopes_optional_observer(tmp_path, monkeypatch, audit, failure):
    from contextlib import contextmanager
    events = []
    @contextmanager
    def observe(assets):
        assert assets == tmp_path
        events.append('observe-enter')
        try: yield
        finally: events.append('observe-exit')
    monkeypatch.setattr(protocol, 'observe_argmax', observe)
    def run(*args):
        events.append('native')
        if failure: raise RuntimeError('native failure')
        return 'complete'
    monkeypatch.setattr(protocol.UntimedEval, 'run', run)
    cls = protocol.ArgmaxEval if audit else protocol.NumericPolicyEval
    adapter = object.__new__(cls)
    adapter.study_root = tmp_path/'studies/policy'
    adapter.study_root.mkdir(parents=True)
    adapter.execution_root = tmp_path
    adapter.config = SimpleNamespace(setting=lambda *a: {'warmup':False})
    before = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, torch.are_deterministic_algorithms_enabled())
    if failure:
        with pytest.raises(RuntimeError, match='native failure'):
            adapter.run(None,SimpleNamespace(module=SimpleNamespace(dtype=torch.bfloat16)),SimpleNamespace(assets_dir=tmp_path),None)
    else:
        assert adapter.run(None,SimpleNamespace(module=SimpleNamespace(dtype=torch.bfloat16)),SimpleNamespace(assets_dir=tmp_path),None)=='complete'
    assert events == (['observe-enter','native','observe-exit'] if audit else ['native'])
    assert before == (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, torch.are_deterministic_algorithms_enabled())
    assert json.loads((tmp_path/'numeric_policy.json').read_text()) == {'dtype':'torch.bfloat16',
        'bf16_reduced_precision_reduction':before[0],'deterministic':before[1]}


@pytest.mark.parametrize('fault', ['none','dtype','backend','model','count','prompt','category','incomplete'])
def test_nonofficial_runtime_guard_preserves_original_model_prompts_count_and_progress(tmp_path, monkeypatch, fault):
    from studies._shared.structure import evidence, validation, resource_admission
    import spec_mtp.structure.algorithm.decode.loop as decoding
    import copy
    baseline = tmp_path/'baseline'; (baseline/'docs').mkdir(parents=True)
    assets = tmp_path/'run-assets'; assets.mkdir()
    prompts = [{'input_ids':[i],'category':'a'} for i in range(480)]
    (baseline/'docs/prompts_manifest.json').write_text(json.dumps({'rendered_prompts':prompts}))
    (baseline/'docs/protocol_evidence_v1.json').write_text(json.dumps({'cells':[{'model_digest':'original-weights'}]}))
    saved = {'dtype':'torch.bfloat16','backend':'eager','model_digest':'original-weights','prompts':copy.deepcopy(prompts)}
    if fault in ('dtype','backend'): saved[fault]='changed'
    elif fault=='model': saved['model_digest']='changed'
    elif fault=='count': saved['prompts'].pop()
    elif fault=='prompt': saved['prompts'][0]['input_ids']=[999]
    elif fault=='category': saved['prompts'][0]['category']='changed'
    events=[]
    monkeypatch.setattr(validation,'check_source_refs',lambda *a: events.append('refs'))
    monkeypatch.setattr(resource_admission,'wait_for_admission',lambda target,**kw: events.append(('wait',target,kw['initial'])) or {'passed':True})
    monkeypatch.setattr(evidence,'capture',lambda *a: saved)
    monkeypatch.setattr(decoding.SpecDecoder,'generate',lambda *a: ('output',{'committed':1,'decode_calls':1}))
    def parent(*args):
        evidence.capture(None)
        for index in range(479 if fault=='incomplete' else 480):
            prompt=[SimpleNamespace(tolist=lambda index=index:[index])]
            decoding.SpecDecoder.generate(SimpleNamespace(nominal_block_complexity=10),prompt)
        return 'complete'
    monkeypatch.setattr(protocol.ArgmaxEval,'run',parent)
    adapter=object.__new__(protocol.RevalidationEval)
    adapter.study_root=adapter.execution_root=tmp_path
    adapter.baseline_root=baseline
    arguments=(None,SimpleNamespace(meta={'initial_resource_admission':{'passed':True}}),SimpleNamespace(assets_dir=assets),None)
    if fault=='none':
        assert adapter.run(*arguments)=='complete'
        progress=json.loads((assets/'revalidation_progress.json').read_text())
        assert progress=={'completed':480,'total':480,'BC':10,'last_committed':1,'last_decode_calls':1}
        assert events[0]==events[-1]=='refs'
        assert events.count(('wait',assets,False))==480
    else:
        with pytest.raises(AssertionError): adapter.run(*arguments)


@pytest.mark.parametrize('kind', ['policy', 'argmax'])
@pytest.mark.parametrize('when', ['before', 'after'])
def test_esp_policy_rechecks_source_snapshot_before_and_after_native_execution(tmp_path, monkeypatch, kind, when):
    from spec_mtp.structure.artifact.layout import materialize_study
    from studies._shared.structure import recipe
    from spec_mtp.structure.model.factory import ModelRegistry
    source=tmp_path/'checkout/studies/policy';source.mkdir(parents=True)
    (source/'study.yaml').write_text('study: synthetic\nrecipe: recipe.py\n')
    (source/'recipe.py').write_text('pass\n')
    runtime=materialize_study(source,tmp_path/'external-runtime')
    assets=runtime/'runs/a/assets';assets.mkdir(parents=True)
    selected=[]
    monkeypatch.setattr(protocol.AlgorithmRegistry,'register',lambda *a:selected.append(a[-1]))
    monkeypatch.setattr(ModelRegistry,'register',lambda *a:None)
    recipe.register_esp(runtime,admission_assets=assets,kind=kind)
    def drift(): (source/'recipe.py').write_text('changed source\n')
    calls=[]
    def run(*a):
        calls.append('native')
        drift()
        return 'complete'
    monkeypatch.setattr(protocol.UntimedEval,'run',run)
    if when=='before':drift()
    adapter=object.__new__(selected[0])
    adapter.config=SimpleNamespace(setting=lambda *a:{'warmup':False})
    with pytest.raises(ValueError):
        adapter.run(None,SimpleNamespace(module=SimpleNamespace(dtype=torch.bfloat16)),SimpleNamespace(assets_dir=assets),None)
    assert calls==([] if when=='before' else ['native'])
    if when=='before':assert not list(assets.iterdir())


@pytest.mark.parametrize('kind', ['policy', 'argmax', 'revalidation'])
def test_full_esp_adapter_combines_untimed_cuda_decode_and_optional_external_audit(tmp_path, monkeypatch, kind):
    from studies._shared.structure import algorithm
    from studies._shared.structure.config import _convert
    from spec_mtp.structure.model.hf_causal import HFCausalLM
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    import spec_mtp.structure.algorithm.decode.loop as decoding
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    print('ESP adapter combination device=cuda:0 name='+torch.cuda.get_device_name(0),flush=True)
    torch.manual_seed(7)
    model=HFCausalLM(tiny_config={'vocab_size':32,'hidden_size':16,'intermediate_size':32,
        'num_hidden_layers':1,'num_attention_heads':2,'num_key_value_heads':1,'max_position_embeddings':64},
        torch_dtype='bfloat16',attn_implementation='eager').to(device='cuda:0',dtype=torch.bfloat16)
    spec=_convert({'method':'esp','num_masks':1,'block_complexity':10,'impl':'naive',
        'tree':{'mode':'dynamic','pruning':True},'mask':{'init':'mean_prompt','lam':.1,'update':True}})
    decoder=decoding.SpecDecoder(model,spec,{'temperature':0.,'max_new_tokens':7,'stop_on_eos':False},
        mask_policy=build_mask_provider(spec,model.embedding_table))
    prompt=torch.tensor([[1,2,3]],device='cuda:0')
    expected,timed=decoder.generate(prompt)
    captured=[]
    class PrefixStop(RuntimeError): pass
    clock,generate,select=decoding.time,decoding.SpecDecoder.generate,decoding.select_token
    def parent(*args):
        if kind=='revalidation':
            from studies._shared.structure import evidence
            evidence.capture(None)
        output,stats=decoder.generate(prompt)
        decoding.ar_generate(model,prompt,{'temperature':0.,'max_new_tokens':1})
        assert stats['prefill_time']==stats['decode_time']==stats['draft_seconds']==stats['verify_seconds']==0
        assert stats['verification_argmax_audit']['checked_committed_tokens']==stats['new_tokens']==7
        assert stats['verification_argmax_audit']['passed']
        if kind=='revalidation':
            captured.append((output,stats))
            raise PrefixStop('bounded one-prompt engineering stop; incomplete480')
        return output,stats
    monkeypatch.setattr(algorithm.SpecMtpEvalAlgorithm,'run',parent)
    cls={'policy':protocol.NumericPolicyEval,'argmax':protocol.ArgmaxEval,'revalidation':protocol.RevalidationEval}[kind]
    adapter=object.__new__(cls)
    adapter.study_root=tmp_path/'checkout/studies/esp';adapter.study_root.mkdir(parents=True)
    adapter.execution_root=tmp_path/'checkout'
    adapter.config=SimpleNamespace(setting=lambda *a:{'warmup':False})
    assets=tmp_path/'assets';assets.mkdir()
    model_wrapper=SimpleNamespace(module=model,meta={'initial_resource_admission':{'passed':True}})
    if kind=='revalidation':
        from studies._shared.structure import evidence,validation,resource_admission
        adapter.baseline_root=adapter.execution_root/'studies/esp_table1_bf16_full_argmax'
        docs=adapter.baseline_root/'docs';docs.mkdir(parents=True)
        prompts=[{'input_ids':[1,2,3],'category':'a'} for _ in range(480)]
        (docs/'prompts_manifest.json').write_text(json.dumps({'rendered_prompts':prompts}))
        (docs/'protocol_evidence_v1.json').write_text(json.dumps({'cells':[{'model_digest':'fixture-weights'}]}))
        monkeypatch.setattr(evidence,'capture',lambda *a:{'dtype':'torch.bfloat16','backend':'eager',
            'model_digest':'fixture-weights','prompts':prompts})
        events=[]
        monkeypatch.setattr(validation,'check_source_refs',lambda *a:events.append('refs'))
        monkeypatch.setattr(resource_admission,'wait_for_admission',lambda target,**kw:events.append((target,kw['initial'])) or {'passed':True})
        with pytest.raises(PrefixStop,match='incomplete480'):
            adapter.run(None,model_wrapper,SimpleNamespace(assets_dir=assets),None)
        actual,stats=captured[0]
        assert events==['refs',(assets,False)]
        progress=json.loads((assets/'revalidation_progress.json').read_text())
        assert progress['completed']==1 and progress['total']==480 and progress.get('complete') is not True
        assert not (assets/'decode_observations.json').exists()
    else:
        actual,stats=adapter.run(None,model_wrapper,SimpleNamespace(assets_dir=assets),None)
    assert torch.equal(actual,expected)
    ignored={'prefill_time','decode_time','draft_seconds','verify_seconds','verification_argmax_audit'}
    assert {k:v for k,v in stats.items() if k not in ignored}=={k:v for k,v in timed.items() if k not in ignored}
    assert decoding.time is clock and decoding.SpecDecoder.generate is generate and decoding.select_token is select
    assert not model._forward_hooks
    path=assets/'argmax_audit.jsonl'
    assert path.exists() is (kind!='policy')
    if kind!='policy':
        lines=path.read_text().splitlines()
        assert len(lines)==1 and json.loads(lines[0])==stats['verification_argmax_audit']
    assert json.loads((assets/'numeric_policy.json').read_text())['dtype']=='torch.bfloat16'
