"""Synthetic pilot wiring: language, isolation of gain selection and correctness gates."""

import json

import pytest
import torch

from studies.trajectory_observation.structure import pilot


def test_language_rows_are_distributions_over_branching_successors():
    language = pilot.make_language(6, 2, 2.0, torch.Generator().manual_seed(0))
    assert torch.allclose(language.sum(-1), torch.ones(6, 6))
    assert ((language > 0).sum(-1) == 2).all()
    rows = pilot.sample(language, 6, 4, 10, torch.Generator().manual_seed(1))
    assert (rows[:, 0] == 6).all() and rows[:, 1:].max() < 6
    for row in rows:
        for position in range(3, 10):
            assert language[row[position - 2], row[position - 1], row[position]] > 0


def test_tiny_seed_keeps_ar_output_and_selects_gain_on_fit_prompts_only(record_property):
    assert torch.cuda.is_available(), 'pilot training requires real CUDA; no CPU fallback'
    config = json.loads(json.dumps(pilot.DEFAULT))
    config.update(content_vocab=8, max_new_tokens=6, trees=[{'num_masks': 2, 'branches': [3, 2]}])
    config['model'].update(hidden_size=16, intermediate_size=32, max_position_embeddings=64)
    config['train'].update(sequences=32, length=16, steps=3, batch=4)
    config['prompts'] = {'fit': 2, 'holdout': 2, 'cross': 2, 'length': 8}
    config['variants'] = [list(variant) for variant in pilot.SUFFIX]
    result = pilot.run_seed(config, 5, device='cuda:0')
    record_property('cuda_device', torch.cuda.get_device_name(0))
    entry = result['trees'][0]
    assert set(entry['fit_grid']['esp_rejection']) == {'0.0', '0.25', '0.5', '0.75', '1.0'}
    for base in pilot.BASES:
        for tag in ('none', 'rejection', 'top1', 'top1_rejection'):
            assert entry['selected_gain'][f'{base}_{tag}'] in pilot.JOINT_GAINS
    names = ('trajectory', *(f'{arm}{suffix}' for suffix in pilot.SUFFIX.values() for arm in ('observation', 'joint')))
    for split in ('holdout', 'cross'):
        body = entry['splits'][split]
        arms = body['arms']
        assert set(arms) == {f'{b}_{a}' for b in pilot.BASES for a in names}
        assert all(arm['ar_equal_all'] and arm['argmax_all_passed'] for arm in arms.values())
        assert arms['esp_joint_rejection']['gain'] == entry['selected_gain']['esp_rejection']
        assert arms['ema_joint_rejection']['condition'] == 'rejection'
        assert arms['esp_joint_top1_rejection']['form'] == 'top1'
        assert arms['esp_joint_top1']['gain'] == entry['selected_gain']['esp_top1']
        assert all(len(rows) == 2 for rows in body['rows'].values())
        for pairs in (body['paired_vs_esp_trajectory'], body['paired_vs_own_trajectory']):
            assert all(pair['incomparable'] == 0 for pair in pairs.values())


@pytest.mark.parametrize('mode', ['correction', 'affine'])
def test_tiny_correction_protocols_keep_fits_and_argmax_on_cuda(mode, record_property):
    assert torch.cuda.is_available(), 'correction wiring requires real CUDA; no CPU fallback'
    config = json.loads(json.dumps(pilot.DEFAULT))
    config.update(content_vocab=8, max_new_tokens=4, variants=[], trees=[{'num_masks': 1, 'branches': [2]}])
    config[mode] = True
    config['model'].update(hidden_size=8, intermediate_size=16, num_hidden_layers=1, max_position_embeddings=64)
    config['train'].update(sequences=16, length=12, steps=1, batch=2)
    config['prompts'] = {'fit': 4, 'holdout': 1, 'cross': 1, 'length': 4}
    result = pilot.run_seed(config, 5, device='cuda:0')
    record_property('cuda_device', torch.cuda.get_device_name(0))
    assert result['fit']['rows'] == 12
    assert all(arm['argmax_all_passed'] and arm['ar_equal_all']
               for split in result['trees'][0]['splits'].values() for arm in split['arms'].values())
    if mode == 'affine':
        assert all(torch.isfinite(torch.tensor(value)) for value in result['fit']['mse'].values())


@pytest.mark.parametrize('tree', [{'num_masks': 1, 'branches': [3]}, {'num_masks': 2, 'branches': [3, 2]},
                                  {'num_masks': 3, 'branches': [2, 2, 2]}])
def test_oracle_masks_accept_full_depth_after_prefill(tree, record_property):
    assert torch.cuda.is_available(), 'pilot oracle validation requires real CUDA; no CPU fallback'
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder, ar_generate
    from spec_mtp.structure.algorithm.mask import build_mask_provider
    from spec_mtp.structure.model.hf_causal import HFCausalLM

    torch.manual_seed(3)
    model = HFCausalLM(tiny_config={'vocab_size': 50, 'hidden_size': 32, 'intermediate_size': 64,
                                    'num_layers': 2, 'num_heads': 4, 'num_kv_heads': 2,
                                    'max_position_embeddings': 128})
    model.hf.float().to('cuda:0')
    prompt = torch.randint(0, 50, (1, 9), generator=torch.Generator().manual_seed(4)).to('cuda:0')
    eval_cfg = {'max_new_tokens': 25, 'temperature': 0.0, 'stop_on_eos': False}
    reference = ar_generate(model, prompt, eval_cfg)[0]
    spec = pilot.spec_for('esp', tree, 0.0)
    # A random model often repeats one token; parent-repeat pruning would ban the true token.
    spec['tree']['pruning'] = False
    provider = pilot.OracleMaskProvider(build_mask_provider(spec, model.embedding_table), model.embedding_table,
                                        reference[0, prompt.shape[1]:].tolist())
    output, stats = SpecDecoder(model, spec, eval_cfg, mask_policy=provider).generate(prompt)
    assert torch.equal(output, reference)
    k = tree['num_masks']
    later = [len(row['accepted_nodes']) for row in stats['trace'][1:] if not row['stopped']]
    assert later and all(accepted == k for accepted in later)
    model.close()
    record_property('cuda_device', torch.cuda.get_device_name(0))


@pytest.fixture
def owned_pilot_session(monkeypatch):
    from types import SimpleNamespace

    config = json.loads(json.dumps(pilot.DEFAULT))
    config['trees'] = []
    state = {'owned': False, 'primary': None, 'cleanup': None, 'closed': []}

    class Rows:
        shape = (1, 4)

        def __getitem__(self, item):
            return self

    class Session:
        def close(self):
            state['closed'].append(True)
            if state['cleanup'] is not None:
                raise state['cleanup']

    def sample(*args, **kwargs):
        if state['owned'] and state['primary'] is not None:
            raise state['primary']
        return Rows()

    def train(*args, **kwargs):
        state['owned'] = True
        return Session(), []

    monkeypatch.setattr(pilot, 'torch', SimpleNamespace(
        Generator=torch.Generator, cat=lambda rows: Rows()))
    monkeypatch.setattr(pilot, 'make_language', lambda *args: None)
    monkeypatch.setattr(pilot, 'sample', sample)
    monkeypatch.setattr(pilot, 'train_model', train)
    monkeypatch.setattr(pilot, 'ar_generate', lambda *args: (None, {}))
    return config, state


@pytest.mark.parametrize('fault', ['close_only', 'body_error', 'interrupt', 'ambient_close_only'])
def test_run_seed_cleanup_preserves_primary(owned_pilot_session, fault):
    config, state = owned_pilot_session
    cleanup = OSError('injected session close failure')
    primary = (None if fault in ('close_only', 'ambient_close_only') else
               KeyboardInterrupt('body interrupted') if fault == 'interrupt'
               else ValueError('body failed'))
    state.update(primary=primary, cleanup=cleanup)
    expected = cleanup if primary is None else primary

    ambient = LookupError('caller exception must not become producer primary')
    with pytest.raises(type(expected)) as caught:
        if fault == 'ambient_close_only':
            try:
                raise ambient
            except LookupError:
                pilot.run_seed(config, 5, device='cuda:0')
        else:
            pilot.run_seed(config, 5, device='cuda:0')
    assert not hasattr(ambient, 'cleanup_errors')

    assert caught.value is expected
    assert state['closed'] == [True]
    assert getattr(expected, 'cleanup_errors', ()) == (
        () if primary is None else (cleanup,))


@pytest.mark.parametrize('fault', ['close_only', 'body_error', 'interrupt', 'ambient_close_only'])
def test_run_oracle_cleanup_preserves_primary(owned_pilot_session, fault):
    config, state = owned_pilot_session
    cleanup = OSError('injected session close failure')
    primary = (None if fault in ('close_only', 'ambient_close_only') else
               KeyboardInterrupt('body interrupted') if fault == 'interrupt'
               else ValueError('body failed'))
    state.update(primary=primary, cleanup=cleanup)
    expected = cleanup if primary is None else primary

    ambient = LookupError('caller exception must not become producer primary')
    with pytest.raises(type(expected)) as caught:
        if fault == 'ambient_close_only':
            try:
                raise ambient
            except LookupError:
                pilot.run_oracle(config, 5, device='cuda:0')
        else:
            pilot.run_oracle(config, 5, device='cuda:0')
    assert not hasattr(ambient, 'cleanup_errors')

    assert caught.value is expected
    assert state['closed'] == [True]
    assert getattr(expected, 'cleanup_errors', ()) == (
        () if primary is None else (cleanup,))


def test_run_seed_timer_failure_still_closes_session(
        owned_pilot_session, monkeypatch):
    config, state = owned_pilot_session
    primary = ValueError('post-training timer failed')
    cleanup = OSError('injected session close failure')
    state['cleanup'] = cleanup

    class Clock:
        @staticmethod
        def perf_counter():
            if state['owned']:
                raise primary
            return 0.0

    monkeypatch.setattr(pilot, 'time', Clock)
    with pytest.raises(ValueError) as caught:
        pilot.run_seed(config, 5, device='cuda:0')

    assert caught.value is primary
    assert state['closed'] == [True]
    assert primary.cleanup_errors == (cleanup,)


@pytest.mark.parametrize('fault', ['normal', 'body_plus_close'])
def test_tiny_oracle_producer_lifecycle_on_cuda(
        monkeypatch, record_property, fault):
    assert torch.cuda.is_available(), 'real CUDA required; no CPU fallback'
    device = torch.device('cuda:0')
    name = torch.cuda.get_device_name(device)
    print(f'pilot oracle lifecycle device={device} name={name} fault={fault}')
    record_property('cuda_device', str(device))
    record_property('cuda_name', name)

    config = json.loads(json.dumps(pilot.DEFAULT))
    config.update(content_vocab=8, max_new_tokens=4,
                  trees=[{'num_masks': 1, 'branches': [2]}])
    config['model'].update(hidden_size=8, intermediate_size=16,
                           num_hidden_layers=1, max_position_embeddings=64)
    config['train'].update(sequences=16, length=12, steps=1, batch=2)
    config['prompts'] = {'fit': 4, 'holdout': 1, 'cross': 1, 'length': 4}
    primary = ValueError('injected failure after actual CUDA AR')
    cleanup = OSError('injected failure after actual session close')
    owned, hooks, close_calls, observed = [], [], [], set()
    original_train, original_ar = pilot.train_model, pilot.ar_generate

    def watch(module, args, kwargs):
        observed.add(str(kwargs['inputs_embeds'].device))

    def train(*args, **kwargs):
        session, losses = original_train(*args, **kwargs)
        assert session.device == device
        actual_close = session.close
        owned.append((session, actual_close))
        hooks.append(session.hf.register_forward_pre_hook(
            watch, with_kwargs=True))

        def close():
            close_calls.append(True)
            actual_close()
            if fault == 'body_plus_close':
                raise cleanup

        monkeypatch.setattr(session, 'close', close)
        return session, losses

    def ar(*args, **kwargs):
        result = original_ar(*args, **kwargs)
        if fault == 'body_plus_close':
            raise primary
        return result

    monkeypatch.setattr(pilot, 'train_model', train)
    monkeypatch.setattr(pilot, 'ar_generate', ar)
    try:
        if fault == 'body_plus_close':
            with pytest.raises(ValueError) as caught:
                pilot.run_oracle(config, 5, device=str(device))
            assert caught.value is primary
            assert primary.cleanup_errors == (cleanup,)
        else:
            result = pilot.run_oracle(config, 5, device=str(device))
            assert result['seed'] == 5 and len(result['trees']) == 1
            splits = result['trees'][0]['splits']
            assert set(splits) == {'holdout', 'cross'}
            for arms in splits.values():
                assert set(arms) == {'esp_trajectory', 'esp_oracle'}
                for arm in arms.values():
                    assert len(arm['rows']) == 1
                    assert all(row['argmax_passed'] is True
                               for row in arm['rows'])
        assert close_calls == [True]
        assert len(owned) == 1 and owned[0][0].closed is True
        assert observed == {str(device)}
        record_property('forward_devices', sorted(observed))
        record_property('session_closed', True)
    finally:
        for hook in hooks:
            hook.remove()
        for session, actual_close in owned:
            if not session.closed:
                actual_close()


@pytest.mark.parametrize('fault', ['normal', 'untied', 'getter_error', 'untied_and_close',
    'getter_and_close', 'interrupt_and_close', 'ambient_success'])
def test_train_model_session_admission_preserves_primary_and_borrowed_native(monkeypatch, fault):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys

    # Zero iterations isolate session admission; this is not an accepted Study training run.
    config = json.loads(json.dumps(pilot.DEFAULT))
    config['train']['steps'] = 0
    device = SimpleNamespace(type='cuda', index=0)
    natives, sessions, close_calls = [], [], []
    getter_error = (KeyboardInterrupt('injected tied check interruption')
                    if fault == 'interrupt_and_close'
                    else RuntimeError('injected tied getter failure'))
    cleanup = OSError('injected borrowed session close failure')
    close_fails = fault in ('untied_and_close', 'getter_and_close', 'interrupt_and_close')
    ambient = LookupError('ambient caller error is not a train_model primary')

    class Native:
        def __init__(self, model_config):
            self.config = model_config
            natives.append(self)

        def to(self, target):
            assert target is device
            return self

        def parameters(self):
            return ()

        def train(self):
            return self

        def eval(self):
            return self

    class Session:
        def __init__(self, *, module):
            self.hf, self.closed = module, False
            sessions.append(self)

        @property
        def tied_embeddings(self):
            if fault in ('getter_error', 'getter_and_close', 'interrupt_and_close'):
                raise getter_error
            return fault not in ('untied', 'untied_and_close')

        def close(self):
            close_calls.append(True)
            self.closed = True
            if close_fails:
                raise cleanup

    proxy = SimpleNamespace(
        device=lambda value: device,
        cuda=SimpleNamespace(is_available=lambda: True),
        random=SimpleNamespace(fork_rng=lambda **options: nullcontext()),
        manual_seed=lambda value: None,
        optim=SimpleNamespace(AdamW=lambda *args, **options: object(),
            lr_scheduler=SimpleNamespace(CosineAnnealingLR=lambda *args: object())))
    monkeypatch.setattr(pilot, 'torch', proxy)
    monkeypatch.setattr(pilot, 'HFCausalLM', Session)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        LlamaConfig=lambda **options: SimpleNamespace(**options), LlamaForCausalLM=Native))

    if fault in ('normal', 'ambient_success'):
        if fault == 'ambient_success':
            try:
                raise ambient
            except LookupError:
                returned, losses = pilot.train_model(config, object(), 5, device='cuda:0')
        else:
            returned, losses = pilot.train_model(config, object(), 5, device='cuda:0')
        assert len(sessions) == 1 and returned is sessions[0]
        assert losses == [] and returned.closed is False and close_calls == []
    else:
        expected_type = ValueError if fault in ('untied', 'untied_and_close') else type(getter_error)
        with pytest.raises(expected_type) as caught:
            pilot.train_model(config, object(), 5, device='cuda:0')
        if fault in ('untied', 'untied_and_close'):
            assert str(caught.value) == 'pilot requires tied embeddings'
        else:
            assert caught.value is getter_error
        assert close_calls == [True] and sessions[0].closed is True
        assert getattr(caught.value, 'cleanup_errors', ()) == ((cleanup,) if close_fails else ())
    assert len(natives) == len(sessions) == 1
    assert sessions[0].hf is natives[0]
    assert not hasattr(ambient, 'cleanup_errors')


@pytest.mark.parametrize('fault', ['cleanup_ok', 'cleanup_error'])
def test_train_model_rejects_untied_cuda_session_without_destroying_native(
        monkeypatch, record_property, fault):
    import transformers

    assert torch.cuda.is_available(), 'real CUDA required; no CPU fallback'
    device = torch.device('cuda:0')
    name = torch.cuda.get_device_name(device)
    print(f'pilot admission device={device} name={name} fault={fault}')
    record_property('cuda_device', str(device))
    record_property('cuda_name', name)
    config = json.loads(json.dumps(pilot.DEFAULT))
    config.update(content_vocab=8, max_new_tokens=4,
                  trees=[{'num_masks': 1, 'branches': [2]}])
    config['model'].update(hidden_size=8, intermediate_size=16,
                           num_hidden_layers=1, max_position_embeddings=64)
    config['train'].update(sequences=16, length=12, steps=1, batch=2)
    config['prompts'] = {'fit': 4, 'holdout': 1, 'cross': 1, 'length': 4}
    generator = torch.Generator(device='cpu').manual_seed(5)
    content = config['content_vocab']
    first = pilot.make_language(content, config['branching'], config['sharpness'], generator)
    second = pilot.make_language(content, config['branching'], config['sharpness'], generator)
    data = torch.cat([pilot.sample(first, content, 8, 12, generator),
                      pilot.sample(second, content + 1, 8, 12, generator)])
    _ = transformers.LlamaConfig
    native_factory, session_factory = transformers.LlamaForCausalLM, pilot.HFCausalLM
    provisional, hooks, close_calls, observed = [], [], [], []
    cleanup = OSError('injected failure after actual borrowed close')

    def watch(module, args, output):
        if isinstance(module, native_factory):
            observed.append(str(output.loss.device))

    def wrap(*, module):
        embedding = module.get_input_embeddings().weight
        assert embedding.device == device and module.config.tie_word_embeddings is True
        module.get_output_embeddings().weight = torch.nn.Parameter(
            module.get_output_embeddings().weight.detach().clone())
        output = module.get_output_embeddings().weight
        assert embedding.data_ptr() != output.data_ptr()
        session = session_factory(module=module)
        assert session.tied_embeddings is False
        actual_close = session.close
        provisional.append({'session': session, 'native': module,
            'weights': (embedding, output),
            'values': (embedding.detach().clone(), output.detach().clone()),
            'pointers': tuple((weight.data_ptr(), weight.untyped_storage().data_ptr())
                              for weight in (embedding, output)),
            'actual_close': actual_close})

        def close():
            close_calls.append(True)
            actual_close()
            if fault == 'cleanup_error':
                raise cleanup

        monkeypatch.setattr(session, 'close', close)
        return session

    hooks.append(torch.nn.modules.module.register_module_forward_hook(watch))
    monkeypatch.setattr(pilot, 'HFCausalLM', wrap)
    try:
        with pytest.raises(ValueError, match='pilot requires tied embeddings') as caught:
            pilot.train_model(config, data, 5, device=str(device))
        assert str(caught.value) == 'pilot requires tied embeddings'
        assert close_calls == [True] and len(provisional) == 1
        assert getattr(caught.value, 'cleanup_errors', ()) == (
            (cleanup,) if fault == 'cleanup_error' else ())
        saved = provisional[0]
        session, native = saved['session'], saved['native']
        assert session.closed is True and session.hf is native
        current = (native.get_input_embeddings().weight, native.get_output_embeddings().weight)
        for index, weight in enumerate(current):
            assert weight is saved['weights'][index] and weight.device == device
            assert (weight.data_ptr(), weight.untyped_storage().data_ptr()) == saved['pointers'][index]
            assert torch.equal(weight, saved['values'][index])
        assert observed and set(observed) == {str(device)}
        record_property('cuda_train_forward_count', len(observed))
        record_property('forward_devices', sorted(set(observed)))
        record_property('borrowed_native_preserved', True)
        record_property('session_closed', True)
    finally:
        for hook in hooks:
            hook.remove()
        for saved in provisional:
            if not saved['session'].closed:
                saved['actual_close']()
