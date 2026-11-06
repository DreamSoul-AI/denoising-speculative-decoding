"""Model resource ownership and concrete KV selection."""

import pytest


from spec_mtp.structure.model import hf_causal as kv_cache

def test_kv_rejects_unsupported_cache_type():
    with pytest.raises(TypeError, match='Unsupported cache type'):
        list(kv_cache._layer_tensors(object()))


@pytest.mark.parametrize('owned', [True, False])
@pytest.mark.parametrize('fault', ['eval', 'config', 'vocab', 'max_positions',
                                  'cancel', 'cleanup', 'ok', 'ambient_ok'])
def test_session_constructor_closes_only_its_acquired_model_on_admission_failure(monkeypatch, owned, fault):
    import sys
    from types import SimpleNamespace

    primary = KeyboardInterrupt('constructor cancelled') if fault == 'cancel' else RuntimeError('admission failed')
    cleanup = OSError('close failed after releasing owned reference')
    ambient = LookupError('caller is already handling an unrelated failure')
    sessions, close_calls, eval_calls = [], [], []

    class Config:
        @property
        def vocab_size(self):
            if fault == 'vocab':
                raise primary
            return 16

        @property
        def max_position_embeddings(self):
            if fault == 'max_positions':
                raise primary
            return 32

    class Native:
        def eval(self):
            eval_calls.append(True)
            if fault in ('eval', 'cancel', 'cleanup'):
                raise primary
            return self

        @property
        def config(self):
            if fault == 'config':
                raise primary
            return Config()

    native = Native()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: native)))

    class Session(kv_cache.HFCausalLM):
        def __init__(self, **kwargs):
            sessions.append(self)
            super().__init__(**kwargs)

        def close(self):
            close_calls.append(True)
            super().close()
            if fault == 'cleanup':
                raise cleanup

    options = {'name_or_path': 'local-control-stub'} if owned else {'module': native}
    if fault in ('ok', 'ambient_ok'):
        if fault == 'ambient_ok':
            try:
                raise ambient
            except LookupError:
                session = Session(**options)
        else:
            session = Session(**options)
        assert session is sessions[0] and not session.closed and session.hf is native
        assert session.vocab_size == 16 and session.max_positions == 32 and close_calls == []
    else:
        with pytest.raises(type(primary)) as caught:
            Session(**options)
        assert caught.value is primary
        assert close_calls == [True] and sessions[0].closed is True
        assert sessions[0].hf is (None if owned else native)
        assert getattr(primary, 'cleanup_errors', ()) == ((cleanup,) if fault == 'cleanup' else ())
    assert len(sessions) == 1 and eval_calls == [True]
    assert sessions[0]._owns_model is owned
    assert not hasattr(ambient, 'cleanup_errors')


@pytest.mark.parametrize('failure', ['type_then_success', 'type_then_type', 'type_then_runtime',
                                    'unsupported_dtype', 'value', 'cancel'])
def test_pretrained_loader_propagates_first_failure_without_retry(monkeypatch, failure):
    import sys
    from types import SimpleNamespace
    import torch

    primary = (KeyboardInterrupt('cancelled loader') if failure == 'cancel' else
               ValueError('invalid local model') if failure == 'value' else
               TypeError("unexpected keyword argument 'dtype'") if failure == 'unsupported_dtype' else
               TypeError('internal model configuration failure'))
    secondary = TypeError('second load type error') if failure == 'type_then_type' else RuntimeError('second load failed')
    calls = []
    native = SimpleNamespace(eval=lambda: None, config=SimpleNamespace(vocab_size=16))

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise primary
        if failure in ('type_then_success', 'unsupported_dtype'):
            return native
        raise secondary

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load)))
    with pytest.raises(type(primary)) as caught:
        kv_cache.HFCausalLM('local-model', torch_dtype='float32', cache_dir='local-cache')
    assert caught.value is primary
    assert calls == [(('local-model',), {'dtype': torch.float32, 'local_files_only': True,
                     'attn_implementation': 'eager', 'cache_dir': 'local-cache'})]


@pytest.mark.parametrize('model_type', [[], {}], ids=['list_model_type', 'mapping_model_type'])
def test_invalid_local_hf_configuration_is_read_once_and_preserves_its_error(tmp_path, monkeypatch, model_type):
    import json
    from transformers import AutoConfig

    (tmp_path / 'config.json').write_text(json.dumps({'model_type': model_type}), encoding='utf-8')
    original = AutoConfig.from_pretrained
    errors, calls = [], []

    def read_config(*args, **kwargs):
        calls.append(args)
        try:
            return original(*args, **kwargs)
        except TypeError as error:
            errors.append(error)
            raise

    monkeypatch.setattr(AutoConfig, 'from_pretrained', read_config)
    with pytest.raises(TypeError) as caught:
        kv_cache.HFCausalLM(str(tmp_path), local_files_only=True)
    assert len(calls) == len(errors) == 1
    assert caught.value is errors[0]
