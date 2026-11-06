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
