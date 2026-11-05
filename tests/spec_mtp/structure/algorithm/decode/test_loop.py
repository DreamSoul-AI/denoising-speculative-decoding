"""Injected scientific session: committed facts, cancellation and failures."""
from dataclasses import replace

import pytest
import torch

from spec_mtp.structure.artifact.observation import audit_run
from spec_mtp.structure.algorithm.decode.loop import DecodeFailure, ar_generate, generate, select_token, tree_probs
from spec_mtp.structure.algorithm.mask import ESPMaskProvider


from tests.conftest import decode_request as request


@pytest.mark.parametrize('budget', [0, 1, 2, 7])
def test_budget_and_native_ar_match_per_token_and_audit(budget, cuda_decode, record_property):
    FakeSession = cuda_decode.FakeSession
    run = cuda_decode.run
    req = replace(request(), max_new_tokens=budget)
    result, model, sink = run(req)
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    native, stats = ar_generate(FakeSession(), torch.tensor([req.prompt.input_ids], device=cuda_decode.device),
                               {'max_new_tokens': budget, 'temperature': 0})
    assert result.committed == tuple(native[0, 3:].tolist())
    assert result.evidence_complete
    audit = audit_run(sink.rows, req.sources)
    assert audit.source_integrity and audit.argmax_coverage and audit.argmax_correctness, audit.reasons
    if budget == 0:
        assert model.calls == 0 and stats['ar_calls'] == 0


def test_one_shot_cancel_signal_is_latched_and_never_reclassified(cuda_decode, record_property):
    run = cuda_decode.run
    signal = iter([True, False])
    result, model, _ = run(request(), cancel=lambda: next(signal))
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    assert result.execution == 'cancelled' and result.termination_reason == 'cancelled'
    assert result.committed == () and model.calls == 0
    assert next(signal) is False  # exactly one cancellation observation consumed


def test_sink_first_commit_failure_preserves_committed_fact_and_stops_next_forward(cuda_decode, record_property):
    FakeSession = cuda_decode.FakeSession
    MemorySink = cuda_decode.MemorySink
    run = cuda_decode.run
    model = FakeSession()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    class FailingSink(MemorySink):
        def write(self, row):
            if row['event'] == 'commit':
                raise OSError('disk full')
            super().write(row)
    with pytest.raises(DecodeFailure) as failure:
        run(request(), model=model, sink=FailingSink())
    assert model.calls == 1
    assert failure.value.result.committed == (4,)
    assert not failure.value.result.evidence_complete
    assert isinstance(failure.value.__cause__, OSError)


@pytest.mark.parametrize('tokens', [
    lambda device: torch.empty((1, 0), dtype=torch.long, device=device),
    lambda device: torch.ones((2, 2), dtype=torch.long, device=device),
    lambda device: torch.ones((1, 2), device=device),
    lambda device: torch.tensor([[-1]], device=device),
    lambda device: torch.tensor([[8]], device=device),
], ids=['tokens0', 'tokens1', 'tokens2', 'tokens3', 'tokens4'])
def test_invalid_prompt_is_rejected_before_forward(tokens, cuda_decode, record_property):
    FakeSession = cuda_decode.FakeSession
    tokens = tokens(cuda_decode.device)
    assert tokens.device == cuda_decode.device
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
    model = FakeSession()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    decoder = SpecDecoder(model, request().spec, {'max_new_tokens': 3},
                          mask_policy=ESPMaskProvider(1, model.embedding_table))
    with pytest.raises(ValueError):
        decoder.generate(tokens)
    assert model.calls == 0


def test_eos_prefill_is_in_complete_argmax_audit(cuda_decode, record_property):
    run = cuda_decode.run
    req = replace(request(), eos_ids=(4,))
    result, model, sink = run(req)
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    assert result.committed == (4,) and model.calls == 1
    assert audit_run(sink.rows, req.sources).argmax_coverage


def test_sink_failure_before_first_forward_is_zero_work(cuda_decode, record_property):
    FakeSession = cuda_decode.FakeSession
    run = cuda_decode.run
    class FailingSink:
        def write(self, row):
            raise OSError('header failed')
    model = FakeSession()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    with pytest.raises(DecodeFailure) as failure:
        run(request(), model=model, sink=FailingSink())
    assert failure.value.result.committed == () and model.calls == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
@pytest.mark.parametrize('temperature', [0., 1.])
def test_invalid_logits_rejected_in_token_and_tree_selection(value, temperature, cuda_decode, record_property):
    logits = torch.full((8,), value, device=cuda_decode.device)
    assert logits.device == cuda_decode.device
    record_property('selection_tensor_device', str(logits.device))
    with pytest.raises(ValueError, match='logits'):
        select_token(logits, temperature, None)
    with pytest.raises(ValueError, match='logits'):
        tree_probs(logits.unsqueeze(0), temperature)


@pytest.mark.parametrize('temperature', [0., 1.])
def test_negative_infinity_bans_tokens_without_invalidating_distribution(temperature, cuda_decode, record_property):
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(cuda_decode.device).clone()
    default_device = torch.get_default_device()
    logits = torch.tensor([float('-inf'), 2.], device=cuda_decode.device)
    assert logits.device == cuda_decode.device
    record_property('selection_tensor_device', str(logits.device))
    assert select_token(logits, temperature, torch.Generator(device=cuda_decode.device).manual_seed(0)) == 1
    assert torch.equal(tree_probs(logits.unsqueeze(0), temperature), torch.tensor([[0., 1.]], device=cuda_decode.device))
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(cuda_decode.device), cuda_rng)
    assert torch.get_default_device() == default_device
    record_property('rng_default_device_preserved', True)


@pytest.mark.parametrize('values', [[2., 3.], [-1., 0.]])
def test_temperature_scaling_overflow_rejected_before_sampling_or_tree(values, cuda_decode, record_property):
    logits = torch.tensor(values, device=cuda_decode.device)
    assert logits.device == cuda_decode.device
    record_property('selection_tensor_device', str(logits.device))
    with pytest.raises(ValueError, match='logits'):
        select_token(logits, 1e-40, None)
    with pytest.raises(ValueError, match='logits'):
        tree_probs(logits.unsqueeze(0), 1e-40)


@pytest.mark.parametrize('failure_at', ['prefill', 'decode', 'mask'])
def test_invalid_model_logits_stop_without_losing_prior_commits(failure_at, cuda_decode, record_property):
    FakeSession = cuda_decode.FakeSession
    MemorySink = cuda_decode.MemorySink
    run = cuda_decode.run
    class InvalidSession(FakeSession):
        def __call__(self, *args, **kwargs):
            out = super().__call__(*args, **kwargs)
            if failure_at == 'prefill' or (failure_at == 'decode' and self.calls == 2):
                out.logits[:] = float('nan')
            elif failure_at == 'mask':
                out.logits[:, -1] = float('nan')
            return out
    model, sink = InvalidSession(), MemorySink()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    with pytest.raises(DecodeFailure) as failure:
        run(request(), model=model, sink=sink)
    expected = () if failure_at == 'prefill' else (4,)
    assert failure.value.result.committed == expected
    assert tuple(row['token'] for row in sink.rows if row['event'] == 'commit') == expected
    assert not failure.value.result.evidence_complete
    assert model.calls == (2 if failure_at == 'decode' else 1)
    assert isinstance(failure.value.__cause__, ValueError)


@pytest.mark.parametrize('budget', [0, 1])
def test_ar_prefill_rejects_context_overflow_before_cache_or_embedding(budget, cuda_decode, record_property):
    FakeSession = cuda_decode.FakeSession
    model = FakeSession()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    model.max_positions = 2
    def forbidden(*args):
        pytest.fail('context overflow reached cache or embedding')
    model.embed = model.new_cache = forbidden
    prompt = torch.tensor([[1, 2, 3]], device=cuda_decode.device)
    if budget:
        with pytest.raises(ValueError, match='context'):
            ar_generate(model, prompt, {'max_new_tokens':budget})
    else:
        output, stats = ar_generate(model, prompt, {'max_new_tokens':budget})
        assert torch.equal(output, prompt) and stats['ar_calls'] == 0
    assert model.calls == 0


def test_ar_full_context_can_emit_prefill_token_without_next_forward(cuda_decode, record_property):
    FakeSession = cuda_decode.FakeSession
    model = FakeSession()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    model.max_positions = 3
    output, stats = ar_generate(model, torch.tensor([[1, 2, 3]], device=cuda_decode.device), {'max_new_tokens':2})
    assert output.tolist() == [[1, 2, 3, 4]] and stats['ar_calls'] == model.calls == 1


@pytest.mark.parametrize('fault', ['embed', 'mask', 'sink', 'sink_first'])
def test_later_token_failure_preserves_successful_commits_within_block(fault, cuda_decode, record_property):
    MemorySink = cuda_decode.MemorySink
    ConstantSession = cuda_decode.ConstantSession
    class FailingSession(ConstantSession):
        after_forward_embeds = 0
        def embed(self, ids):
            if self.calls == 2:
                self.after_forward_embeds += 1
                if fault == 'embed' and self.after_forward_embeds == 2:
                    raise ValueError('second decode embedding failed')
            return super().embed(ids)
    class RecordingMask(ESPMaskProvider):
        committed = 0
        def on_token_committed(self, new_embed, prev_embed):
            if fault == 'mask' and self.committed == 2:
                raise ValueError('second decode mask update failed')
            super().on_token_committed(new_embed, prev_embed)
            self.committed += 1
    class FailingSink(MemorySink):
        def write(self, row):
            failed_index = 1 if fault == 'sink_first' else 2
            if fault in ('sink', 'sink_first') and row['event'] == 'commit' and row['index'] == failed_index:
                raise OSError('second decode commit write failed')
            super().write(row)
    model, sink = FailingSession(), FailingSink()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    mask = RecordingMask(1, model.embedding_table)
    req = replace(request(), spec={'num_masks':1, 'impl':'efficient',
        'tree':{'mode':'static', 'branches':[1], 'pruning':False}})
    with pytest.raises(DecodeFailure) as failure:
        generate(req, model, mask, sink)
    result = failure.value.result
    count = 3 if fault == 'sink' else 2
    assert result.committed == (4,) * count
    assert result.stats['committed'] == count - 1
    assert mask.committed == count and model.calls == 2
    assert model.after_forward_embeds == (1 if fault == 'sink_first' else 2)
    assert [r['token'] for r in sink.rows if r['event'] == 'commit'] == ([4] if fault == 'sink_first' else [4, 4])
    assert not result.evidence_complete


@pytest.mark.parametrize('stop', ['budget', 'eos'])
def test_normal_accepted_bonus_block_preserves_budget_and_eos(stop, cuda_decode, record_property):
    MemorySink = cuda_decode.MemorySink
    ConstantSession = cuda_decode.ConstantSession
    class Session(ConstantSession):
        def __call__(self, *args, **kwargs):
            out = super().__call__(*args, **kwargs)
            if stop == 'eos':
                rows = out.logits[:, -1:] if self.calls == 1 else out.logits
                rows.zero_()
                rows[..., 5] = 20
            return out
    model, sink = Session(), MemorySink()
    assert model.embedding_table.device == cuda_decode.device
    record_property('model_tensor_device', str(model.embedding_table.device))
    req = replace(request(), max_new_tokens=3 if stop == 'budget' else 7,
        eos_ids=() if stop == 'budget' else (5,), spec={'num_masks':1, 'impl':'efficient',
            'tree':{'mode':'static', 'branches':[1], 'pruning':False}})
    result = generate(req, model, ESPMaskProvider(1, model.embedding_table), sink)
    assert result.committed == ((4, 4, 4) if stop == 'budget' else (4, 5))
    assert result.termination_reason == stop and model.calls == 2
    assert audit_run(sink.rows, req.sources).argmax_coverage
