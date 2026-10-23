"""Injected scientific session: committed facts, cancellation and failures."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from spec_mtp.audit import audit_run
from spec_mtp.contracts import DecodeRequest, Prompt
from spec_mtp.decode import DecodeFailure, ar_generate, generate, select_token, tree_probs
from spec_mtp.masks import ESPMaskProvider


class FakeSession:
    """Enumerable +1 token model; cache storage is opaque to the decoder."""
    def __init__(self):
        self.embedding_table = torch.eye(8)
        self.device, self.dtype = torch.device('cpu'), torch.float32
        self.vocab_size, self.max_positions, self.tied_embeddings = 8, 128, True
        self.calls = 0
        self.kept = []

    def embed(self, ids):
        return self.embedding_table[ids]

    def min_mask_value(self):
        return torch.finfo(self.dtype).min

    def new_cache(self):
        return SimpleNamespace(length=0)

    def cache_len(self, cache):
        return cache.length

    def crop_cache(self, cache, length):
        cache.length = length

    def keep_cache(self, cache, prefix, indices):
        self.kept.append((prefix, list(indices)))
        cache.length = prefix + len(indices)

    def __call__(self, embeds, positions, *, past_key_values, **_):
        self.calls += 1
        tokens = (embeds.argmax(-1) + 1) % 8
        logits = torch.nn.functional.one_hot(tokens, 8).float() * 20
        past_key_values.length += embeds.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class MemorySink:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(dict(row))


def request(**kwargs):
    return DecodeRequest(Prompt('p0', (1, 2, 3)),
                         {'num_masks': 1, 'method': 'esp', 'impl': 'efficient',
                          'tree': {'mode': 'static', 'branches': [3], 'pruning': True}},
                         max_new_tokens=7, sources={'science': 'frozen'}, **kwargs)


def run(req, model=None, sink=None, cancel=None):
    model, sink = model or FakeSession(), sink or MemorySink()
    result = generate(req, model, ESPMaskProvider(1, model.embedding_table), sink, cancel)
    return result, model, sink


@pytest.mark.parametrize('budget', [0, 1, 2, 7])
def test_budget_and_native_ar_match_per_token_and_audit(budget):
    req = replace(request(), max_new_tokens=budget)
    result, model, sink = run(req)
    native, stats = ar_generate(FakeSession(), torch.tensor([req.prompt.input_ids]),
                               {'max_new_tokens': budget, 'temperature': 0})
    assert result.committed == tuple(native[0, 3:].tolist())
    assert result.evidence_complete
    audit = audit_run(sink.rows, req.sources)
    assert audit.source_integrity and audit.argmax_coverage and audit.argmax_correctness, audit.reasons
    if budget == 0:
        assert model.calls == 0 and stats['ar_calls'] == 0


def test_one_shot_cancel_signal_is_latched_and_never_reclassified():
    signal = iter([True, False])
    result, model, _ = run(request(), cancel=lambda: next(signal))
    assert result.execution == 'cancelled' and result.termination_reason == 'cancelled'
    assert result.committed == () and model.calls == 0
    assert next(signal) is False  # exactly one cancellation observation consumed


def test_sink_first_commit_failure_preserves_committed_fact_and_stops_next_forward():
    model = FakeSession()
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


@pytest.mark.parametrize('tokens', [torch.empty((1, 0), dtype=torch.long), torch.ones((2, 2), dtype=torch.long),
                                    torch.ones((1, 2)), torch.tensor([[-1]]), torch.tensor([[8]])])
def test_invalid_prompt_is_rejected_before_forward(tokens):
    from spec_mtp.decode import SpecDecoder
    model = FakeSession()
    decoder = SpecDecoder(model, request().spec, {'max_new_tokens': 3},
                          mask_policy=ESPMaskProvider(1, model.embedding_table))
    with pytest.raises(ValueError):
        decoder.generate(tokens)
    assert model.calls == 0


def test_eos_prefill_is_in_complete_argmax_audit():
    req = replace(request(), eos_ids=(4,))
    result, model, sink = run(req)
    assert result.committed == (4,) and model.calls == 1
    assert audit_run(sink.rows, req.sources).argmax_coverage


def test_sink_failure_before_first_forward_is_zero_work():
    class FailingSink:
        def write(self, row):
            raise OSError('header failed')
    model = FakeSession()
    with pytest.raises(DecodeFailure) as failure:
        run(request(), model=model, sink=FailingSink())
    assert failure.value.result.committed == () and model.calls == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
@pytest.mark.parametrize('temperature', [0., 1.])
def test_invalid_logits_rejected_in_token_and_tree_selection(value, temperature):
    logits = torch.full((8,), value)
    with pytest.raises(ValueError, match='logits'):
        select_token(logits, temperature, None)
    with pytest.raises(ValueError, match='logits'):
        tree_probs(logits.unsqueeze(0), temperature)


@pytest.mark.parametrize('temperature', [0., 1.])
def test_negative_infinity_bans_tokens_without_invalidating_distribution(temperature):
    logits = torch.tensor([float('-inf'), 2.])
    assert select_token(logits, temperature, torch.Generator().manual_seed(0)) == 1
    assert torch.equal(tree_probs(logits.unsqueeze(0), temperature), torch.tensor([[0., 1.]]))


@pytest.mark.parametrize('values', [[2., 3.], [-1., 0.]])
def test_temperature_scaling_overflow_rejected_before_sampling_or_tree(values):
    logits = torch.tensor(values)
    with pytest.raises(ValueError, match='logits'):
        select_token(logits, 1e-40, None)
    with pytest.raises(ValueError, match='logits'):
        tree_probs(logits.unsqueeze(0), 1e-40)


@pytest.mark.parametrize('failure_at', ['prefill', 'decode', 'mask'])
def test_invalid_model_logits_stop_without_losing_prior_commits(failure_at):
    class InvalidSession(FakeSession):
        def __call__(self, *args, **kwargs):
            out = super().__call__(*args, **kwargs)
            if failure_at == 'prefill' or (failure_at == 'decode' and self.calls == 2):
                out.logits[:] = float('nan')
            elif failure_at == 'mask':
                out.logits[:, -1] = float('nan')
            return out
    model, sink = InvalidSession(), MemorySink()
    with pytest.raises(DecodeFailure) as failure:
        run(request(), model=model, sink=sink)
    expected = () if failure_at == 'prefill' else (4,)
    assert failure.value.result.committed == expected
    assert tuple(row['token'] for row in sink.rows if row['event'] == 'commit') == expected
    assert not failure.value.result.evidence_complete
    assert model.calls == (2 if failure_at == 'decode' else 1)
    assert isinstance(failure.value.__cause__, ValueError)


@pytest.mark.parametrize('budget', [0, 1])
def test_ar_prefill_rejects_context_overflow_before_cache_or_embedding(budget):
    model = FakeSession()
    model.max_positions = 2
    def forbidden(*args):
        pytest.fail('context overflow reached cache or embedding')
    model.embed = model.new_cache = forbidden
    prompt = torch.tensor([[1, 2, 3]])
    if budget:
        with pytest.raises(ValueError, match='context'):
            ar_generate(model, prompt, {'max_new_tokens':budget})
    else:
        output, stats = ar_generate(model, prompt, {'max_new_tokens':budget})
        assert torch.equal(output, prompt) and stats['ar_calls'] == 0
    assert model.calls == 0


def test_ar_full_context_can_emit_prefill_token_without_next_forward():
    model = FakeSession()
    model.max_positions = 3
    output, stats = ar_generate(model, torch.tensor([[1, 2, 3]]), {'max_new_tokens':2})
    assert output.tolist() == [[1, 2, 3, 4]] and stats['ar_calls'] == model.calls == 1
