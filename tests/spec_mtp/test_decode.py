"""Injected scientific session: committed facts, cancellation and failures."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from spec_mtp.audit import audit_run
from spec_mtp.contracts import DecodeRequest, Prompt
from spec_mtp.decode import DecodeFailure, ar_generate, generate
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
