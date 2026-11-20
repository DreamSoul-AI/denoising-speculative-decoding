"""loop.py retained-token state timing through the real decoder, without weights."""

from types import SimpleNamespace

import pytest
import torch

from spec_mtp.audit import audit_run
from spec_mtp.model import HFCausalLM
from spec_mtp.structure.algorithm.decode.loop import DecodeFailure, SpecDecoder, ar_generate, generate
from spec_mtp.structure.algorithm.mask import InputTrajectoryMaskProvider, build_mask_provider
from spec_mtp.structure.api.decode_api import DecodeRequest, Prompt


class PositionSession:
    """Absolute position selects the next token; masks seed a perfect depth-2 path."""

    def __init__(self, reject=False):
        self.embedding_table = torch.eye(32)
        self.device, self.dtype = torch.device('cpu'), torch.float32
        self.vocab_size, self.max_positions, self.tied_embeddings = 32, 64, True
        self.calls, self.reject = 0, reject
        self.blocks = []

    def embed(self, ids):
        return self.embedding_table[ids]

    def min_mask_value(self):
        return torch.finfo(self.dtype).min

    def new_cache(self):
        return SimpleNamespace(length=0)

    def cache_len(self, cache):
        return cache.length

    def crop_cache(self, cache, keep_len):
        cache.length = keep_len

    def keep_cache(self, cache, prefix_len, indices):
        cache.length = prefix_len + len(indices)

    def __call__(self, embeds, positions, *, past_key_values, **_):
        self.calls += 1
        self.blocks.append(embeds.clone())
        targets = (positions + 1) % self.vocab_size
        logits = torch.nn.functional.one_hot(targets, self.vocab_size).float() * 20
        if self.reject:
            # The verification root uses its genuine position, while all
            # prospective mask rows predict 20. No draft ever matches.
            logits[:, 1:] = 0
            logits[:, 1:, 20] = 20
            if self.calls == 1:
                logits[:, 2] = 0
                logits[:, 2, 3] = 20
        past_key_values.length += embeds.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class Sink:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


def spec(kind='holt', mode='static'):
    return {'method': 'input_trajectory', 'num_masks': 2,
            'trajectory': {'kind': kind, 'alpha': .2, 'beta': .8},
            'impl': 'efficient' if mode == 'static' else 'naive',
            'block_complexity': 12,
            'tree': {'mode': mode, 'branches': [2, 1], 'pruning': False}}


@pytest.mark.parametrize('kind', ['order0', 'difference', 'holt', 'kalman', 'fixed_gain'])
@pytest.mark.parametrize('budget,eos,expected_A,expected_C', [
    (0, (), 0, 0), (1, (), 0, 0), (2, (), 1, 1), (4, (), 2, 1),
    (5, (), 3, 2), (12, (4,), 1, 1), (12, (5,), 2, 1), (12, (3,), 0, 0),
])
def test_state_only_absorbs_actual_output_after_budget_or_eos(kind, budget, eos, expected_A, expected_C):
    model, sink = PositionSession(), Sink()
    config = spec(kind)
    provider = build_mask_provider(config, model.embedding_table)
    request = DecodeRequest(Prompt('positions', (0, 1, 2)), config,
                            max_new_tokens=budget, eos_ids=eos)
    result = generate(request, model, provider, sink)
    stats, tokens = result.stats, result.committed
    assert stats['retained_accepted'] == expected_A
    assert stats['decode_calls'] == expected_C
    S = max(0, len(tokens) - 1)
    assert stats['acceptance_metrics'] == {
        'protocol': 'retained-acceptance-v1', 'A': expected_A, 'C': expected_C, 'S': S,
        'accepted_per_call': expected_A / expected_C if expected_C else None,
        'tau': S / expected_C if expected_C else None}
    assert sum(len(t['retained_accepted_nodes']) for t in stats['trace']) == expected_A
    assert all(t['retained_accepted_nodes'] == t['accepted_nodes'][:len(t['committed_tokens'])]
               for t in stats['trace'])
    assert stats['accepted'] >= expected_A
    if budget == 2 or eos == (4,):
        assert stats['accepted'] == 2 and expected_A == 1
    if budget:
        replay = InputTrajectoryMaskProvider(2, kind=kind, alpha=.2, beta=.8)
        replay.init_from_prompt(model.embedding_table[:3])
        for token in tokens:
            replay.on_token_committed(model.embedding_table[token], torch.zeros(32))
        assert provider.committed_steps == len(tokens)
        torch.testing.assert_close(provider.M, replay.M)
        torch.testing.assert_close(provider.V, replay.V)
        if provider.covariance is not None:
            torch.testing.assert_close(provider.covariance, replay.covariance)
        assert stats['verification_argmax_audit']['checked_committed_tokens'] == len(tokens)
    audit = audit_run(sink.rows, request.sources)
    assert audit.source_integrity and audit.argmax_coverage and audit.argmax_correctness, audit.reasons


def test_rejected_tree_never_enters_state_and_next_block_uses_updated_horizon():
    model = PositionSession(reject=True)
    provider = build_mask_provider(spec(), model.embedding_table)
    output, stats = SpecDecoder(model, spec(), {'max_new_tokens': 5}, mask_policy=provider).generate(
        torch.tensor([[0, 1, 2]]))
    assert output[0, 3:].tolist() == [3, 4, 5, 6, 7]
    assert stats['retained_accepted'] == stats['accepted'] == 0
    assert stats['acceptance_metrics']['tau'] == 1
    assert provider.committed_steps == 5
    replay = InputTrajectoryMaskProvider(2, kind='holt', alpha=.2, beta=.8)
    replay.init_from_prompt(model.embedding_table[:3])
    replay.on_token_committed(model.embedding_table[3], model.embedding_table[2])
    # Four owners, k=2; first depth-1 owner's first slot is at horizon 2.
    torch.testing.assert_close(model.blocks[1][0, 6], replay.M + 2 * replay.V)
    for token in (4, 5, 6, 7):
        replay.on_token_committed(model.embedding_table[token], torch.zeros(32))
    torch.testing.assert_close(provider.M, replay.M)
    torch.testing.assert_close(provider.V, replay.V)


def test_partial_failure_reports_only_already_committed_accepts():
    class FailingSession(PositionSession):
        after_forward = 0

        def embed(self, ids):
            if self.calls == 2:
                self.after_forward += 1
                if self.after_forward == 2:
                    raise RuntimeError('second commit failed')
            return super().embed(ids)

    model, sink = FailingSession(), Sink()
    provider = build_mask_provider(spec(), model.embedding_table)
    request = DecodeRequest(Prompt('partial', (0, 1, 2)), spec(), max_new_tokens=10)
    with pytest.raises(DecodeFailure) as failure:
        generate(request, model, provider, sink)
    result = failure.value.result
    assert result.committed == (3, 4) and provider.committed_steps == 2
    assert result.stats['accepted'] == 2
    assert result.stats['acceptance_metrics']['A'] == result.stats['retained_accepted'] == 1
    assert result.stats['acceptance_metrics']['S'] == 1
    assert not result.evidence_complete


@pytest.mark.parametrize('kind', ['order0', 'difference', 'holt', 'kalman', 'fixed_gain'])
@pytest.mark.parametrize('mode', ['static', 'dynamic'])
def test_tiny_transformer_committed_argmax_and_cpu_ar_diagnostic(kind, mode):
    torch.manual_seed(12)
    model = HFCausalLM(tiny_config={'vocab_size': 31, 'hidden_size': 16, 'intermediate_size': 32,
                                   'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1,
                                   'max_position_embeddings': 64})
    config = spec(kind, mode)
    provider = build_mask_provider(config, model.embedding_table)
    prompt = torch.tensor([[1, 2, 3]])
    evaluation = {'max_new_tokens': 9, 'stop_on_eos': False}
    output, stats = SpecDecoder(model, config, evaluation, mask_policy=provider).generate(prompt)
    reference, _ = ar_generate(model, prompt, evaluation)
    assert stats['verification_argmax_audit']['passed']
    assert stats['verification_argmax_audit']['checked_committed_tokens'] == 9
    assert torch.equal(output, reference)  # CPU diagnostic, not a BF16 guarantee.
    assert provider.committed_steps == 9
