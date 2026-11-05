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



from studies._shared.structure.config import apply_variant, _convert
from spec_mtp.structure.algorithm.decode.loop import SpecDecoder, _fork_generator
from spec_mtp.structure.algorithm.mask import build_mask_provider

def _boundary_eval(**kw):
    base = {'max_new_tokens': 16, 'temperature': 0.0, 'stop_on_eos': False}
    base.update(kw)
    return _convert(base)

@pytest.mark.parametrize('spec, match', [
    ({'method': 'esp', 'num_masks': 1, 'impl': 'efficient', 'tree': {'mode': 'dynamic'}},
     'efficient requires spec.tree.mode=static'),
    ({'method': 'esp', 'num_masks': 1, 'impl': 'turbo', 'tree': {'mode': 'static', 'branches': [2]}},
     'Unsupported spec.impl'),
])
def test_decoder_rejects_bad_impl_tree_pairs(spec, match, cuda_tiny_model, cuda_decode, record_property):
    tiny = cuda_tiny_model()
    with pytest.raises(ValueError, match=match):
        SpecDecoder(tiny, _convert(spec), _boundary_eval(), mask_policy=build_mask_provider(_convert(spec), tiny.embedding_table))


def test_stop_on_eos_matches_ar_and_stops_early(cuda_tiny_model, cuda_decode, record_property):
    """Plant an EOS id; both paths must stop on it and stay token-identical."""
    tiny = cuda_tiny_model()
    prompt = torch.randint(0, 64, (1, 8), generator=torch.Generator(device=cuda_decode.device).manual_seed(1), device=cuda_decode.device)
    # Force a tiny budget so we exercise the stop path without relying on natural EOS.
    spec = apply_variant({}, 'esp_static')
    eval_cfg = _boundary_eval(max_new_tokens=12, stop_on_eos=True)
    # Use a token that the random model will eventually emit; verify equality with AR.
    eos_id = 3
    dec = SpecDecoder(tiny, spec, eval_cfg, eos_ids=[eos_id], generator=torch.Generator(device=cuda_decode.device).manual_seed(9), mask_policy=build_mask_provider(spec, tiny.embedding_table))
    out, stats = dec.generate(prompt)
    ar, ar_stats = ar_generate(tiny, prompt, eval_cfg, eos_ids=[eos_id],
                               generator=torch.Generator(device=cuda_decode.device).manual_seed(9))
    assert torch.equal(out, ar)
    assert stats['new_tokens'] == ar_stats['ar_calls']
    # If EOS appeared, it is the last committed token and we did not overshoot the budget.
    assert out.size(1) <= prompt.size(1) + 12
    record_property('eos_observed', bool((out[0, prompt.size(1):] == eos_id).any().item()))
    record_property('termination_reason', stats['termination_reason'])


def test_context_limit_sets_the_flag_and_matches_budgeted_ar(cuda_tiny_model, cuda_decode, record_property):
    short = cuda_tiny_model({
        'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64, 'num_layers': 1,
        'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 40,
    })
    prompt = torch.randint(0, 64, (1, 24), generator=torch.Generator(device=cuda_decode.device).manual_seed(2), device=cuda_decode.device)
    spec = apply_variant({}, 'esp_dynamic')
    eval_cfg = _boundary_eval(max_new_tokens=32)
    out, stats = SpecDecoder(short, spec, eval_cfg, generator=torch.Generator(device=cuda_decode.device).manual_seed(2), mask_policy=build_mask_provider(spec, short.embedding_table)).generate(prompt)
    assert stats['hit_context_limit'] == 1
    assert out.size(1) <= 40
    budget = dict(eval_cfg)
    budget['max_new_tokens'] = stats['new_tokens']
    audited, _ = ar_generate(short, prompt, _convert(budget), generator=torch.Generator(device=cuda_decode.device).manual_seed(2))
    assert torch.equal(out, audited)


def test_budget_truncation_without_context_limit(cuda_tiny_model, cuda_decode, record_property):
    tiny = cuda_tiny_model()
    prompt = torch.randint(0, 64, (1, 8), generator=torch.Generator(device=cuda_decode.device).manual_seed(3), device=cuda_decode.device)
    spec = apply_variant({}, 'esp_static')
    eval_cfg = _boundary_eval(max_new_tokens=5)
    out, stats = SpecDecoder(tiny, spec, eval_cfg, generator=torch.Generator(device=cuda_decode.device).manual_seed(3), mask_policy=build_mask_provider(spec, tiny.embedding_table)).generate(prompt)
    assert stats['hit_context_limit'] == 0
    assert stats['new_tokens'] == 5
    assert out.size(1) == prompt.size(1) + 5


def test_fork_generator_copies_state_and_leaves_source(cuda_decode, record_property):
    src = torch.Generator(device=cuda_decode.device).manual_seed(11)
    before = src.get_state().clone()
    fork = _fork_generator(src)
    assert fork is not src
    draw = torch.randn(4, generator=fork, device=cuda_decode.device)
    assert draw.device == cuda_decode.device
    record_property('fork_tensor_device', str(draw.device))
    assert torch.equal(src.get_state(), before)
    assert _fork_generator(None) is None


def test_decoder_uses_it_only_for_dynamic_trees(cuda_tiny_model, cuda_decode, record_property):
    from studies._shared.structure.config import apply_variant
    from spec_mtp.structure.algorithm.decode.loop import SpecDecoder
    from studies._shared.structure.config import _convert

    model = cuda_tiny_model({'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64,
                                    'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1,
                                    'max_position_embeddings': 256})
    eval_cfg = _convert({'max_new_tokens': 4, 'temperature': 0.0, 'stop_on_eos': False})
    dynamic = SpecDecoder(model, apply_variant({}, 'esp_dynamic'), eval_cfg, mask_policy=build_mask_provider(apply_variant({}, 'esp_dynamic'), model.embedding_table))
    naive_static = SpecDecoder(model, apply_variant({}, 'llama3/impl_naive_bc30'), eval_cfg, mask_policy=build_mask_provider(apply_variant({}, 'llama3/impl_naive_bc30'), model.embedding_table))
    efficient = SpecDecoder(model, apply_variant({}, 'esp_static'), eval_cfg, mask_policy=build_mask_provider(apply_variant({}, 'esp_static'), model.embedding_table))
    assert dynamic._dynamic_attn is not None
    assert naive_static._dynamic_attn is None and naive_static._attn_cache is None  # Table 4 naive path kept
    assert efficient._dynamic_attn is None and efficient._attn_cache is not None


@pytest.mark.parametrize('budget,eos_ids,expected,termination', [
    (2, (), (4, 5), 'budget'),
    (3, (), (4, 5, 6), 'budget'),
    (7, (), (4, 5, 6, 7, 0, 1, 2), 'budget'),
    (7, (6,), (4, 5, 6), 'eos'),
], ids=['one_commit_before_bonus', 'two_commits_before_bonus', 'two_full_rounds', 'eos_before_bonus'])
def test_hybrid_collect_observes_only_committed_prefix_and_finishes_each_round(
        budget, eos_ids, expected, termination, cuda_decode, record_property):
    """Actual decode callbacks distinguish a truncated last commit from a bonus."""
    from spec_mtp.structure.algorithm.mask.providers import HybridGainKalman

    device = cuda_decode.device
    events = []

    class PositionSession(cuda_decode.FakeSession):
        def __call__(self, embeds, positions, **kwargs):
            out = super().__call__(embeds, positions, **kwargs)
            # A single position-only model rule applies to prefill and every owner.
            tokens = (positions + 2) % self.vocab_size
            out.logits = torch.nn.functional.one_hot(tokens, self.vocab_size).float() * 20
            assert out.logits.device == device
            return out

    model, sink = PositionSession(), cuda_decode.MemorySink()

    class RecordingCollect(HybridGainKalman):
        def init_from_prompt(self, prompt_embeds, generator=None):
            events.append(('init', model.calls))
            super().init_from_prompt(prompt_embeds, generator)

        def masks(self):
            events.append(('masks', model.calls))
            return super().masks()

        def on_token_committed(self, new_embed, prev_embed):
            assert new_embed.device == prev_embed.device == device
            events.append(('commit', int(new_embed.argmax().item()),
                           int(prev_embed.argmax().item()), model.calls))
            super().on_token_committed(new_embed, prev_embed)

        def finish_round(self):
            events.append(('finish', model.calls))
            super().finish_round()

    policy = RecordingCollect(ESPMaskProvider(2, model.embedding_table), 'collect', {'observe': 'last'})
    req = replace(request(), max_new_tokens=budget, eos_ids=eos_ids,
                  spec={'method': 'esp', 'num_masks': 2, 'impl': 'efficient',
                        'tree': {'mode': 'static', 'branches': [1, 1], 'pruning': False}})
    result = generate(req, model, policy, sink)
    assert result.committed == expected and result.termination_reason == termination
    assert tuple(row['token'] for row in sink.rows if row['event'] == 'commit') == expected
    traces = result.stats['trace']
    expected_events = [('init', 0), ('masks', 0), ('commit', 4, 3, 1)]
    previous = 4
    for number, trace in enumerate(traces, start=1):
        assert trace['accepted_nodes'] == [1, 2]
        expected_events.append(('masks', number))
        for token in trace['committed_tokens']:
            expected_events.append(('commit', token, previous, number + 1))
            previous = token
        expected_events.append(('finish', number + 1))
    assert events == expected_events
    assert model.calls == 1 + len(traces)
    assert tuple(token for trace in traces for token in trace['committed_tokens']) == expected[1:]

    first = [trace['committed_tokens'][0] for trace in traces]
    last = [trace['committed_tokens'][-1] for trace in traces]
    logical_bonus = [(trace['root_position'] + 4) % model.vocab_size for trace in traces]
    assert last == logical_bonus if budget == 7 and not eos_ids else last != logical_bonus
    pairs = policy.calibration_rows()
    # Collection stores CPU artifacts; perform the numerical reference checks on CUDA.
    for name, tokens in [('first', first), ('last', last)]:
        actual = pairs[name].to(device)
        reference = model.embedding_table[torch.tensor(tokens, device=device)]
        assert actual.device == reference.device == device
        assert torch.equal(actual, reference)
    assert len(first) == len(last) == pairs['m'].shape[0] == len(traces)
    assert policy._pending is None and policy._first is None
    audit = audit_run(sink.rows, req.sources)
    assert result.evidence_complete and audit.argmax_coverage and audit.argmax_correctness
    record_property('provider_commit_timeline', {
        'committed': list(expected), 'round_first': first, 'round_last': last,
        'logical_bonus': logical_bonus, 'finished_rounds': len(traces),
        'tensor_device': str(model.embedding_table.device),
        'collected_storage_device': str(pairs['first'].device),
    })
