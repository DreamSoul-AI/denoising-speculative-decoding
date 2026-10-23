"""EMA history audit: commit alignment, rejection isolation, synthetic geometry.

Lossless token match against AR does not prove the velocity state is clean.
Sample matching accepts only the base model's own draws, so a polluted EMA
still reproduces the AR string while changing later masks and acceptance.

This file checks what tests/check_correctness.py leaves implicit:

  1. Position alignment. Mask i sits at position offset i and extrapolates i
     velocity steps.
  2. Synthetic trajectories. A constant embedding difference is recovered
     exactly. An alternating difference cancels in the raw EMA and keeps a
     typical step length after direction normalization.
  3. Live decoding with mixed accept/reject passes. on_token_committed sees
     exactly the committed token embeddings, in order, and the final state
     equals a fresh EMA over the committed sequence only.
  4. update=false on the live path freezes velocity and still moves the anchor.
"""

from spec_mtp.masks import build_mask_provider

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import torch

from studies._shared.structure.config import _convert
from spec_mtp.model import HFCausalLM
from spec_mtp.decode import SpecDecoder, ar_generate
from spec_mtp.masks import EMAVelocityMaskProvider, ESPMaskProvider
from spec_mtp.tree import build_static_tree
from spec_mtp.tree import block_layout

PASSED = 0


def check(name, condition):
    global PASSED
    if not condition:
        raise AssertionError(f'FAILED: {name}')
    PASSED += 1
    print(f'  ok: {name}')


# ------------------------------------------------------------ alignment
def test_position_index_matches_extrapolation_step():
    print('[position alignment]')
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(2, 20), dim=-1)
    tree = build_static_tree(probs, [3, 2], root_token=1, pruning=False)
    _, rel, owners, _ = block_layout(tree, 2)
    base = 1 + tree.num_nodes
    check('root masks are the first k mask rows',
          [owners[j] for j in range(2)] == [0, 0])
    check('root mask j sits at position offset j+1',
          [rel[base + j] for j in range(2)] == [1, 2])

    table = torch.zeros(4, 4)
    prompt = torch.tensor([[0., 0., 0., 0.],
                           [1., 0., 0., 0.],
                           [2., 0., 0., 0.]])
    provider = EMAVelocityMaskProvider(2, table, beta=0.9, step_scale=1.0)
    provider.init_from_prompt(prompt)
    expected = torch.tensor([[3., 0., 0., 0.],
                             [4., 0., 0., 0.]])
    check('mask i extrapolates i steps, matching the position offset',
          torch.allclose(provider.masks(), expected))


# ------------------------------------------------------------ synthetic
def test_synthetic_trajectories():
    print('[synthetic EMA geometry]')
    beta = 0.9
    v = torch.tensor([3.0, -1.0, 2.0])
    table = torch.zeros(2, 3)

    linear = torch.stack([i * v for i in range(8)])
    provider = EMAVelocityMaskProvider(3, table, beta=beta, step_scale=1.0)
    provider.init_from_prompt(linear)
    check('constant diff: vhat equals the diff at any beta',
          torch.allclose(provider._velocity, v))
    future = torch.stack([linear[-1] + i * v for i in (1, 2, 3)])
    check('constant diff: masks equal the true future embeddings',
          torch.allclose(provider.masks(), future))

    # Bounce between 0 and v: diffs are +v, -v, +v, -v, ... Raw EMA cancels;
    # the unit-direction form keeps a typical-diff step length.
    alternating = torch.stack([
        torch.zeros_like(v) if i % 2 == 0 else v for i in range(80)
    ])
    diffs = alternating[1:] - alternating[:-1]
    vhat = diffs[0].clone()
    ref = diffs[0].float().norm()
    for j in range(1, diffs.size(0)):
        vhat = beta * vhat + (1.0 - beta) * diffs[j]
        ref = beta * ref + (1.0 - beta) * diffs[j].float().norm()
    raw = EMAVelocityMaskProvider(1, table, beta=beta, step_scale=1.0)
    raw.init_from_prompt(alternating)
    check('alternating diff: raw EMA matches the hand-computed recursion',
          torch.allclose(raw._velocity, vhat, atol=1e-5))
    raw_step = raw.masks()[0] - raw._anchor
    check('alternating diff: raw step is much smaller than a typical hop',
          raw_step.norm() < 0.2 * v.norm())

    normed = EMAVelocityMaskProvider(
        1, table, beta=beta, step_scale=1.0, normalize=True)
    normed.init_from_prompt(alternating)
    norm_step = normed.masks()[0] - normed._anchor
    check('alternating diff: normalization restores the EMA of hop lengths',
          torch.allclose(norm_step.norm(), ref, atol=1e-4))
    check('alternating diff: normalized direction follows residual vhat',
          torch.allclose(norm_step / norm_step.norm(),
                         vhat / vhat.norm(), atol=1e-5))
    print(f'  synthetic raw_step_ratio={float(raw_step.norm() / v.norm()):.4f} '
          f'norm_step_ratio={float(norm_step.norm() / v.norm()):.4f}')


# ------------------------------------------------------------ live decode
def _tiny_model():
    torch.manual_seed(7)
    return HFCausalLM(tiny_config={
        'vocab_size': 199, 'hidden_size': 64, 'intermediate_size': 128,
        'num_layers': 2, 'num_heads': 4, 'num_kv_heads': 2,
        'max_position_embeddings': 1024,
    })


def _spec_cfg(num_masks, branches, update=True):
    return _convert({
        'method': 'ema_velocity',
        'num_masks': num_masks,
        'impl': 'efficient',
        'tree': {'mode': 'static', 'branches': branches, 'pruning': True},
        'ema': {'beta': 0.9, 'step_scale': 1.0, 'normalize': False,
                'update': update},
    })


def _record_commits(model, spec_cfg, eval_cfg, prompt, seed):
    calls = []
    original = EMAVelocityMaskProvider.on_token_committed

    def wrapped(self, new_embed, prev_embed):
        calls.append((new_embed.detach().cpu().clone(),
                      prev_embed.detach().cpu().clone()))
        return original(self, new_embed, prev_embed)

    EMAVelocityMaskProvider.on_token_committed = wrapped
    try:
        dec = SpecDecoder(model, spec_cfg, eval_cfg,
                          generator=torch.Generator().manual_seed(seed), mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
        out, stats = dec.generate(prompt)
    finally:
        EMAVelocityMaskProvider.on_token_committed = original
    return out, stats, calls


def test_rejected_drafts_do_not_enter_history():
    print('[live commit history, mixed accept/reject]')
    model = _tiny_model()
    eval_cfg = _convert({'max_new_tokens': 48, 'temperature': 0.0,
                         'stop_on_eos': False})
    # (num_masks, branches, prompt seed): chosen so the random tiny model
    # accepts some drafts and rejects others.
    cases = [(1, [14], 2), (2, [7, 2], 0)]
    for k, branches, prompt_seed in cases:
        tag = f'k={k} branches={branches} prompt_seed={prompt_seed}'
        prompt = torch.randint(0, 199, (1, 20),
                               generator=torch.Generator().manual_seed(prompt_seed))
        out, stats, calls = _record_commits(
            model, _spec_cfg(k, branches), eval_cfg, prompt, seed=100)
        ar_out, _ = ar_generate(model, prompt, eval_cfg,
                                generator=torch.Generator().manual_seed(100))
        check(f'{tag}: output matches AR', torch.equal(out, ar_out))

        draft_nodes = stats['decode_calls'] * sum(branches)
        check(f'{tag}: some drafts accepted and some rejected',
              0 < stats['accepted'] < draft_nodes)
        attempt = stats['attempt_by_depth']
        accept = stats['accept_by_depth']
        check(f'{tag}: depth counts add up to accepted and stay within attempts',
              sum(accept) == stats['accepted']
              and all(0 <= accept[d] <= attempt[d] for d in range(4))
              and attempt[1] == stats['decode_calls'])

        embeds = model.embed(out)[0].detach().cpu()
        t = prompt.size(1)
        check(f'{tag}: one history update per committed token',
              len(calls) == stats['new_tokens'] == out.size(1) - t)
        check(f'{tag}: every update is the committed (new, prev) pair in order',
              all(torch.equal(new, embeds[t + i])
                  and torch.equal(prev, embeds[t + i - 1])
                  for i, (new, prev) in enumerate(calls)))

        # Replay over committed tokens only; must equal a fresh EMA over the
        # full output, so no rejected node ever leaked into the state.
        table = model.embedding_table.detach().cpu()
        replay = EMAVelocityMaskProvider(k, table, beta=0.9)
        replay.init_from_prompt(embeds[:t])
        prev = embeds[t - 1]
        for tok_embed in embeds[t:]:
            replay.on_token_committed(tok_embed, prev)
            prev = tok_embed
        fresh = EMAVelocityMaskProvider(k, table, beta=0.9)
        fresh.init_from_prompt(embeds)
        check(f'{tag}: committed-only replay equals fresh full-output EMA',
              torch.allclose(replay._velocity, fresh._velocity, atol=1e-5)
              and torch.equal(replay._anchor, embeds[-1]))
        print(f'  {tag}: updates={len(calls)} accepted={stats["accepted"]} '
              f'draft_nodes={draft_nodes} decode_calls={stats["decode_calls"]}')


def test_frozen_velocity_still_tracks_anchor():
    print('[live frozen velocity]')
    model = _tiny_model()
    eval_cfg = _convert({'max_new_tokens': 16, 'temperature': 0.0,
                         'stop_on_eos': False})
    prompt = torch.randint(0, 199, (1, 20), generator=torch.Generator().manual_seed(2))
    snapshots = []
    original = EMAVelocityMaskProvider.on_token_committed

    def wrapped(self, new_embed, prev_embed):
        before = self._velocity.detach().cpu().clone()
        original(self, new_embed, prev_embed)
        snapshots.append((before,
                          self._velocity.detach().cpu().clone(),
                          self._anchor.detach().cpu().clone(),
                          new_embed.detach().cpu().clone()))

    EMAVelocityMaskProvider.on_token_committed = wrapped
    try:
        dec = SpecDecoder(model, _spec_cfg(1, [14], update=False), eval_cfg,
                          generator=torch.Generator().manual_seed(4), mask_policy=build_mask_provider(_spec_cfg(1, [14], update=False), model.embedding_table))
        out, stats = dec.generate(prompt)
    finally:
        EMAVelocityMaskProvider.on_token_committed = original

    check('frozen run commits several tokens', len(snapshots) > 1)
    check('frozen run never changes velocity',
          all(torch.equal(before, after) for before, after, _, _ in snapshots))
    check('frozen run anchor follows each committed embedding',
          all(torch.equal(anchor, new) for _, _, anchor, new in snapshots))
    embeds = model.embed(out)[0].detach().cpu()
    check('frozen anchor ends on the last output token',
          torch.equal(snapshots[-1][2], embeds[-1]))


def test_order0_level_plus_slope():
    print('[order-0 level plus order-1 slope]')
    torch.manual_seed(0)
    table = torch.zeros(2, 3)
    prompt = torch.tensor([[0., 0., 0.],
                           [2., 0., 0.],
                           [3., 1., 0.],
                           [3., 3., 0.]])
    commits = [torch.tensor([4., 3., 1.]), torch.tensor([4., 2., 2.])]

    esp = ESPMaskProvider(2, table, lam=0.1, update=True)
    both = EMAVelocityMaskProvider(
        2, table, beta=0.9, step_scale=0.0, level='order0', lam=0.1)
    esp.init_from_prompt(prompt)
    both.init_from_prompt(prompt)
    check('order0 with gamma 0 matches the prompt mean',
          torch.allclose(both.masks(), esp.masks()))
    prev = prompt[-1]
    for new in commits:
        esp.on_token_committed(new, prev)
        both.on_token_committed(new, prev)
        prev = new
    check('order0 with gamma 0 tracks the ESP level after commits',
          torch.allclose(both.masks(), esp.masks()))

    sloped = EMAVelocityMaskProvider(
        2, table, beta=0.9999, step_scale=1.0, level='order0', lam=0.1)
    sloped.init_from_prompt(prompt)
    # beta=0.9999 barely leaves the first difference, (2,0,0).
    first = prompt[1] - prompt[0]
    check('beta 0.9999 keeps vhat near the first difference',
          torch.allclose(sloped._velocity, first, atol=1e-3))
    steps = torch.tensor([[1.], [2.]])
    expected = sloped._level.unsqueeze(0) + steps * sloped._velocity.unsqueeze(0)
    check('order0 plus slope is level + i * vhat',
          torch.allclose(sloped.masks(), expected))


def main():
    test_position_index_matches_extrapolation_step()
    test_synthetic_trajectories()
    test_order0_level_plus_slope()
    test_rejected_drafts_do_not_enter_history()
    test_frozen_velocity_still_tracks_anchor()
    print(f'\nAll {PASSED} EMA history checks passed.')


if __name__ == '__main__':
    main()
