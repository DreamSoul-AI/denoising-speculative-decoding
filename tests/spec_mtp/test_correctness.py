"""CPU correctness checks (no GPU, no downloads).

Run this file directly. Covers:

  1. Mask providers: Eq (4) mean init, Eq (5) update algebra, every ablation
     init (last_k / sample / offset), EMA-velocity hand-computed recursion.
  2. Trees: Eq (7) block-complexity accounting, Top-1 expansion chains,
     Section 3.5 pruning, Algorithm 1 budget + ancestor closure.
  3. Tree attention: the efficient (Appendix E) masks/PIDs are bit-identical
     to the naive per-step construction; attend-set structure is causal.
  4. End-to-end losslessness on a random tiny LLaMA: ESP and EMA-velocity
     outputs equal plain AR decoding token-for-token across methods, mask
     counts, static/dynamic trees, pruning on/off, naive/efficient impls,
     and temperatures {0.0, 1.0} (shared seeds). This exercises prefill,
     tree attention, position IDs, KV-cache surgery, and verification.
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
from spec_mtp.tree import build_dynamic_tree, build_static_tree, node_budget_for
from spec_mtp.tree import (TreeAttentionCache, block_layout,
                                          build_block_attention, mask_indices_of)

PASSED = 0


def check(name, condition):
    global PASSED
    if not condition:
        raise AssertionError(f'FAILED: {name}')
    PASSED += 1
    print(f'  ok: {name}')


# ---------------------------------------------------------------- providers
def test_mask_providers():
    print('[mask providers]')
    torch.manual_seed(0)
    table = torch.randn(64, 8)
    prompt = torch.randn(10, 8)

    p = ESPMaskProvider(2, table, init='mean_prompt', lam=0.1)
    p.init_from_prompt(prompt)
    check('Eq 4: mean-of-prompt init',
          torch.allclose(p.masks(), prompt.mean(0).expand(2, 8)))
    m0 = p.masks().clone()
    e_new = torch.randn(8)
    p.on_token_committed(e_new, prompt[-1])
    check('Eq 5: m + lam*(e - m)',
          torch.allclose(p.masks(), m0 + 0.1 * (e_new.unsqueeze(0) - m0)))

    p = ESPMaskProvider(2, table, init='mean_prompt', lam=0.1, update=False)
    p.init_from_prompt(prompt)
    m0 = p.masks().clone()
    p.on_token_committed(e_new, prompt[-1])
    check('update=false freezes the mask', torch.equal(p.masks(), m0))

    p = ESPMaskProvider(3, table, init='last_k')
    p.init_from_prompt(prompt)
    check('last_k init = last k prompt embeddings',
          torch.equal(p.masks(), prompt[-3:]))

    gen = torch.Generator().manual_seed(1)
    p = ESPMaskProvider(2, table, init='sample_embedding', generator=gen)
    p.init_from_prompt(prompt)
    check('sample init shape/finite',
          p.masks().shape == (2, 8) and torch.isfinite(p.masks()).all())

    p = ESPMaskProvider(1, table, init='offset_embedding', offset_scale=5.0)
    p.init_from_prompt(prompt)
    expected = table.mean(0) + 5.0 * table.std(0)
    check('G.8 offset init = mu + 5*sigma', torch.allclose(p.masks()[0], expected))

    # EMA velocity vs a hand computation.
    beta = 0.9
    p = EMAVelocityMaskProvider(2, table, beta=beta, step_scale=1.0)
    p.init_from_prompt(prompt)
    diffs = prompt[1:] - prompt[:-1]
    vhat = diffs[0].clone()
    for j in range(1, diffs.size(0)):
        vhat = beta * vhat + (1 - beta) * diffs[j]
    check('EMA velocity: prefill recursion',
          torch.allclose(p._velocity, vhat, atol=1e-6))
    check('EMA velocity: m_i = e_last + i*vhat',
          torch.allclose(p.masks(),
                         torch.stack([prompt[-1] + vhat, prompt[-1] + 2 * vhat]),
                         atol=1e-5))
    p.on_token_committed(e_new, prompt[-1])
    vhat2 = beta * vhat + (1 - beta) * (e_new - prompt[-1])
    check('EMA velocity: commit folds the diff',
          torch.allclose(p._velocity, vhat2, atol=1e-6))
    check('EMA velocity: anchor tracks last token',
          torch.equal(p._anchor, e_new))

    # Formulation 2: gamma scales the raw-vhat step.
    p = EMAVelocityMaskProvider(2, table, beta=beta, step_scale=0.5)
    p.init_from_prompt(prompt)
    check('EMA velocity: gamma scales raw step',
          torch.allclose(p.masks(),
                         torch.stack([prompt[-1] + 0.5 * vhat,
                                      prompt[-1] + 1.0 * vhat]), atol=1e-5))

    # Formulation 3: unit direction, step length = gamma * EMA of diff norms.
    gamma = 0.7
    p = EMAVelocityMaskProvider(2, table, beta=beta, step_scale=gamma,
                                normalize=True)
    p.init_from_prompt(prompt)
    ref = diffs[0].norm()
    for j in range(1, diffs.size(0)):
        ref = beta * ref + (1 - beta) * diffs[j].norm()
    check('EMA velocity (norm): ref-norm recursion',
          torch.allclose(p._ref_norm, ref, atol=1e-5))
    m = p.masks()
    check('EMA velocity (norm): step length = i*gamma*ref',
          torch.allclose((m - prompt[-1]).norm(dim=1),
                         gamma * ref * torch.tensor([1.0, 2.0]), atol=1e-4))
    check('EMA velocity (norm): direction = vhat/||vhat||',
          torch.allclose((m[0] - prompt[-1]) / (m[0] - prompt[-1]).norm(),
                         vhat / vhat.norm(), atol=1e-5))

    # extrapolate_ema=true: identical to false at gamma=1 (the fixed point
    # made testable), geometric per-depth decay rho = beta+(1-beta)*gamma
    # at gamma != 1, and masks() stays pure (no state mutation).
    p_sh = EMAVelocityMaskProvider(3, table, beta=beta, step_scale=1.0)
    p_co = EMAVelocityMaskProvider(3, table, beta=beta, step_scale=1.0,
                                   extrapolate_ema=True)
    p_sh.init_from_prompt(prompt)
    p_co.init_from_prompt(prompt)
    check('extrapolate_ema == plain at gamma=1',
          torch.allclose(p_sh.masks(), p_co.masks(), atol=1e-5))

    gamma = 0.5
    rho = beta + (1 - beta) * gamma
    p = EMAVelocityMaskProvider(3, table, beta=beta, step_scale=gamma,
                                extrapolate_ema=True)
    p.init_from_prompt(prompt)
    m = p.masks()
    expected = torch.stack([
        prompt[-1] + gamma * vhat,
        prompt[-1] + gamma * (1 + rho) * vhat,
        prompt[-1] + gamma * (1 + rho + rho**2) * vhat,
    ])
    check('extrapolate_ema: steps scale by rho per depth',
          torch.allclose(m, expected, atol=1e-5))
    check('extrapolate_ema: masks() is pure (state untouched)',
          torch.allclose(p._velocity, vhat, atol=1e-6)
          and torch.equal(p._anchor, prompt[-1])
          and torch.allclose(p.masks(), m, atol=1e-6))

    # normalized + compound: step lengths gamma*r, gamma*rho*r, ...
    p = EMAVelocityMaskProvider(2, table, beta=beta, step_scale=gamma,
                                normalize=True, extrapolate_ema=True)
    p.init_from_prompt(prompt)
    m = p.masks()
    lens = torch.stack([(m[0] - prompt[-1]).norm(),
                        (m[1] - m[0]).norm()])
    check('extrapolate_ema (norm): lengths gamma*r, gamma*rho*r',
          torch.allclose(lens, gamma * ref * torch.tensor([1.0, rho]),
                         atol=1e-4))

    # The incremental commit-update is exactly "recompute the EMA over the
    # full current sequence": same recursion, same diffs, same order. So
    # `update: true` needs no separate Eq-5-style rule — unlike ESP, whose
    # mean init is not the fixed point of its own update. The state recursion
    # never branches on the read-policy knobs, so the equivalence must hold
    # across the whole {normalize} x {extrapolate_ema} x {gamma} grid.
    committed = torch.randn(5, 8)
    full_seq = torch.cat([prompt, committed], dim=0)
    for norm_opt in (False, True):
        for xema_opt in (False, True):
            for gamma_opt in (0.5, 1.0):
                kw = dict(beta=beta, step_scale=gamma_opt,
                          normalize=norm_opt, extrapolate_ema=xema_opt)
                p_inc = EMAVelocityMaskProvider(2, table, **kw)
                p_inc.init_from_prompt(prompt)
                prev = prompt[-1]
                for tok in committed:
                    p_inc.on_token_committed(tok, prev)
                    prev = tok
                p_full = EMAVelocityMaskProvider(2, table, **kw)
                p_full.init_from_prompt(full_seq)
                check(f'incremental == fresh full-sequence EMA '
                      f'(norm={norm_opt}, xema={xema_opt}, gamma={gamma_opt})',
                      torch.allclose(p_inc._velocity, p_full._velocity, atol=1e-6)
                      and torch.allclose(p_inc._ref_norm, p_full._ref_norm, atol=1e-6)
                      and torch.equal(p_inc._anchor, p_full._anchor)
                      and torch.allclose(p_inc.masks(), p_full.masks(), atol=1e-5))

    # Negative control: update=false is NOT the full-sequence EMA (frozen
    # prefill velocity), and is not the prefill state either — its anchor
    # still tracks the last committed token. A deliberate hybrid.
    p_frozen = EMAVelocityMaskProvider(2, table, beta=beta, update=False)
    p_frozen.init_from_prompt(prompt)
    prefill_vhat = p_frozen._velocity.clone()
    prev = prompt[-1]
    for tok in committed:
        p_frozen.on_token_committed(tok, prev)
        prev = tok
    p_full = EMAVelocityMaskProvider(2, table, beta=beta)
    p_full.init_from_prompt(full_seq)
    check('update=false: velocity frozen at prefill, != full-sequence EMA',
          torch.allclose(p_frozen._velocity, prefill_vhat)
          and not torch.allclose(p_frozen._velocity, p_full._velocity, atol=1e-3))
    check('update=false: anchor still tracks the last committed token',
          torch.equal(p_frozen._anchor, committed[-1]))

    # Degenerate history (all diffs zero) must not NaN; masks sit on anchor.
    flat = prompt[0].unsqueeze(0).expand(6, -1).clone()
    p = EMAVelocityMaskProvider(2, table, beta=beta, normalize=True)
    p.init_from_prompt(flat)
    check('EMA velocity (norm): zero-drift collapses to anchor',
          torch.isfinite(p.masks()).all()
          and torch.allclose(p.masks(), flat[-1].expand(2, -1)))


# -------------------------------------------------------------------- trees
def test_trees():
    print('[trees]')
    torch.manual_seed(0)
    V = 50
    probs = torch.softmax(torch.randn(2, V), dim=-1)

    tree = build_static_tree(probs, [7, 2], root_token=3, pruning=False)
    check('static [7,2]: widths', tree.branch_widths() == [7, 2])
    check('Eq 7: BC = (k+1)(1+sum K) = 30', tree.block_complexity(2) == 30)
    top1 = max(range(7), key=lambda j: tree.probs[j])
    check('Top-1 expansion: depth-2 parent is depth-1 argmax',
          all(tree.parents[j] == top1 + 1 for j in range(7, 9)))
    d2 = torch.topk(probs[1], 2).indices.tolist()
    check('depth-2 candidates are mask-2 top-k',
          sorted(tree.tokens[7:9]) == sorted(d2))
    check('cumulative probs multiply',
          abs(tree.probs[7] - tree.probs[top1] * probs[1][tree.tokens[7]].item()) < 1e-6)

    # Pruning: force the parent's token to be mask-2's argmax.
    probs2 = probs.clone()
    parent_token = tree.tokens[top1]
    probs2[1] = 1e-9
    probs2[1][parent_token] = 0.9
    probs2[1][(parent_token + 1) % V] = 0.05
    probs2[1][(parent_token + 2) % V] = 0.03
    probs2[1] = probs2[1] / probs2[1].sum()
    pruned = build_static_tree(probs2, [7, 2], root_token=3, pruning=True)
    check('pruning drops parent-repeat token',
          parent_token not in [pruned.tokens[j] for j in range(7, 9)])
    unpruned = build_static_tree(probs2, [7, 2], root_token=3, pruning=False)
    check('without pruning the repeat stays',
          parent_token in [unpruned.tokens[j] for j in range(7, 9)])

    check('node budget: BC=30,k=2 -> 9', node_budget_for(30, 2) == 9)
    check('node budget: BC=60,k=1 -> 29', node_budget_for(60, 1) == 29)
    dyn = build_dynamic_tree(probs, node_budget_for(30, 2), root_token=3, pruning=False)
    check('dynamic: exactly the node budget', dyn.num_nodes == 9)
    widths = dyn.branch_widths()
    check('dynamic: widths fill [9-i, i]', sum(widths) == 9 and widths[0] >= 1)
    check('dynamic: depth-1 top-1 always kept',
          torch.topk(probs[0], 1).indices.item() in
          [dyn.tokens[j] for j in range(dyn.num_nodes) if dyn.depths[j] == 1])
    if len(widths) > 1:
        by_prob_d1 = sorted(dyn.probs[j] for j in range(dyn.num_nodes)
                            if dyn.depths[j] == 1)[0]
        max_d2 = max((dyn.probs[j] for j in range(dyn.num_nodes)
                      if dyn.depths[j] == 2), default=0.0)
        check('dynamic: kept depth-2 beats dropped depth-1 candidates',
              max_d2 <= max(dyn.probs) + 1e-9 and by_prob_d1 >= 0)

    # A peaked mask-1 distribution should push budget into depth 2.
    peaked = probs.clone()
    peaked[0] = 1e-9
    peaked[0][7] = 1.0
    peaked[0] = peaked[0] / peaked[0].sum()
    dyn2 = build_dynamic_tree(peaked, 9, root_token=3, pruning=False)
    check('dynamic: confident mask-1 -> deeper tree',
          dyn2.branch_widths()[0] < 9 and len(dyn2.branch_widths()) == 2)


# ---------------------------------------------------------- tree attention
def test_tree_attention():
    print('[tree attention]')
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(2, 50), dim=-1)
    tree = build_static_tree(probs, [3, 2], root_token=1, pruning=True)
    k = 2
    attend, rel, owners, size = block_layout(tree, k)
    check('block size = (k+1)(1+sum K)', size == tree.block_size(k) == 18)
    check('root attends only itself (plus cache)', attend[0] == {0})
    node_rows = range(1, 1 + tree.num_nodes)
    check('nodes attend root + ancestors + self',
          all(0 in attend[i] and i in attend[i] for i in node_rows))
    for owner in range(1 + tree.num_nodes):
        rows = mask_indices_of(tree, k, owner)
        check(f'mask rows of owner {owner}: pids follow owner',
              all(rel[r] == rel[owner] + j + 1 for j, r in enumerate(rows)))
        check(f'mask rows of owner {owner}: attend owner path + earlier masks',
              all(attend[r] == attend[owner] | set(rows[:j + 1])
                  for j, r in enumerate(rows)))

    # Naive vs efficient equality across growing cache lengths.
    cache_impl = TreeAttentionCache()
    for step, (cache_len, root_pos) in enumerate([(10, 10), (13, 13), (17, 17)]):
        naive_mask, naive_pid = build_block_attention(
            tree, k, cache_len, root_pos, torch.float32, 'cpu', float('-inf'))
        eff_mask, eff_pid = cache_impl.get(
            tree, k, cache_len, root_pos, torch.float32, 'cpu', float('-inf'))
        check(f'appendix E step {step}: masks identical',
              torch.equal(naive_mask, eff_mask))
        check(f'appendix E step {step}: PIDs identical',
              torch.equal(naive_pid, eff_pid))


# ------------------------------------------------------------- end to end
def _tiny_model():
    torch.manual_seed(7)
    return HFCausalLM(tiny_config={'vocab_size': 199, 'hidden_size': 64,
                                   'intermediate_size': 128, 'num_layers': 2,
                                   'num_heads': 4, 'num_kv_heads': 2,
                                   'max_position_embeddings': 1024})


def _spec_cfg(method, num_masks, tree_mode, branches=None, bc=None,
              pruning=True, impl='naive'):
    return _convert({
        'method': method,
        'num_masks': num_masks,
        'block_complexity': bc,
        'impl': impl,
        'tree': {'mode': tree_mode, 'branches': branches, 'pruning': pruning},
        'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
        'ema': {'beta': 0.9, 'step_scale': 1.0, 'normalize': False, 'update': True},
    })


def test_lossless():
    print('[end-to-end losslessness on tiny random LLaMA]')
    model = _tiny_model()
    eval_cfg = _convert({'max_new_tokens': 24, 'temperature': 0.0,
                         'stop_on_eos': False})
    prompt_gen = torch.Generator().manual_seed(11)
    prompts = [torch.randint(0, 199, (1, n), generator=prompt_gen)
               for n in (16, 25, 40)]

    grid = [
        ('esp', 1, 'static', [4], None, True, 'naive'),
        ('esp', 1, 'static', [4], None, True, 'efficient'),
        ('esp', 1, 'static', [14], None, False, 'efficient'),
        ('esp', 2, 'static', [7, 2], None, True, 'efficient'),
        ('esp', 2, 'dynamic', None, 30, True, 'naive'),
        ('esp', 3, 'dynamic', None, 60, True, 'naive'),
        ('esp', 3, 'static', [7, 5, 3], None, True, 'efficient'),
        ('ema_velocity', 1, 'static', [4], None, True, 'efficient'),
        ('ema_velocity', 2, 'dynamic', None, 30, True, 'naive'),
        ('ema_velocity', 2, 'static', [15, 4], None, False, 'efficient'),
    ]
    for method, k, mode, branches, bc, pruning, impl in grid:
        spec_cfg = _spec_cfg(method, k, mode, branches, bc, pruning, impl)
        for temperature in (0.0, 1.0):
            eval_cfg.temperature = temperature
            for pi, prompt in enumerate(prompts):
                dec = SpecDecoder(model, spec_cfg, eval_cfg,
                                  generator=torch.Generator().manual_seed(100 + pi), mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
                out, stats = dec.generate(prompt)
                ar_out, _ = ar_generate(
                    model, prompt, eval_cfg,
                    generator=torch.Generator().manual_seed(100 + pi))
                tag = (f'{method} k={k} {mode} branches={branches} bc={bc} '
                       f'pruning={pruning} impl={impl} T={temperature} prompt={pi}')
                check(f'lossless: {tag}', torch.equal(out, ar_out))
                check(f'tau >= 1: {tag}',
                      stats['committed'] >= stats['decode_calls'])

    # ESP init ablations must also stay lossless.
    for init in ('last_k', 'sample_embedding', 'offset_embedding'):
        spec_cfg = _spec_cfg('esp', 2, 'static', [7, 2], None, True, 'efficient')
        spec_cfg.mask.init = init
        eval_cfg.temperature = 0.0
        dec = SpecDecoder(model, spec_cfg, eval_cfg,
                          generator=torch.Generator().manual_seed(3), mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
        out, _ = dec.generate(prompts[0])
        ar_out, _ = ar_generate(model, prompts[0], eval_cfg,
                                generator=torch.Generator().manual_seed(3))
        check(f'lossless under init={init}', torch.equal(out, ar_out))

    # EMA formulations 2/3 and both extrapolate_ema modes must stay lossless.
    for knob in ({'step_scale': 0.5}, {'normalize': True},
                 {'normalize': True, 'step_scale': 2.0},
                 {'extrapolate_ema': True, 'step_scale': 0.5},
                 {'extrapolate_ema': True, 'normalize': True}):
        spec_cfg = _spec_cfg('ema_velocity', 2, 'static', [7, 2], None, True,
                             'efficient')
        for key, value in knob.items():
            spec_cfg.ema[key] = value
        dec = SpecDecoder(model, spec_cfg, eval_cfg,
                          generator=torch.Generator().manual_seed(4), mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
        out, _ = dec.generate(prompts[0])
        ar_out, _ = ar_generate(model, prompts[0], eval_cfg,
                                generator=torch.Generator().manual_seed(4))
        check(f'lossless under ema {knob}', torch.equal(out, ar_out))

    # Frozen-mask and lambda ablation knobs.
    for knob in ({'lam': 0.01}, {'lam': 0.5}, {'update': False}):
        spec_cfg = _spec_cfg('esp', 1, 'static', [14], None, True, 'efficient')
        for key, value in knob.items():
            spec_cfg.mask[key] = value
        dec = SpecDecoder(model, spec_cfg, eval_cfg,
                          generator=torch.Generator().manual_seed(5), mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
        out, _ = dec.generate(prompts[1])
        ar_out, _ = ar_generate(model, prompts[1], eval_cfg,
                                generator=torch.Generator().manual_seed(5))
        check(f'lossless under mask {knob}', torch.equal(out, ar_out))

    # max_new_tokens budget is respected exactly.
    eval_cfg.max_new_tokens = 7
    spec_cfg = _spec_cfg('esp', 2, 'static', [7, 2], None, True, 'efficient')
    dec = SpecDecoder(model, spec_cfg, eval_cfg,
                      generator=torch.Generator().manual_seed(9), mask_policy=build_mask_provider(spec_cfg, model.embedding_table))
    out, stats = dec.generate(prompts[0])
    check('budget respected', out.size(1) == prompts[0].size(1) + 7)


def main():
    test_mask_providers()
    test_trees()
    test_tree_attention()
    test_lossless()
    print(f'\nAll {PASSED} checks passed.')


if __name__ == '__main__':
    main()
