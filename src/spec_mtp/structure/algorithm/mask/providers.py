"""Committed mask providers. Block-local prediction does not mutate committed state."""

from typing import Any
import torch

from spec_mtp.structure.algorithm.mask.fit import _als_lowrank


class OutputHiddenMaskProvider:
    needs_hidden = True

    def __init__(self, num_masks, embedding_table, order=1, beta=.9, beta_prime=.99):
        self.num_masks = int(num_masks)
        self.order = int(order)
        self.beta, self.beta_prime = float(beta), float(beta_prime)
        if self.order not in (0, 1) or not 0 <= self.beta <= self.beta_prime <= 1:
            raise ValueError('output masks require order0/1 and 0 <= beta <= beta_prime <= 1')
        self.scale = embedding_table.detach().float().norm(dim=-1).mean()
        self.anchor: Any = None
        self.velocity: Any = None

    def init_from_prompt(self, prompt_embeds, generator=None):
        if generator is not None:
            self.generator = generator
        self.anchor, self.velocity = None, None
        self.prefill_masks = prompt_embeds.mean(dim=0).expand(self.num_masks, -1).clone()

    def masks(self):
        return self.prefill_masks

    def observe_committed(self, hidden):
        y = hidden.detach().float()
        if not torch.isfinite(y).all():
            raise ValueError('nonfinite output hidden observation')
        if self.anchor is None:
            self.velocity = torch.zeros_like(y)
        elif self.order:
            self.velocity = self.beta * self.velocity + (1 - self.beta) * (y - self.anchor)
        self.anchor = y.clone()

    def init_hidden(self, prompt_hidden):
        for y in prompt_hidden:
            self.observe_committed(y)

    def trajectory(self, proxy_hiddens):
        """Hidden slots and the observation at each owner. Does not mutate committed state.

        Observation 0 is the last committed output hidden. Each later observation
        is the anchor after the previous in-block proxy has been absorbed.
        """
        if self.anchor is None or proxy_hiddens.shape != (self.num_masks, self.anchor.numel()):
            raise ValueError('output proxy shape mismatch')
        anchor, velocity = self.anchor.clone(), self.velocity.clone()
        slots, observations = [], []
        for owner in range(self.num_masks + 1):
            observations.append(anchor.clone())
            if owner < self.num_masks:
                y = proxy_hiddens[owner].detach().float()
                if self.order:
                    velocity = self.beta_prime * velocity + (1 - self.beta_prime) * (y - anchor)
                anchor = y
            else:
                anchor = anchor + velocity
            slots.append(torch.stack([anchor + slot * velocity for slot in range(self.num_masks)]))
        return torch.stack(slots), torch.stack(observations)

    def masks_per_owner(self, proxy_hiddens):
        """Predict depths 0..k; the decoder shares each depth across tree siblings."""
        # Proxies are at root..penultimate draft positions. The last owner
        # has no observation; simulate only, without mutating committed state.
        slots, _observations = self.trajectory(proxy_hiddens)
        return self._map_hidden_slots(slots)

    def _map_hidden_slots(self, slots):
        norms = slots.norm(dim=-1, keepdim=True)
        if not torch.isfinite(slots).all() or not torch.isfinite(self.scale) or self.scale <= 0 or (norms <= 0).any():
            raise ValueError('output mask mapping requires finite nonzero norms')
        return self.scale * slots / norms




class ESPMaskProvider:
    needs_hidden = False

    """Mask tokens per ESP Section 3.1 (Eq 4 init variants + Eq 5 update)."""

    def __init__(self, num_masks, embedding_table, init='mean_prompt', lam=0.1,
                 update=True, offset_scale=5.0, generator=None):
        self.num_masks = int(num_masks)
        self.embedding_table = embedding_table
        self.init = str(init).lower()
        self.lam = float(lam)
        self.update = bool(update)
        self.offset_scale = float(offset_scale)
        self.generator = generator
        self._masks = None  # [k, d]

    def init_from_prompt(self, prompt_embeds, generator=None):
        if generator is not None:
            self.generator = generator
        """prompt_embeds: [t, d] input embeddings of the full prompt."""
        k, d = self.num_masks, prompt_embeds.size(-1)
        if self.init == 'mean_prompt':
            # Eq (4): m_i = (1/t) sum_j e_j, identical for all i.
            mean = prompt_embeds.mean(dim=0, keepdim=True)
            self._masks = mean.expand(k, d).clone()
        elif self.init == 'last_k':
            if prompt_embeds.size(0) < k:
                raise ValueError(f'last_k init needs a prompt with >= {k} tokens')
            self._masks = prompt_embeds[-k:].clone()
        elif self.init == 'sample_embedding':
            mu = self.embedding_table.mean(dim=0)
            sigma = self.embedding_table.std(dim=0)
            noise = torch.randn(
                (k, d), generator=self.generator,
                device=mu.device, dtype=torch.float32,
            ).to(mu.dtype)
            self._masks = mu.unsqueeze(0) + noise * sigma.unsqueeze(0)
        elif self.init == 'offset_embedding':
            # G.8: mu + c * sigma pushes the mask outside the table distribution.
            mu = self.embedding_table.mean(dim=0)
            sigma = self.embedding_table.std(dim=0)
            mask = mu + self.offset_scale * sigma
            self._masks = mask.unsqueeze(0).expand(k, d).clone()
        else:
            raise ValueError(f'Unsupported mask.init: {self.init}')

    def masks(self):
        """Current mask embeddings [k, d]; shared across all tree positions."""
        return self._masks

    def on_token_committed(self, new_embed, prev_embed):
        """Eq (5) with e_{t+s} = the newly generated token's embedding."""
        if not self.update:
            return
        self._masks = self._masks + self.lam * (new_embed.unsqueeze(0) - self._masks)


class EMAVelocityMaskProvider:
    needs_hidden = False

    """Ours: EMA-of-differences velocity extrapolation replacing Eq (4)/(5).

    Three switchable formulations of the extrapolation step (spec.ema):

      1. normalize: false, step_scale: 1.0   (default; the meeting spec)
             m_i = origin + i * vhat
         origin is e_last when level=last, and the order-0 embedding EMA
         (prompt mean, then Eq (5)) when level=order0.
         Raw vhat: its data-dependent magnitude doubles as a confidence
         signal (directionally scattered history -> cancellation -> small
         step; coherent drift -> full step).
      2. normalize: false, step_scale: gamma
             m_i = origin + gamma * i * vhat
         Same, with a global shrink/stretch of the extrapolation (gamma is
         the regression-coefficient / trust-region knob; gamma = 1/(1-beta)
         also recovers raw-momentum-accumulator semantics).
      3. normalize: true, step_scale: gamma
             m_i = origin + gamma * i * r * vhat / ||vhat||
         Unit direction: the confidence signal is removed and gamma alone
         controls the step length, measured in units of r = EMA of observed
         per-token diff norms (so gamma stays dimensionless and
         model-agnostic; gamma = 1 is one typical token-step). Degenerate
         ||vhat|| ~ 0 collapses the masks onto the anchor.

    Orthogonal switch — how the velocity behaves across the k speculative
    extrapolation steps (the MTP mask slots), `extrapolate_ema`:

      false     (default) every step extrapolates with the same vhat:
                m_i = anchor + i * step. Exact closed form of "extrapolate,
                fold, extrapolate again" at gamma = 1, since folding the
                extrapolated diff into the EMA is a fixed point.
      true      the EMA keeps running WHILE extrapolating: each realized
                step is folded into (vhat, r) before the next step is
                taken. Identical to false at gamma = 1 (normalize: false);
                at gamma != 1 the per-step size scales geometrically by
                rho = beta + (1-beta)*gamma — a smooth depth-decaying
                (gamma < 1) or growing (gamma > 1) trust schedule.
                Within-block only: the committed-token state is never
                touched by this simulation.
    """

    _NORM_EPS = 1e-8

    def __init__(self, num_masks, embedding_table, beta=0.9, step_scale=1.0,
                 normalize=False, extrapolate_ema=False, update=True,
                 level='last', lam=0.1, generator=None):
        self.num_masks = int(num_masks)
        self.embedding_table = embedding_table  # unused; kept for a uniform ctor
        self.beta = float(beta)
        self.step_scale = float(step_scale)
        self.normalize = bool(normalize)
        self.extrapolate_ema = bool(extrapolate_ema)
        self.update = bool(update)
        self.level_mode = str(level).lower()
        if self.level_mode not in ('last', 'order0'):
            raise ValueError(f'Unsupported ema.level: {level}')
        self.lam = float(lam)
        self.generator = generator
        self._velocity: Any = None  # vhat, [d]
        self._anchor = None    # e of the last committed token, [d]
        self._level = None     # order-0 EMA of embeddings, [d]
        self._ref_norm: Any = None  # EMA of ||diff||, scalar (formulation 3 unit)

    def init_from_prompt(self, prompt_embeds, generator=None):
        if generator is not None:
            self.generator = generator
        if prompt_embeds.size(0) < 2:
            raise ValueError('EMA velocity needs a prompt with >= 2 tokens')
        diffs = prompt_embeds[1:] - prompt_embeds[:-1]  # v_j = e_{j+1} - e_j
        vhat = diffs[0].clone()
        ref = diffs[0].float().norm()
        for j in range(1, diffs.size(0)):
            vhat = self.beta * vhat + (1.0 - self.beta) * diffs[j]
            ref = self.beta * ref + (1.0 - self.beta) * diffs[j].float().norm()
        self._velocity = vhat
        self._ref_norm = ref
        self._anchor = prompt_embeds[-1].clone()
        # Order-0 initial value: mean of the prompt embeddings, same as ESP Eq (4).
        self._level = prompt_embeds.mean(dim=0).clone()

    def _step_vector(self, velocity, ref_norm):
        """One extrapolation step under the active formulation."""
        if not self.normalize:
            return self.step_scale * velocity
        norm = velocity.float().norm()
        if norm < self._NORM_EPS:
            return torch.zeros_like(velocity)
        unit = velocity / norm.to(velocity.dtype)
        return (self.step_scale * ref_norm.to(velocity.dtype)) * unit

    def _origin(self):
        """Level the first-order term is added to.

        last: the latest committed embedding. order0: the embedding EMA,
        initialized at the prompt mean and updated with lam.
        """
        if self.level_mode == 'order0':
            return self._level
        return self._anchor

    def masks(self):
        origin = self._origin()
        if not self.extrapolate_ema:
            # m_i = origin + i * step: closed form of iterated
            # extrapolate-and-fold at gamma = 1 (folding the extrapolated
            # diff into the EMA is a fixed point: beta*v + (1-beta)*v = v).
            steps = torch.arange(
                1, self.num_masks + 1,
                device=origin.device, dtype=origin.dtype,
            ).unsqueeze(1)
            step = self._step_vector(self._velocity, self._ref_norm)
            return origin.unsqueeze(0) + steps * step.unsqueeze(0)

        # extrapolate_ema: keep the EMA running over the realized steps. Local
        # copies only — masks() must stay pure w.r.t. the committed state.
        velocity = self._velocity.clone()
        ref_norm = self._ref_norm.clone()
        pos = origin.clone()
        out = []
        for _ in range(self.num_masks):
            step = self._step_vector(velocity, ref_norm)
            pos = pos + step
            out.append(pos.clone())
            velocity = self.beta * velocity + (1.0 - self.beta) * step
            ref_norm = (self.beta * ref_norm
                        + (1.0 - self.beta) * step.float().norm())
        return torch.stack(out)

    def on_token_committed(self, new_embed, prev_embed):
        if self.update:
            diff = new_embed - prev_embed
            self._velocity = self.beta * self._velocity + (1.0 - self.beta) * diff
            self._ref_norm = (self.beta * self._ref_norm
                              + (1.0 - self.beta) * diff.float().norm())
        # The anchor always tracks the last committed token, even when the
        # velocity EMA is frozen (update: false ablation).
        self._anchor = new_embed.clone()
        if self.level_mode == 'order0':
            # Same update as ESP Eq (5). Independent of the velocity flag.
            self._level = self._level + self.lam * (new_embed - self._level)


class ObservationBlendMaskProvider:
    """Protocol `trajectory-observation-blend-v1`.

    z_i = softmax(l_i) @ E is the expected input embedding under the previous
    verify forward's logits at mask slot i of the deepest accepted node; they
    predict the token at depth i of the current tree. Slot j of an owner at
    depth d stands in for depth d + j, so
        m[d, j] = mhat_j + g * (z_{d+j} - mhat_j)   if d + j < k, else mhat_j.
    Depth d and slot j are zero-based.
    The observation lives for one round and is never written into the
    trajectory state, so g = 0 is the wrapped provider token for token.

    condition='rejection' (`trajectory-observation-rejection-v1`): an owner's
    mask logits seed the next tree only if it is the deepest accepted node, in
    which case every child token was rejected. Slot 0 of an owner with
    children therefore uses z_d conditioned on "not a child token":
        z = (z_d - sum_c p_c e_c) / (1 - sum_c p_c),
    falling back to mhat_0 when the children hold all the mass.

    form='top1' replaces the expectation with the embedding of the most likely
    token (under rejection: the most likely non-child token), keeping the
    mask input on the token manifold.
    """
    needs_hidden = False
    PROTOCOLS = {('mean', 'none'): 'trajectory-observation-blend-v1',
                 ('mean', 'rejection'): 'trajectory-observation-rejection-v1',
                 ('top1', 'none'): 'trajectory-observation-top1-v1',
                 ('top1', 'rejection'): 'trajectory-observation-top1-rejection-v1'}
    _MIN_REMAINING = 1e-6

    def __init__(self, base, embedding_table, gain, condition='none', form='mean'):
        if getattr(base, 'needs_hidden', False):
            raise ValueError('observation blend wraps input-embedding providers only')
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not 0 <= gain <= 1:
            raise ValueError('observation gain must be a number in [0, 1]')
        if condition not in ('none', 'rejection'):
            raise ValueError('observation condition must be none or rejection')
        if form not in ('mean', 'top1'):
            raise ValueError('observation form must be mean or top1')
        self.base = base
        self.num_masks = base.num_masks
        self.embedding_table = embedding_table
        self.gain = float(gain)
        self.condition = condition
        self.form = form
        self.protocol = self.PROTOCOLS[form, condition]
        self._observation: Any = None
        self._probs: Any = None

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        self._observation = None
        self._probs = None

    def observe_mask_logits(self, mask_logits):
        if mask_logits.shape[0] != self.num_masks:
            raise ValueError('observation needs one logit row per mask slot')
        if not torch.isfinite(mask_logits).any(dim=-1).all() or torch.isnan(mask_logits).any():
            raise ValueError('observation logits need a finite candidate per row')
        probs = torch.softmax(mask_logits.float(), dim=-1)
        table = self.embedding_table
        self._probs = probs
        self._observation = table[probs.argmax(dim=-1)] if self.form == 'top1' else probs.to(table.dtype) @ table

    def observation(self):
        return self._observation

    def masks_by_depth(self):
        """[k + 1, k, d]: masks for owners at tree depth 0..k."""
        trajectory = self.base.masks()
        k = self.num_masks
        out = trajectory.unsqueeze(0).repeat(k + 1, 1, 1)
        if self._observation is None or self.gain == 0:
            return out
        z = self._observation.to(trajectory.device, trajectory.dtype)
        for depth in range(k):
            slots = k - depth
            out[depth, :slots] = trajectory[:slots] + self.gain * (z[depth:] - trajectory[:slots])
        return out

    def masks_for_tree(self, tree):
        """[1 + nodes, k, d]: per-owner masks in block order (root first)."""
        owners = self.masks_by_depth()[[0] + list(tree.depths)]
        if self.condition != 'rejection' or self._observation is None or self.gain == 0:
            return owners
        trajectory = self.base.masks()
        children: dict[int, list[int]] = {}
        for node, parent in enumerate(tree.parents, start=1):
            children.setdefault(parent, []).append(tree.tokens[node - 1])
        for owner in range(owners.shape[0]):
            depth = 0 if owner == 0 else tree.depths[owner - 1]
            if depth >= self.num_masks or owner not in children:
                continue
            tokens = torch.tensor(sorted(set(children[owner])), device=self._probs.device)
            p = self._probs[depth, tokens]
            remaining = 1.0 - float(p.sum())
            if remaining <= self._MIN_REMAINING:
                owners[owner, 0] = trajectory[0]
                continue
            if self.form == 'top1':
                allowed = self._probs[depth].clone()
                allowed[tokens] = -1.0
                z = self.embedding_table[allowed.argmax()]
            else:
                rejected = p.to(self.embedding_table.dtype) @ self.embedding_table[tokens]
                z = (self._observation[depth] - rejected) / remaining
            z = z.to(trajectory.device, trajectory.dtype)
            owners[owner, 0] = trajectory[0] + self.gain * (z - trajectory[0])
        return owners

    def masks(self):
        return self.masks_by_depth()[0]

    def on_token_committed(self, new_embed, prev_embed):
        self.base.on_token_committed(new_embed, prev_embed)


class BonusSlotMaskProvider:
    """Bonus-slot input, no extra forward.

    `hard` (`bonus-slot-hard-v1`): slot 0 of a depth-d owner is the embedding
    of argmax on the previous mask-logit row d.
    `topk` (`bonus-slot-topk-v1`): the same slot is the expected embedding of
    that row restricted to the draft's top-w tokens, w = tree branch width at
    that depth, renormalized. Deeper slots stay on the trajectory.
    """
    needs_hidden = False
    PROTOCOLS = {'hard': 'bonus-slot-hard-v1', 'topk': 'bonus-slot-topk-v1'}

    def __init__(self, base, embedding_table, mode='hard', widths=()):
        if getattr(base, 'needs_hidden', False):
            raise ValueError('bonus slot wraps input-embedding providers only')
        if mode not in self.PROTOCOLS:
            raise ValueError('bonus slot mode must be hard or topk')
        self.base = base
        self.num_masks = base.num_masks
        self.embedding_table = embedding_table
        self.mode = mode
        self.protocol = self.PROTOCOLS[mode]
        self.widths = [int(w) for w in widths]
        self._logits = None

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        self._logits = None

    def observe_mask_logits(self, mask_logits):
        if mask_logits.shape[0] != self.num_masks:
            raise ValueError('bonus slot needs one logit row per mask slot')
        if not torch.isfinite(mask_logits).any(dim=-1).all() or torch.isnan(mask_logits).any():
            raise ValueError('bonus slot logits need a finite candidate per row')
        self._logits = mask_logits.detach()

    def _slot_embedding(self, depth, trajectory):
        logits = self._logits
        assert logits is not None
        row = logits[depth]
        table = self.embedding_table
        if self.mode == 'hard':
            token = int(row.argmax())
            return table[token].to(device=trajectory.device, dtype=trajectory.dtype)
        width = self.widths[depth] if depth < len(self.widths) else 0
        if width <= 0:
            return None
        k = min(width, row.numel())
        vals, idx = torch.topk(row, k)
        weights = torch.softmax(vals.float(), dim=0).to(dtype=table.dtype)
        picked = table[idx].to(device=trajectory.device, dtype=trajectory.dtype)
        return weights.to(device=trajectory.device) @ picked

    def masks_for_tree(self, tree):
        trajectory = self.base.masks()
        owners = trajectory.unsqueeze(0).repeat(1 + tree.num_nodes, 1, 1)
        if self._logits is None:
            return owners
        for owner, depth in enumerate([0] + list(tree.depths)):
            if depth < self.num_masks:
                embed = self._slot_embedding(depth, trajectory)
                if embed is not None:
                    owners[owner, 0] = embed
        return owners

    def masks(self):
        out = self.base.masks().clone()
        if self._logits is not None and self.num_masks > 0:
            embed = self._slot_embedding(0, out)
            if embed is not None:
                out[0] = embed
        return out

    def on_token_committed(self, new_embed, prev_embed):
        self.base.on_token_committed(new_embed, prev_embed)


class KalmanMaskProvider:
    """Scalar random-walk Kalman update of the mask level. Protocol `kalman-mask-scalar-v1`.

    One shared variance, not a per-dimension or higher-order covariance:
    predict m, P+q; correct with K = P-/(P-+R) on the committed embedding.
    `online` reestimates scalar q, r from the innovation (Sage-Husa). `frozen`
    keeps the gain from the end of the prompt. This is not the affine residual.
    """
    needs_hidden = False
    PROTOCOLS = {'online': 'kalman-mask-scalar-v1', 'frozen': 'kalman-mask-frozen-k-v1'}

    def __init__(self, num_masks, embedding_table, mode='online', forget=0.95,
                 r_min=1e-6, q_min=1e-8, q_scale=0.01, generator=None):
        if mode not in self.PROTOCOLS:
            raise ValueError('kalman mode must be online or frozen')
        if isinstance(forget, bool) or not isinstance(forget, (int, float)) or not 0 < float(forget) <= 1:
            raise ValueError('kalman forget must be in (0, 1]')
        for name, value in (('r_min', r_min), ('q_min', q_min), ('q_scale', q_scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError(f'kalman {name} must be a positive number')
        self.num_masks = int(num_masks)
        self.embedding_table = embedding_table
        self.mode = mode
        self.protocol = self.PROTOCOLS[mode]
        self.forget = float(forget)
        self.r_min = float(r_min)
        self.q_min = float(q_min)
        self.q_scale = float(q_scale)
        self.generator = generator
        self._level = None
        self._p = self._q = self._r = self._k = 0.0
        self._t = 0
        self._locked = False

    def init_from_prompt(self, prompt_embeds, generator=None):
        if generator is not None:
            self.generator = generator
        if prompt_embeds.size(0) < 1:
            raise ValueError('kalman mask needs a prompt embedding')
        level = prompt_embeds.mean(dim=0).clone()
        variance = float((prompt_embeds - level).float().pow(2).mean().clamp_min(self.r_min))
        self._level = level
        self._r = variance
        self._q = max(variance * self.q_scale, self.q_min)
        self._p = variance
        self._t = 0
        self._locked = False
        self._k = 0.0
        for index in range(prompt_embeds.size(0)):
            self._step(prompt_embeds[index])
        if self.mode == 'frozen':
            self._locked = True

    def _step(self, observation):
        predicted = self._p + self._q
        error = observation - self._level
        innov = float(error.float().pow(2).mean())
        gain = predicted / (predicted + self._r)
        if self._locked:
            gain = self._k
        self._level = self._level + error * gain
        if self._locked:
            return
        self._k = gain
        self._p = (1.0 - gain) ** 2 * predicted + gain ** 2 * self._r
        if self.mode == 'online' and self.forget < 1.0:
            self._t += 1
            delta = (1.0 - self.forget) / (1.0 - self.forget ** (self._t + 1))
            self._r = max((1.0 - delta) * self._r + delta * (innov - predicted), self.r_min)
            self._q = max((1.0 - delta) * self._q + delta * (gain * gain * innov), self.q_min)

    def on_token_committed(self, new_embed, prev_embed):
        del prev_embed
        self._step(new_embed)

    def masks(self):
        level = self._level
        assert level is not None
        return level.unsqueeze(0).expand(self.num_masks, level.shape[-1]).clone()

    def diagnostics(self):
        phi = self._q / self._r
        root = (phi + (phi * phi + 4.0 * phi) ** 0.5) / 2.0
        return {'k': self._k, 'q': self._q, 'r': self._r, 'p': self._p,
                'phi': phi, 'prop8_beta_star': 1.0 / (1.0 + root)}


class VerifyCorrectionMaskProvider:
    """m = m_hat + K(z - m_hat) on slot 0. z is the last committed embedding.

    K is supplied by an offline fit. `reject_token` instead copies z into slot 0
    only after a decode round that committed one token. Online updates are the
    scalar least squares continued from that fit, not a new Kalman recursion.
    """
    needs_hidden = False

    def __init__(self, base, kind, payload=None, online=False):
        if getattr(base, 'needs_hidden', False):
            raise ValueError('verify correction wraps input-embedding providers only')
        if kind not in ('scalar', 'diagonal', 'affine', 'lowrank', 'reject_token'):
            raise ValueError('correction kind must be scalar, diagonal, affine, lowrank or reject_token')
        self.base = base
        self.num_masks = base.num_masks
        self.kind = kind
        self.online = bool(online)
        if self.online and kind not in ('scalar', 'affine', 'lowrank'):
            raise ValueError('online correction is scalar, affine or lowrank')
        payload = dict(payload or {})
        self._k = self._num = self._den = None
        self._diag = self._A = self._b = None
        self._U = self._V = self._U0 = self._V0 = None
        self._delta0 = self._error0 = self._extra_delta = self._extra_error = None
        self._rank = None
        self._ridge = None
        self._gram = self._rhs = self._gram0 = self._rhs0 = self._A0 = self._b0 = None
        if kind == 'scalar':
            self._k = float(payload['k'])
            self._num = float(payload.get('num', 0.0))
            self._den = float(payload.get('den', 0.0))
        elif kind == 'diagonal':
            self._diag = torch.tensor(payload['k'], dtype=torch.float32)
        elif kind == 'affine':
            self._A = torch.tensor(payload['A'], dtype=torch.float32)
            self._b = torch.tensor(payload['b'], dtype=torch.float32)
            self._ridge = float(payload['ridge']) if 'ridge' in payload else None
            if self.online:
                self._gram0 = torch.tensor(payload['gram'], dtype=torch.float64)
                self._rhs0 = torch.tensor(payload['rhs'], dtype=torch.float64)
                self._A0 = self._A.clone()
                self._b0 = self._b.clone()
                self._gram = self._gram0.clone()
                self._rhs = self._rhs0.clone()
        elif kind == 'lowrank':
            self._U = torch.as_tensor(payload['U'], dtype=torch.float32)
            self._V = torch.as_tensor(payload['V'], dtype=torch.float32)
            self._b = torch.as_tensor(payload['b'], dtype=torch.float32)
            self._rank = int(payload['rank'])
            self._ridge = float(payload['ridge'])
            if self.online:
                self._U0, self._V0, self._b0 = self._U.clone(), self._V.clone(), self._b.clone()
                self._delta0 = torch.as_tensor(payload['delta'], dtype=torch.float32)
                self._error0 = torch.as_tensor(payload['error'], dtype=torch.float32)
        self._z = None
        self._round_commits = 0
        self._masks_calls = 0
        self._pending = None
        self._online_updates = 0

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        self._z = None
        self._round_commits = 0
        self._masks_calls = 0
        self._pending = None
        self._online_updates = 0
        if self._gram0 is not None:
            rhs0, a0, b0 = self._rhs0, self._A0, self._b0
            assert rhs0 is not None and a0 is not None and b0 is not None
            self._gram = self._gram0.to(device=prompt_embeds.device).clone()
            self._rhs = rhs0.to(device=prompt_embeds.device).clone()
            self._A = a0.to(device=prompt_embeds.device).clone()
            self._b = b0.to(device=prompt_embeds.device).clone()
        if self._U0 is not None:
            v0, b0 = self._V0, self._b0
            assert v0 is not None and b0 is not None
            self._U, self._V, self._b = self._U0.clone(), v0.clone(), b0.clone()
            self._extra_delta = self._extra_error = None

    def on_token_committed(self, new_embed, prev_embed):
        if self._pending is not None and self.online and self._round_commits == 0:
            mhat, zed = self._pending
            error = (new_embed - mhat).float()
            delta = (zed - mhat).float()
            if self.kind == 'scalar':
                num, den = self._num, self._den
                assert num is not None and den is not None
                self._num = num + float((error * delta).sum())
                self._den = den + float((delta * delta).sum())
                self._k = self._num / max(self._den, 1e-12)
            elif self.kind == 'lowrank':
                row_d = delta.float().reshape(1, -1)
                row_e = error.float().reshape(1, -1)
                self._extra_delta = row_d if self._extra_delta is None else torch.cat([self._extra_delta, row_d])
                self._extra_error = row_e if self._extra_error is None else torch.cat([self._extra_error, row_e])
                delta0, error0 = self._delta0, self._error0
                assert delta0 is not None and error0 is not None
                delta_fit = delta0 if self._extra_delta is None else torch.cat([delta0, self._extra_delta])
                error_fit = error0 if self._extra_error is None else torch.cat([error0, self._extra_error])
                self._U, self._V, self._b = _als_lowrank(delta_fit, error_fit, self._rank, self._ridge, iters=1, v_init=self._V)
            else:
                row = torch.cat([delta.double(), delta.new_ones(1).double()])
                gram, rhs = self._gram, self._rhs
                assert gram is not None and rhs is not None
                self._gram = gram + row[:, None] * row[None, :]
                self._rhs = rhs + row[:, None] * error.double()[None, :]
                eye = torch.eye(self._gram.shape[0], dtype=row.dtype, device=row.device)
                if self._ridge is None:
                    penalty = 1e-8 * eye
                else:
                    eye[-1, -1] = 0.0
                    penalty = self._ridge * eye
                solution = torch.linalg.solve(self._gram + penalty, self._rhs)
                self._A = solution[:-1].T.float()
                self._b = solution[-1].float()
            self._online_updates += 1
            self._pending = None
        self.base.on_token_committed(new_embed, prev_embed)
        self._z = new_embed.detach()
        self._round_commits += 1

    def _corrected(self, mhat):
        observed = self._z
        assert observed is not None
        delta = observed.to(device=mhat.device, dtype=mhat.dtype) - mhat
        if self.kind == 'scalar':
            return mhat + delta * mhat.new_tensor(self._k)
        if self.kind == 'diagonal':
            diag = self._diag
            assert diag is not None
            return mhat + delta * diag.to(device=mhat.device, dtype=mhat.dtype)
        if self.kind == 'lowrank':
            left_v, right_u, bias = self._V, self._U, self._b
            assert left_v is not None and right_u is not None and bias is not None
            left = left_v.to(device=mhat.device, dtype=mhat.dtype)
            right = right_u.to(device=mhat.device, dtype=mhat.dtype)
            bias = bias.to(device=mhat.device, dtype=mhat.dtype)
            return mhat + (delta @ left) @ right.T + bias
        weight, bias = self._A, self._b
        assert weight is not None and bias is not None
        return mhat + delta @ weight.to(device=mhat.device, dtype=mhat.dtype).T + bias.to(device=mhat.device, dtype=mhat.dtype)

    def masks(self):
        out = self.base.masks().clone()
        if self.kind == 'reject_token':
            if self._masks_calls >= 2 and self._round_commits == 1 and self._z is not None:
                out[0] = self._z.to(dtype=out.dtype, device=out.device)
        elif self._z is not None and self._masks_calls >= 1:
            mhat = out[0]
            out[0] = self._corrected(mhat)
            if self.online:
                self._pending = (mhat.detach(), self._z.detach())
        self._masks_calls += 1
        self._round_commits = 0
        return out

    def diagnostics(self):
        report: dict[str, Any] = {'online_updates': self._online_updates}
        if self._A0 is not None:
            weight = self._A
            assert weight is not None
            report['A_drift'] = float((weight.double() - self._A0.to(device=weight.device).double()).pow(2).sum().sqrt())
        if self._U0 is not None:
            left, right, v0 = self._U, self._V, self._V0
            assert left is not None and right is not None and v0 is not None
            report['factor_drift'] = float((left - self._U0).pow(2).sum().sqrt() + (right - v0).pow(2).sum().sqrt())
        return report


class OnlineVerifyKalman:
    """ESP prior plus one verify update per round. Protocol `online-trajectory-kalman-v1`.

    Slot 0 becomes m_hat + K(z - m_hat). K, P, Q and R are per-dimension and
    reset to fixed constants on every prompt. `trajectory` emits ESP.
    `observation` keeps the verify update but replaces the prior with a random walk.
    """
    needs_hidden = False

    def __init__(self, base, part='full', forget=0.95, p0=1.0, q0=0.01, r0=1.0,
                 r_min=1e-6, q_min=1e-8):
        if getattr(base, 'needs_hidden', False):
            raise ValueError('online kalman wraps input-embedding providers only')
        if part not in ('full', 'trajectory', 'observation'):
            raise ValueError('online kalman part must be full, trajectory or observation')
        if isinstance(forget, bool) or not isinstance(forget, (int, float)) or not 0 < float(forget) <= 1:
            raise ValueError('online kalman forget must be in (0, 1]')
        for name, value in (('p0', p0), ('q0', q0), ('r0', r0), ('r_min', r_min), ('q_min', q_min)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError(f'online kalman {name} must be a positive number')
        self.base = base
        self.num_masks = base.num_masks
        self.part = part
        self.protocol = 'online-trajectory-kalman-v1'
        self.forget = float(forget)
        self.p0 = float(p0)
        self.q0 = float(q0)
        self.r0 = float(r0)
        self.r_min = float(r_min)
        self.q_min = float(q_min)
        self._p = self._q = self._r = None
        self._z = self._x = None
        self._pending = None
        self._t = 0
        self._seen = False

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        width = prompt_embeds.shape[-1]
        state = {'device': prompt_embeds.device, 'dtype': torch.float32}
        self._p = torch.full((width,), self.p0, **state)
        self._q = torch.full((width,), self.q0, **state)
        self._r = torch.full((width,), self.r0, **state)
        self._z = None
        self._x = torch.zeros(width, **state)
        self._pending = None
        self._t = 0
        self._seen = False

    def masks(self):
        out = self.base.masks().clone()
        if self.part == 'trajectory':
            self._pending = None
            return out
        prior = out[0].detach().to(dtype=torch.float32).clone()
        if self.part == 'observation':
            prior = self._x
        variance, process, noise = self._p, self._q, self._r
        assert variance is not None and process is not None and noise is not None
        p_pred = variance + process
        gain = p_pred / (p_pred + noise)
        if self.part == 'full' and self._z is not None:
            out[0] = (prior + gain * (self._z - prior)).to(dtype=out.dtype)
        elif self.part == 'observation' and self._seen:
            out[0] = prior.to(dtype=out.dtype)
        self._pending = (prior, p_pred, gain)
        return out

    def on_token_committed(self, new_embed, prev_embed):
        self.base.on_token_committed(new_embed, prev_embed)
        if self.part == 'trajectory' or self._pending is None:
            return
        prior, p_pred, gain = self._pending
        observed = new_embed.detach().to(device=prior.device, dtype=torch.float32).reshape(-1)
        innov = observed - prior
        innov2 = innov * innov
        variance, process, noise = self._p, self._q, self._r
        assert variance is not None and process is not None and noise is not None
        self._p = (1.0 - gain) ** 2 * p_pred + gain ** 2 * noise
        self._t += 1
        delta = (1.0 - self.forget) / (1.0 - self.forget ** (self._t + 1))
        self._r = torch.clamp((1.0 - delta) * noise + delta * (innov2 - p_pred), min=self.r_min)
        self._q = torch.clamp((1.0 - delta) * process + delta * (gain * gain * innov2), min=self.q_min)
        if self.part == 'full':
            self._z = observed
        else:
            self._x = prior + gain * innov
            self._seen = True
        self._pending = None

    def diagnostics(self):
        variance, process, noise = self._p, self._q, self._r
        assert variance is not None and process is not None and noise is not None
        return {'part': self.part, 'updates': self._t,
                'p_mean': float(variance.mean()), 'q_mean': float(process.mean()),
                'r_mean': float(noise.mean())}


class AlignedResidualKalman:
    """Scalar correlation on aligned residuals. Protocol `online-aligned-kalman-v1`.

    The carried state is F, not a per-dimension embedding gain. The innovation is
    y - m_hat at the same round. R is updated from that cosine and does not
    subtract P. Slot 0 is m_hat + F * previous residual.
    """
    needs_hidden = False

    def __init__(self, base, part='full', forget=0.95, p0=1.0, q0=0.01, r0=1.0,
                 r_min=1e-3, q_min=1e-6):
        if getattr(base, 'needs_hidden', False):
            raise ValueError('aligned kalman wraps input-embedding providers only')
        if part not in ('full', 'trajectory', 'observation'):
            raise ValueError('aligned kalman part must be full, trajectory or observation')
        if isinstance(forget, bool) or not isinstance(forget, (int, float)) or not 0 < float(forget) <= 1:
            raise ValueError('aligned kalman forget must be in (0, 1]')
        for name, value in (('p0', p0), ('q0', q0), ('r0', r0), ('r_min', r_min), ('q_min', q_min)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise ValueError(f'aligned kalman {name} must be a positive number')
        self.base = base
        self.num_masks = base.num_masks
        self.part = part
        self.protocol = 'online-aligned-kalman-v1'
        self.forget = float(forget)
        self.p0 = float(p0)
        self.q0 = float(q0)
        self.r0 = float(r0)
        self.r_min = float(r_min)
        self.q_min = float(q_min)
        self._f = 0.0
        self._p = self._q = self._r = 0.0
        self._nu = None
        self._state = None
        self._seen = False
        self._pending = None
        self._t = 0

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        self._f = 0.0
        self._p = self.p0
        self._q = self.q0
        self._r = self.r0
        self._nu = None
        self._state = torch.zeros(prompt_embeds.shape[-1], device=prompt_embeds.device, dtype=torch.float32)
        self._seen = False
        self._pending = None
        self._t = 0

    def masks(self):
        out = self.base.masks().clone()
        prior = out[0].detach().to(dtype=torch.float32).clone()
        if self.part == 'observation' and self._seen:
            state = self._state
            assert state is not None
            prior = state
            out[0] = prior.to(dtype=out.dtype)
        elif self.part == 'full' and self._nu is not None:
            out[0] = (prior + self._f * self._nu).to(dtype=out.dtype)
        self._pending = prior
        return out

    def on_token_committed(self, new_embed, prev_embed):
        self.base.on_token_committed(new_embed, prev_embed)
        if self.part == 'trajectory' or self._pending is None:
            return
        prior = self._pending
        observed = new_embed.detach().to(device=prior.device, dtype=torch.float32).reshape(-1)
        residual = observed - prior
        if self._nu is not None:
            denom = float(residual.norm() * self._nu.norm())
            cosine = float(torch.dot(residual, self._nu) / denom) if denom else 0.0
            predicted = self._p + self._q
            gain = predicted / (predicted + self._r)
            before = self._f
            updated = before + gain * (cosine - before)
            self._f = min(1.0, max(-1.0, updated))
            self._p = (1.0 - gain) ** 2 * predicted + gain ** 2 * self._r
            self._t += 1
            weight = (1.0 - self.forget) / (1.0 - self.forget ** (self._t + 1))
            gap = cosine - before
            self._r = max((1.0 - weight) * self._r + weight * gap * gap, self.r_min)
            step = self._f - before
            self._q = max((1.0 - weight) * self._q + weight * step * step, self.q_min)
        self._nu = residual
        if self.part == 'observation':
            if self._seen:
                self._state = self._state + self._f * (observed - self._state)
            else:
                self._state = observed.clone()
                self._seen = True
        self._pending = None

    def diagnostics(self):
        return {'part': self.part, 'updates': self._t, 'f': self._f, 'p': self._p,
                'q': self._q, 'r': self._r}


class HybridGainKalman:
    """m = m_hat + K(z - m_hat), with K updated from the verify residual.

    `hybrid` starts from a calibrated K and from P, Q, R in residual units.
    `scratch` starts at K=0, P=R=1. `frozen` keeps the calibrated K.
    `trajectory` emits ESP. `opened` multiplies the calibrated P and Q by
    `boost` so one prompt can move K. `branch` keeps the ESP mask and stores
    the corrected embedding for an extra depth-1 draft. `replace` uses that
    same embedding, then swaps it into the weakest depth-1 leaf so the node
    count stays equal to the ESP tree. `collect` emits ESP and records the
    slot-0 prior, z, and both commit targets.

    `observe='first'` updates K on the first committed token of the round
    (`hybrid-gain-kalman-v1`). `observe='last'` waits until the round's last
    committed token, the bonus the next-tree mask stands in for
    (`hybrid-gain-bonus-aligned-v1`).
    """
    needs_hidden = False

    def __init__(self, base, mode='hybrid', payload=None):
        if getattr(base, 'needs_hidden', False):
            raise ValueError('hybrid gain wraps input-embedding providers only')
        if mode not in ('hybrid', 'scratch', 'frozen', 'trajectory', 'opened', 'branch', 'replace', 'collect'):
            raise ValueError(
                'hybrid gain mode must be hybrid, scratch, frozen, trajectory, opened, branch, replace or collect')
        self.base = base
        self.num_masks = base.num_masks
        self.mode = mode
        payload = dict(payload or {})
        observe = payload.get('observe', 'first')
        if observe not in ('first', 'last'):
            raise ValueError('hybrid gain observe must be first or last')
        self.observe = observe
        self.protocol = 'hybrid-gain-bonus-aligned-v1' if observe == 'last' else 'hybrid-gain-kalman-v1'
        self._given = payload
        self._boost = 1.0
        self._k0 = self._p0 = self._r0 = self._q0 = None
        self._k = self._p = self._r = self._q = None
        self._z = None
        self._pending = None
        self._probe = None
        self._extra = None
        self._weight_sum = 0.0
        self._weight_n = 0
        self.extra_added = 0
        self.extra_present = 0
        self.max_nodes = 0

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        width = prompt_embeds.shape[-1]
        state = {'device': prompt_embeds.device, 'dtype': torch.float32}
        if self.mode == 'scratch':
            k0 = torch.zeros(width, **state)
            p0 = torch.ones(width, **state)
            r0 = torch.ones(width, **state)
            q0 = torch.full((width,), 0.01, **state)
        elif self.mode in ('trajectory', 'collect'):
            k0 = torch.zeros(width, **state)
            p0 = r0 = q0 = k0.clone()
        else:
            loaded: dict[str, torch.Tensor] = {}
            for name in ('k', 'p', 'r', 'q'):
                value = torch.as_tensor(self._given[name], dtype=torch.float32, device=prompt_embeds.device)
                if value.shape != (width,):
                    raise ValueError(f'hybrid gain {name} must have shape {(width,)}')
                loaded[name] = value
            k0, p0, r0, q0 = loaded['k'], loaded['p'], loaded['r'], loaded['q']
            if self.mode == 'opened':
                boost = self._given.get('boost')
                if isinstance(boost, bool) or not isinstance(boost, (int, float)) or float(boost) < 1.0:
                    raise ValueError('opened hybrid gain needs boost >= 1')
                self._boost = float(boost)
                p0 = p0 * self._boost
                q0 = q0 * self._boost
        self._k0, self._p0, self._r0, self._q0 = k0, p0, r0, q0
        self._k = k0.clone()
        self._p = p0.clone()
        self._r = r0.clone()
        self._q = q0.clone()
        self._z = None
        self._pending = None
        self._first = None
        self._probe = None
        self._extra = None
        self._weight_sum = 0.0
        self._weight_n = 0
        self._pair_m, self._pair_z, self._pair_first, self._pair_last = [], [], [], []

    def masks(self):
        out = self.base.masks().clone()
        prior = out[0].detach().to(dtype=torch.float32).clone()
        self._probe = None
        if self.mode == 'collect':
            self._first = None
            observed = self._z
            if observed is None:
                self._pending = None
            else:
                gain = self._k
                assert gain is not None
                self._pending = (prior, observed.detach().clone(), gain.clone())
            return out
        observed = self._z
        if self.mode != 'trajectory' and observed is not None:
            gain = self._k
            assert gain is not None
            correction = gain * (observed - prior)
            corrected = prior + correction
            if self.mode in ('branch', 'replace'):
                self._probe = corrected.to(dtype=out.dtype)
            else:
                out[0] = corrected.to(dtype=out.dtype)
                if self.mode != 'frozen':
                    self._pending = (prior, observed.detach().clone(), gain.clone())
        return out

    def _apply(self, observed):
        pending = self._pending
        assert pending is not None
        prior, zed, gain = pending
        observed = observed.detach().to(dtype=torch.float32, device=gain.device).reshape(-1)
        regressor = zed - prior
        innov = (observed - prior) - gain * regressor
        variance, process, noise = self._p, self._q, self._r
        assert variance is not None and process is not None and noise is not None
        predicted = variance + process
        scale = noise + predicted * regressor * regressor
        step = predicted * regressor / scale
        self._weight_sum += float((step * regressor).mean())
        self._weight_n += 1
        self._k = gain + step * innov
        self._p = (1.0 - step * regressor) ** 2 * predicted + step ** 2 * noise
        self._pending = None

    def on_token_committed(self, new_embed, prev_embed):
        self.base.on_token_committed(new_embed, prev_embed)
        observed = new_embed.detach().to(dtype=torch.float32).reshape(-1)
        if self._pending is not None and self.mode == 'collect' and self._first is None:
            self._first = observed.clone()
        elif self._pending is not None and self.observe == 'first' and self.mode not in ('frozen', 'trajectory', 'branch', 'replace', 'collect'):
            self._apply(observed)
        if self.mode != 'trajectory':
            gain = self._k
            assert gain is not None
            self._z = observed.to(device=gain.device)

    def finish_round(self):
        """Close the round. `observe='last'` learns from the bonus token."""
        if self._pending is None:
            return
        if self.mode == 'collect':
            first = self._first
            last = self._z
            if first is None or last is None:
                return
            pending = self._pending
            assert pending is not None
            prior, zed, _gain = pending
            self._pair_m.append(prior.detach().cpu())
            self._pair_z.append(zed.detach().cpu())
            self._pair_first.append(first.detach().cpu())
            self._pair_last.append(last.detach().to(dtype=torch.float32).cpu())
            self._pending = None
            self._first = None
            return
        if self.observe == 'last' and self.mode not in ('frozen', 'trajectory', 'branch', 'replace'):
            last = self._z
            assert last is not None
            self._apply(last)

    def calibration_rows(self):
        if not self._pair_m:
            raise ValueError('hybrid gain has no calibration rows')
        return {'m': torch.stack(self._pair_m), 'z': torch.stack(self._pair_z),
                'first': torch.stack(self._pair_first), 'last': torch.stack(self._pair_last)}

    def consume_probe(self):
        embed = self._probe
        self._probe = None
        return embed

    def note_extra(self, token):
        self._extra = int(token)

    def take_extra_branch(self):
        token = self._extra
        self._extra = None
        return token

    @property
    def extra_edit(self):
        return 'replace' if self.mode == 'replace' else 'append'

    def record_branch(self, added, num_nodes=None):
        if added:
            self.extra_added += 1
        else:
            self.extra_present += 1
        if num_nodes is not None:
            self.max_nodes = max(self.max_nodes, int(num_nodes))

    def diagnostics(self):
        gain, variance, noise, process, gain0 = self._k, self._p, self._r, self._q, self._k0
        assert gain is not None and variance is not None and noise is not None and process is not None and gain0 is not None
        delta = (gain - gain0).abs()
        ratio = variance / noise.clamp_min(1e-12)
        return {'mode': self.mode, 'observe': self.observe, 'protocol': self.protocol,
                'k_mean': float(gain.mean()), 'p_mean': float(variance.mean()),
                'r_mean': float(noise.mean()), 'q_mean': float(process.mean()),
                'k_abs_delta': float(delta.mean()), 'p_over_r': float(ratio.mean()),
                'weight_mean': None if self._weight_n == 0 else self._weight_sum / self._weight_n,
                'extra_added': self.extra_added, 'extra_present': self.extra_present,
                'max_nodes': self.max_nodes}


class OutputHiddenHybridKalman:
    """Diagonal Kalman correction on the output-hidden trajectory.

    The correction is applied in FP32 before the norm map. `hybrid` starts
    from a calibrated prior and multiplies P and Q by `boost`. `scratch`
    starts at K=0. `frozen` keeps K. `trajectory` and `collect` emit the
    uncorrected trajectory; `collect` records root pairs for calibration.
    """

    needs_hidden = True
    protocol = 'output-hidden-hybrid-kalman-v1'

    def __init__(self, base, mode='hybrid', payload=None):
        if not isinstance(base, OutputHiddenMaskProvider):
            raise ValueError('output hidden kalman wraps an output hidden trajectory')
        if mode not in ('hybrid', 'scratch', 'frozen', 'trajectory', 'collect'):
            raise ValueError('output kalman mode must be hybrid, scratch, frozen, trajectory or collect')
        self.base = base
        self.num_masks = base.num_masks
        self.mode = mode
        self._given = dict(payload or {})
        self._boost = 1.0
        self._k0 = self._p0 = self._r0 = self._q0 = None
        self._k = self._p = self._r = self._q = None
        self._reset_counts()

    @property
    def scale(self):
        return self.base.scale

    def _reset_counts(self):
        self._current = []
        self._wait = None
        self._weight_sum = 0.0
        self._weight_n = 0
        self._root_err = self._root_base = 0.0
        self._deep_err = self._deep_base = 0.0
        self._root_n = self._deep_n = 0
        self._pair_h, self._pair_z, self._pair_y = [], [], []

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        width = prompt_embeds.shape[-1]
        state = {'device': prompt_embeds.device, 'dtype': torch.float32}
        if self.mode in ('scratch', 'trajectory', 'collect'):
            k0 = torch.zeros(width, **state)
            p0 = torch.ones(width, **state)
            r0 = torch.ones(width, **state)
            q0 = torch.full((width,), 0.01, **state)
        else:
            loaded: dict[str, torch.Tensor] = {}
            for name in ('k', 'p', 'r', 'q'):
                value = torch.as_tensor(self._given[name], dtype=torch.float32, device=prompt_embeds.device)
                if value.shape != (width,) or not torch.isfinite(value).all():
                    raise ValueError(f'output kalman {name} must be a finite vector of width {width}')
                loaded[name] = value
            k0, p0, r0, q0 = loaded['k'], loaded['p'], loaded['r'], loaded['q']
            if self.mode == 'hybrid':
                boost = self._given.get('boost')
                if isinstance(boost, bool) or not isinstance(boost, (int, float)) or float(boost) < 1.0:
                    raise ValueError('hybrid output kalman needs boost >= 1')
                self._boost = float(boost)
                p0 = p0 * self._boost
                q0 = q0 * self._boost
        self._k0, self._p0, self._r0, self._q0 = k0, p0, r0, q0
        self._k = k0.clone()
        self._p = p0.clone()
        self._r = r0.clone()
        self._q = q0.clone()
        self._reset_counts()

    def masks(self):
        return self.base.masks()

    def init_hidden(self, prompt_hidden):
        self.base.init_hidden(prompt_hidden)

    def observe_committed(self, hidden):
        self.base.observe_committed(hidden)

    def masks_per_owner(self, proxy_hiddens):
        slots, observations = self.base.trajectory(proxy_hiddens)
        corrected = slots
        gain = self._k
        assert gain is not None
        if self.mode in ('hybrid', 'scratch', 'frozen'):
            corrected = slots + (observations.unsqueeze(1) - slots) * gain.view(1, 1, -1)
        self._current = []
        for owner in range(corrected.shape[0]):
            self._current.append({
                'h': slots[owner, 0].detach().clone(),
                'z': observations[owner].detach().clone(),
                'c': corrected[owner, 0].detach().clone(),
                'k': gain.detach().clone(),
            })
        return self.base._map_hidden_slots(corrected)

    def note_block_hiddens(self, root_hidden, accepted_hiddens):
        root = root_hidden.detach().float().reshape(-1)
        accepted = [hidden.detach().float().reshape(-1) for hidden in accepted_hiddens]
        if self._wait is not None:
            kind, prediction = self._wait
            self._wait = None
            self._score(kind, prediction, root)
        current = self._current
        if not current:
            return
        if accepted:
            self._score('root', current[0], accepted[0])
        else:
            self._wait = ('root', current[0])
        reached = len(accepted)
        for owner in range(1, len(current)):
            if owner < reached:
                self._score('deep', current[owner], accepted[owner])
            elif owner == reached:
                self._wait = ('deep', current[owner])
            else:
                break

    def _score(self, kind, prediction, target):
        if target.shape != prediction['h'].shape or not torch.isfinite(target).all():
            raise ValueError('output hidden target must be a finite vector of the hidden width')
        base = float((prediction['h'] - target).pow(2).mean())
        error = float((prediction['c'] - target).pow(2).mean())
        if kind == 'root':
            self._root_base += base
            self._root_err += error
            self._root_n += 1
            if self.mode == 'collect':
                self._pair_h.append(prediction['h'].detach().cpu())
                self._pair_z.append(prediction['z'].detach().cpu())
                self._pair_y.append(target.detach().cpu())
            if self.mode in ('hybrid', 'scratch'):
                self._update(prediction, target)
        else:
            self._deep_base += base
            self._deep_err += error
            self._deep_n += 1

    def _update(self, prediction, target):
        gain = prediction['k']
        regressor = prediction['z'] - prediction['h']
        innov = (target - prediction['h']) - gain * regressor
        variance, process, noise = self._p, self._q, self._r
        assert variance is not None and process is not None and noise is not None
        predicted = variance + process
        scale = noise + predicted * regressor * regressor
        step = predicted * regressor / scale
        self._weight_sum += float((step * regressor).mean())
        self._weight_n += 1
        self._k = gain + step * innov
        self._p = (1.0 - step * regressor) ** 2 * predicted + step ** 2 * noise

    def calibration_rows(self):
        if not self._pair_h:
            raise ValueError('output kalman has no calibration rows')
        return {'h': torch.stack(self._pair_h), 'z': torch.stack(self._pair_z), 'y': torch.stack(self._pair_y)}

    def diagnostics(self):
        gain, variance, noise, process, gain0 = self._k, self._p, self._r, self._q, self._k0
        assert gain is not None and variance is not None and noise is not None and process is not None and gain0 is not None
        delta = (gain - gain0).abs()
        ratio = variance / noise.clamp_min(1e-12)
        def mean(total, count):
            return None if count == 0 else total / count
        return {'mode': self.mode, 'protocol': self.protocol, 'boost': self._boost,
                'k_mean': float(gain.mean()), 'p_mean': float(variance.mean()),
                'r_mean': float(noise.mean()), 'q_mean': float(process.mean()),
                'k_abs_delta': float(delta.mean()), 'p_over_r': float(ratio.mean()),
                'weight_mean': None if self._weight_n == 0 else self._weight_sum / self._weight_n,
                'root_error': mean(self._root_err, self._root_n),
                'root_base_error': mean(self._root_base, self._root_n),
                'deep_error': mean(self._deep_err, self._deep_n),
                'deep_base_error': mean(self._deep_base, self._deep_n),
                'root_n': self._root_n, 'deep_n': self._deep_n}


class PreforwardSignalCollector:
    """Record a before-forward bundle and the bonus embedding. Masks stay ESP.

    The bundle is the previous forward's single mask slot: full-softmax expected
    embedding, top-4 renormalized expected embedding, final-norm hidden, entropy
    and top-4 mass. `finish_round` pairs it with this round's first and last
    committed embeddings. Slot 0 is not modified.
    """
    needs_hidden = False
    TOPK = 4

    def __init__(self, base, embedding_table):
        if getattr(base, 'needs_hidden', False):
            raise ValueError('preforward signal wraps input-embedding providers only')
        if int(base.num_masks) != 1:
            raise ValueError('preforward signal is specified for one mask slot')
        self.base = base
        self.num_masks = base.num_masks
        self.embedding_table = embedding_table
        self._bundle = None
        self._pending = None
        self._z = None
        self._first = None
        self._commits = 0
        self._prev_accepted = None
        self._rows = []

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        self._bundle = None
        self._pending = None
        self._z = None
        self._first = None
        self._commits = 0
        self._prev_accepted = None
        self._rows = []

    def masks(self):
        out = self.base.masks().clone()
        self._pending = None
        self._first = None
        self._commits = 0
        if self._bundle is None or self._z is None:
            return out
        if self._bundle.get('hidden') is None:
            raise RuntimeError('preforward signal is missing the mask hidden')
        prior = out[0].detach().to(dtype=torch.float32)
        self._pending = {
            'm': prior.cpu(),
            'z': self._z.detach().to(dtype=torch.float32).cpu(),
            'expected': self._bundle['expected'].detach().to(dtype=torch.float32).cpu(),
            'topk': self._bundle['topk'].detach().to(dtype=torch.float32).cpu(),
            'hidden': self._bundle['hidden'].detach().to(dtype=torch.float32).cpu(),
            'entropy': self._bundle['entropy'],
            'top_mass': self._bundle['top_mass'],
            'prev_accepted': -1 if self._prev_accepted is None else int(self._prev_accepted),
        }
        return out

    def observe_mask_logits(self, mask_logits):
        if mask_logits.shape[0] != 1:
            raise ValueError('preforward signal needs one logit row')
        row = mask_logits[0].float()
        if row.shape[0] != self.embedding_table.shape[0]:
            raise ValueError('preforward signal logits and the embedding table disagree on vocab size')
        if not torch.isfinite(row).any() or torch.isnan(row).any():
            raise ValueError('preforward signal logits need a finite candidate')
        probs = torch.softmax(row, dim=-1)
        table = self.embedding_table.detach().float()
        mass, index = torch.topk(probs, self.TOPK)
        self._bundle = {
            'expected': probs @ table,
            'topk': (mass / mass.sum().clamp_min(1e-12)) @ table[index],
            'hidden': None,
            'entropy': float(-(probs * probs.clamp_min(1e-12).log()).sum()),
            'top_mass': float(mass.sum()),
        }

    def observe_mask_hidden(self, hidden):
        if self._bundle is None:
            raise ValueError('preforward signal hidden arrived before logits')
        row = hidden[0].detach().float().reshape(-1)
        if row.shape[0] != self.embedding_table.shape[1] or not torch.isfinite(row).all():
            raise ValueError('preforward signal hidden must be one finite vector of embedding width')
        self._bundle['hidden'] = row

    def on_token_committed(self, new_embed, prev_embed):
        self.base.on_token_committed(new_embed, prev_embed)
        observed = new_embed.detach().to(dtype=torch.float32).reshape(-1)
        if self._pending is not None and self._first is None:
            self._first = observed.clone()
        self._commits += 1
        self._z = observed

    def finish_round(self):
        accepted = None if self._commits < 1 else self._commits - 1
        if self._pending is not None:
            if self._first is None or self._z is None or accepted is None:
                raise RuntimeError('preforward signal closed a round without a commit')
            row = dict(self._pending)
            row['first'] = self._first.detach().to(dtype=torch.float32).cpu()
            row['last'] = self._z.detach().to(dtype=torch.float32).cpu()
            row['accepted'] = int(accepted)
            self._rows.append(row)
        self._prev_accepted = accepted
        self._pending = None
        self._first = None

    def rows(self):
        return list(self._rows)
