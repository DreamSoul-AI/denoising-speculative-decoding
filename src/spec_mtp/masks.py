"""Committed mask state and pure block-local extrapolation."""


from typing import Any
import torch


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

    def masks_per_owner(self, proxy_hiddens):
        """Predict depths 0..k; the decoder shares each depth across tree siblings."""
        # Proxies are at root..penultimate draft positions. The last owner
        # has no observation; simulate only, without mutating committed state.
        if proxy_hiddens.shape != (self.num_masks, self.anchor.numel()):
            raise ValueError('output proxy shape mismatch')
        anchor, velocity = self.anchor.clone(), self.velocity.clone()
        owners = []
        for owner in range(self.num_masks + 1):
            if owner < self.num_masks:
                y = proxy_hiddens[owner].detach().float()
                if self.order:
                    velocity = self.beta_prime * velocity + (1 - self.beta_prime) * (y - anchor)
                anchor = y
            else:
                anchor = anchor + velocity
            slots = torch.stack([anchor + slot * velocity for slot in range(self.num_masks)])
            norms = slots.norm(dim=-1, keepdim=True)
            if not torch.isfinite(slots).all() or not torch.isfinite(self.scale) or self.scale <= 0 or (norms <= 0).any():
                raise ValueError('output mask mapping requires finite nonzero norms')
            owners.append(self.scale * slots / norms)
        return torch.stack(owners)




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
        m[d, j] = mhat_j + g * (z_{d+j} - mhat_j)   if d + j <= k, else mhat_j.
    The observation lives for one round and is never written into the
    trajectory state, so g = 0 is the wrapped provider token for token.

    condition='rejection' (`trajectory-observation-rejection-v1`): an owner's
    mask logits seed the next tree only if it is the deepest accepted node, in
    which case every child token was rejected. Slot 1 of an owner with
    children therefore uses z_{d+1} conditioned on "not a child token":
        z = (z_{d+1} - sum_c p_c e_c) / (1 - sum_c p_c),
    falling back to mhat_1 when the children hold all the mass.

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
        row = self._logits[depth]
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
        return self._level.unsqueeze(0).expand(self.num_masks, self._level.shape[-1]).clone()

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
        if kind not in ('scalar', 'diagonal', 'affine', 'reject_token'):
            raise ValueError('correction kind must be scalar, diagonal, affine or reject_token')
        self.base = base
        self.num_masks = base.num_masks
        self.kind = kind
        self.online = bool(online)
        if self.online and kind not in ('scalar', 'affine'):
            raise ValueError('online correction is scalar or affine')
        payload = dict(payload or {})
        self._k = self._num = self._den = None
        self._diag = self._A = self._b = None
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
            if self.online:
                self._gram0 = torch.tensor(payload['gram'], dtype=torch.float64)
                self._rhs0 = torch.tensor(payload['rhs'], dtype=torch.float64)
                self._A0 = self._A.clone()
                self._b0 = self._b.clone()
                self._gram = self._gram0.clone()
                self._rhs = self._rhs0.clone()
        self._z = None
        self._round_commits = 0
        self._masks_calls = 0
        self._pending = None

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.base.init_from_prompt(prompt_embeds, generator=generator)
        self._z = None
        self._round_commits = 0
        self._masks_calls = 0
        self._pending = None
        if self._gram0 is not None:
            self._gram = self._gram0.clone()
            self._rhs = self._rhs0.clone()
            self._A = self._A0.clone()
            self._b = self._b0.clone()

    def on_token_committed(self, new_embed, prev_embed):
        if self._pending is not None and self.online and self._round_commits == 0:
            mhat, zed = self._pending
            error = (new_embed - mhat).float()
            delta = (zed - mhat).float()
            if self.kind == 'scalar':
                self._num += float((error * delta).sum())
                self._den += float((delta * delta).sum())
                self._k = self._num / max(self._den, 1e-12)
            else:
                row = torch.cat([delta.double(), delta.new_ones(1).double()])
                self._gram = self._gram + row[:, None] * row[None, :]
                self._rhs = self._rhs + row[:, None] * error.double()[None, :]
                eye = torch.eye(self._gram.shape[0], dtype=torch.float64)
                solution = torch.linalg.solve(self._gram + 1e-8 * eye, self._rhs)
                self._A = solution[:-1].T.float()
                self._b = solution[-1].float()
            self._pending = None
        self.base.on_token_committed(new_embed, prev_embed)
        self._z = new_embed.detach()
        self._round_commits += 1

    def _corrected(self, mhat):
        delta = self._z.to(device=mhat.device, dtype=mhat.dtype) - mhat
        if self.kind == 'scalar':
            return mhat + delta * mhat.new_tensor(self._k)
        if self.kind == 'diagonal':
            return mhat + delta * self._diag.to(device=mhat.device, dtype=mhat.dtype)
        return mhat + delta @ self._A.to(device=mhat.device, dtype=mhat.dtype).T + self._b.to(device=mhat.device, dtype=mhat.dtype)

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


def fit_mask_correction(m_hat, z, target, kind):
    """Least squares for m = m_hat + K(z - m_hat), or plus a bias when affine."""
    if kind not in ('scalar', 'diagonal', 'affine'):
        raise ValueError('fit kind must be scalar, diagonal or affine')
    delta = (z - m_hat).double()
    error = (target - m_hat).double()
    if delta.ndim != 2 or error.shape != delta.shape or delta.shape[0] < 2:
        raise ValueError('correction fit needs at least two rows')
    if not torch.isfinite(delta).all() or not torch.isfinite(error).all():
        raise ValueError('correction fit inputs must be finite')
    if kind == 'scalar':
        num = (error * delta).sum()
        den = (delta * delta).sum().clamp_min(1e-12)
        return {'k': float(num / den), 'num': float(num), 'den': float(den)}
    if kind == 'diagonal':
        num = (error * delta).sum(dim=0)
        den = (delta * delta).sum(dim=0).clamp_min(1e-12)
        return {'k': (num / den).tolist()}
    design = torch.cat([delta, torch.ones(delta.shape[0], 1, dtype=delta.dtype)], dim=1)
    solution = torch.linalg.lstsq(design, error).solution
    gram = design.T @ design
    rhs = design.T @ error
    return {'A': solution[:-1].T.tolist(), 'b': solution[-1].tolist(),
            'gram': gram.tolist(), 'rhs': rhs.tolist()}


def build_mask_provider(spec_cfg, embedding_table, generator=None):
    observation = spec_cfg.get('observation')
    bonus = spec_cfg.get('bonus_slot')
    correction = spec_cfg.get('correction')
    if correction is not None and (observation is not None or bonus is not None or str(spec_cfg.get('method', 'esp')).lower() == 'kalman'):
        raise ValueError('correction does not combine with observation, bonus_slot or kalman')
    if str(spec_cfg.get('method', 'esp')).lower() == 'kalman' and (observation is not None or bonus is not None):
        raise ValueError('kalman does not combine with observation or bonus_slot')
    if correction is not None:
        if not isinstance(correction, dict) or correction.get('kind') not in ('scalar', 'diagonal', 'affine', 'reject_token'):
            raise ValueError('spec.correction kind must be scalar, diagonal, affine or reject_token')
        if not set(correction) <= {'kind', 'online', 'k', 'num', 'den', 'A', 'b', 'gram', 'rhs'}:
            raise ValueError('spec.correction has an unknown field')
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'correction'}
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return VerifyCorrectionMaskProvider(base, correction['kind'], correction, online=bool(correction.get('online', False)))
    if observation is not None and bonus is not None:
        raise ValueError('bonus_slot does not combine with observation')
    if bonus is not None:
        if not isinstance(bonus, dict) or set(bonus) != {'mode'} or bonus.get('mode') not in ('hard', 'topk'):
            raise ValueError('bonus_slot accepts only mode hard or topk')
        widths = list((spec_cfg.get('tree') or {}).get('branches') or [])
        if bonus['mode'] == 'topk' and (not widths or any(not isinstance(w, int) or isinstance(w, bool) or w <= 0 for w in widths)):
            raise ValueError('topk bonus slot needs positive integer tree branch widths')
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'bonus_slot'}
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return BonusSlotMaskProvider(base, embedding_table, bonus['mode'], widths)
    if observation is not None:
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'observation'}
        if 'gain' not in observation or not set(observation) <= {'gain', 'condition', 'form'}:
            raise ValueError('spec.observation accepts only gain, condition and form')
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return ObservationBlendMaskProvider(base, embedding_table, observation['gain'],
                                            observation.get('condition', 'none'), observation.get('form', 'mean'))
    method = str(spec_cfg.get('method', 'esp')).lower()
    num_masks = int(spec_cfg.get('num_masks', 1))
    if method == 'esp':
        mask_cfg = spec_cfg.get('mask', {})
        return ESPMaskProvider(
            num_masks=num_masks,
            embedding_table=embedding_table,
            init=mask_cfg.get('init', 'mean_prompt'),
            lam=mask_cfg.get('lam', 0.1),
            update=mask_cfg.get('update', True),
            offset_scale=mask_cfg.get('offset_scale', 5.0),
            generator=generator,
        )
    if method == 'kalman':
        cfg = dict(spec_cfg.get('kalman') or {})
        if not set(cfg) <= {'mode', 'forget', 'r_min', 'q_min', 'q_scale'}:
            raise ValueError('spec.kalman accepts only mode, forget, r_min, q_min and q_scale')
        return KalmanMaskProvider(
            num_masks=num_masks, embedding_table=embedding_table, generator=generator,
            mode=cfg.get('mode', 'online'), forget=cfg.get('forget', 0.95),
            r_min=cfg.get('r_min', 1e-6), q_min=cfg.get('q_min', 1e-8), q_scale=cfg.get('q_scale', 0.01))
    if method == 'ema_velocity':
        ema_cfg = spec_cfg.get('ema', {})
        return EMAVelocityMaskProvider(
            num_masks=num_masks,
            embedding_table=embedding_table,
            beta=ema_cfg.get('beta', 0.9),
            step_scale=ema_cfg.get('step_scale', 1.0),
            normalize=ema_cfg.get('normalize', False),
            extrapolate_ema=ema_cfg.get('extrapolate_ema', False),
            update=ema_cfg.get('update', True),
            level=ema_cfg.get('level', 'last'),
            lam=ema_cfg.get('lam', 0.1),
            generator=generator,
        )
    if method == 'output_hidden':


        return OutputHiddenMaskProvider(num_masks, embedding_table, **dict(spec_cfg.get('output', {})))
    raise ValueError(f'Unsupported spec.method: {method}')
