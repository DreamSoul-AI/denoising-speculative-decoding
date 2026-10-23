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


def build_mask_provider(spec_cfg, embedding_table, generator=None):
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
