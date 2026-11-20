"""Input-embedding trajectories indexed by committed tokens, never by forwards.

State arithmetic is float32; the shared 2x2 covariance is float64 on CPU.
Prediction reads are pure. Draft embeddings are deliberately not observations.
"""

import math

import torch


def covariance_step(covariance, process_noise, measurement_noise):
    """One unit-step F=[[1,1],[0,1]], H=[1,0], with Joseph correction."""
    transition = covariance.new_tensor([[1., 1.], [0., 1.]])
    prior = transition @ covariance @ transition.T + process_noise
    gain = prior[:, 0] / (prior[0, 0] + measurement_noise)
    residual_map = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    residual_map[:, 0] -= gain
    posterior = residual_map @ prior @ residual_map.T + measurement_noise * torch.outer(gain, gain)
    return (posterior + posterior.T) * 0.5, gain


def steady_state_gain(covariance, process_noise, measurement_noise, *, tolerance=1e-10, max_steps=10000):
    """Return a converged posterior/gain; failure is explicit, never a fallback."""
    current = covariance.clone()
    for _ in range(max_steps):
        posterior, gain = covariance_step(current, process_noise, measurement_noise)
        if torch.allclose(posterior, current, atol=tolerance, rtol=tolerance):
            return posterior, gain
        current = posterior
    raise ValueError('input trajectory steady-state covariance did not converge')


class InputTrajectoryMaskProvider:
    """Controlled zero/difference/Holt/Kalman input route, without K calibration.

    alpha weights the new level; beta retains the old difference velocity.
    Covariance ratios use common normalized units (shared across features).
    Fixed Q/R gains depend on step/covariance, not on residual magnitudes.
    """

    needs_hidden = False
    protocol = 'input-trajectory-v1'

    def __init__(self, num_masks, *, kind='order0', alpha=0.1, beta=0.9,
                 anchor='level', horizon='owner_depth', q_level=0.01,
                 q_velocity=0.01, r=1.0, p0=1.0):
        if type(num_masks) is not int or num_masks < 1:
            raise ValueError('input trajectory num_masks must be a positive integer')
        if kind not in ('order0', 'difference', 'holt', 'kalman', 'fixed_gain'):
            raise ValueError('unknown input trajectory kind')
        if anchor not in ('level', 'last') or (anchor != 'level' and kind != 'difference'):
            raise ValueError('last anchor is supported only for difference trajectories')
        if horizon not in ('owner_depth', 'slot'):
            raise ValueError('input trajectory horizon must be owner_depth or slot')
        numbers = [alpha, beta, q_level, q_velocity, r, p0]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in numbers):
            raise ValueError('input trajectory parameters must be finite numbers')
        if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise ValueError('input trajectory requires alpha and beta in [0, 1]')
        if min(q_level, q_velocity, p0) < 0 or r <= 0:
            raise ValueError('input trajectory requires nonnegative Q/P0 and positive R')
        self.num_masks = num_masks
        self.kind, self.alpha, self.beta = kind, float(alpha), float(beta)
        self.anchor, self.horizon = anchor, horizon
        self.q_level, self.q_velocity, self.r, self.p0 = map(float, (q_level, q_velocity, r, p0))
        self.M = self.V = self.last_observation = None
        self.covariance = self.gain = None
        self.committed_steps = 0

    def init_from_prompt(self, prompt_embeds, generator=None):
        if (prompt_embeds.ndim != 2 or min(prompt_embeds.shape) < 1
                or not torch.isfinite(prompt_embeds).all()):
            raise ValueError('input trajectory requires a finite nonempty prompt embedding matrix')
        observations = prompt_embeds.detach().float()
        mean = observations.mean(0)
        if not torch.isfinite(observations).all() or not torch.isfinite(mean).all():
            raise ValueError('input trajectory prompt overflows float32 state')
        self.M = mean
        self.V = torch.zeros_like(self.M)
        self.last_observation = observations[-1].clone()
        self.committed_steps = 0
        self.covariance = self.gain = None
        if self.kind in ('kalman', 'fixed_gain'):
            self.process_noise = torch.diag(torch.tensor([self.q_level, self.q_velocity], dtype=torch.float64))
            # P, Q and R are in common scale-free units: a shared positive
            # fit scale s^2 cancels from G. No observation-noise fit is claimed.
            self.measurement_noise = self.r
            self.covariance = torch.eye(2, dtype=torch.float64) * self.p0
            if self.kind == 'fixed_gain':
                self.covariance, self.gain = steady_state_gain(
                    self.covariance, self.process_noise, self.measurement_noise)

    def masks(self):
        """Prefill slots, at horizons 1..k from the prompt's last position."""
        return self._predict([0])[0]

    def masks_for_tree(self, tree):
        """Root and each draft owner, in exactly the decoder's flattened order."""
        return self._predict([0] + tree.depths)

    def _predict(self, depths):
        if self.M is None:
            raise RuntimeError('input trajectory must be initialized before prediction')
        slots = torch.arange(1, self.num_masks + 1, dtype=self.M.dtype, device=self.M.device)
        distances = slots.unsqueeze(0).expand(len(depths), -1)
        if self.horizon == 'owner_depth':
            distances = distances + self.M.new_tensor(depths).unsqueeze(1)
        origin = self.last_observation if self.anchor == 'last' else self.M
        return origin[None, None, :] + distances[:, :, None] * self.V[None, None, :]

    def on_token_committed(self, new_embed, prev_embed):
        """Absorb one retained token, including bonus; prev_embed is API compatibility.

        Use our stored observation for differences, so predictions cannot advance
        the observation index. Calls are owned by the decoder after truncation.
        """
        if self.M is None:
            raise RuntimeError('input trajectory must be initialized before update')
        y = new_embed.detach().to(device=self.M.device, dtype=torch.float32)
        if y.shape != self.M.shape or not torch.isfinite(y).all():
            raise ValueError('input trajectory requires a finite matching observation')
        if self.kind in ('order0', 'difference'):
            next_M = self.M + self.alpha * (y - self.M)
            next_V = self.V
            if self.kind == 'difference':
                next_V = self.beta * self.V + (1 - self.beta) * (y - self.last_observation)
        else:
            prediction = self.M + self.V
            residual = y - prediction
            if self.kind == 'holt':
                level_gain, velocity_gain = self.alpha, self.alpha * (1 - self.beta)
            else:
                if self.kind == 'kalman':
                    covariance, gain = covariance_step(self.covariance, self.process_noise, self.measurement_noise)
                else:
                    covariance, gain = self.covariance, self.gain
                level_gain, velocity_gain = gain.tolist()
            next_M = prediction + level_gain * residual
            next_V = self.V + velocity_gain * residual
        if not torch.isfinite(next_M).all() or not torch.isfinite(next_V).all():
            raise ValueError('nonfinite input trajectory update')
        self.M, self.V = next_M, next_V
        if self.kind in ('kalman', 'fixed_gain'):
            self.covariance, self.gain = covariance, gain
        self.last_observation = y.clone()
        self.committed_steps += 1
