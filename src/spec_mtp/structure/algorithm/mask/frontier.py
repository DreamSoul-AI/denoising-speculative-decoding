"""Jason 534a79 frontier affine correction, using the shared decoder.

z_post=z_probe+[E(bonus)-placeholder;1]K. Delayed next-root labels
the same position; no unverified token is used as a true hidden label.
"""

import torch

from spec_mtp.structure.algorithm.mask.providers import ESPMaskProvider


class FrontierMaskProvider(ESPMaskProvider):
    needs_final_hidden = True

    def __init__(self, embedding_table, output_head, gain=None):
        super().__init__(1, embedding_table, lam=.1)
        self.head = output_head
        self.gain = gain

    def init_from_prompt(self, prompt_embeds, generator=None):
        super().init_from_prompt(prompt_embeds, generator)
        self.pending = None
        self.pairs = []
        self.pair_positions = []

    def label_root(self, hidden, position=None):
        if self.pending is not None:
            probe, feature = self.pending
            if position != self.pending_position:
                raise ValueError('frontier label is not at its pending probe position')
            self.pairs.append((probe.cpu(), hidden.detach().float().cpu(), feature.cpu()))
            self.pair_positions.append((self.pending_position, position))
            self.pending = None

    def frontier_logits(self, hidden, placeholder, bonus_embed, logits, position=None):
        probe = hidden.detach().float()
        feature = torch.cat([bonus_embed.detach().float() - placeholder.float(), probe.new_ones(1)])
        self.pending = (probe.clone(), feature)
        self.pending_position = position
        if self.gain is None:
            return logits
        if self.gain.shape != (probe.numel()+1, probe.numel()) or not torch.isfinite(self.gain).all():
            raise ValueError('invalid frontier gain')
        corrected = probe + feature @ self.gain.to(device=probe.device, dtype=torch.float32)
        return (corrected.to(self.head.dtype) @ self.head.T).to(logits.dtype)
