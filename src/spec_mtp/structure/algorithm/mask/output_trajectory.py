"""Positioned committed-hidden trajectory with an untied-safe probability readout."""

import torch

from spec_mtp.structure.algorithm.mask.trajectory import InputTrajectoryMaskProvider


class OutputTrajectoryMaskProvider:
    needs_hidden = True
    positioned_hidden = True
    requires_tied_embeddings = False

    def __init__(self, num_masks, embedding_table, output_head, **state_config):
        if output_head is None or output_head.shape != embedding_table.shape:
            raise ValueError('output trajectory requires matching actual output-head weights')
        self.state = InputTrajectoryMaskProvider(num_masks, **state_config)
        self.num_masks = num_masks
        self.embeddings, self.output_head = embedding_table.detach(), output_head.detach()

    def init_from_prompt(self, prompt_embeds, generator=None):
        self.prefill_masks = prompt_embeds.mean(0).expand(self.num_masks, -1).clone()
        self.observed_position = None
        self.ledger = []

    def masks(self):
        return self.prefill_masks

    def init_hidden(self, prompt_hidden):
        self.state.init_from_prompt(prompt_hidden)
        self.scale = prompt_hidden.detach().float().norm(dim=-1).mean()
        self.observed_position = len(prompt_hidden) - 1
        self.ledger = [{'forward': 0, 'position_start': 0, 'position_end': self.observed_position,
                        'kind': 'prompt_mean_initialization'}]

    def observe_position(self, hidden, position, forward):
        if position != self.observed_position + 1:
            raise ValueError('output hidden observation must be the next real committed position')
        self.state.on_token_committed(hidden, self.state.last_observation)
        self.observed_position = position
        self.ledger.append({'position': position, 'forward': forward})

    def tree_masks(self, tree, root_position):
        state = self.state
        slots = torch.arange(self.num_masks, device=state.M.device, dtype=state.M.dtype)
        # slot index is zero-based here: mask q=root+depth+slot+1 reads h_(q-1).
        positions = state.M.new_tensor([0] + tree.depths)[:, None] + root_position + slots[None, :]
        distances = positions - self.observed_position
        if (distances < 0).any():
            raise ValueError('output prediction precedes its observation')
        origin = state.last_observation if state.anchor == 'last' else state.M
        hidden = origin[None, None, :] + distances[:, :, None] * state.V[None, None, :]
        norms = hidden.norm(dim=-1, keepdim=True)
        if not torch.isfinite(hidden).all() or (norms <= 0).any() or self.scale <= 0:
            raise ValueError('invalid output trajectory readout')
        hidden = self.scale * hidden / norms
        logits = hidden.to(self.output_head.dtype) @ self.output_head.T
        probabilities = torch.softmax(logits.float(), dim=-1)
        result = probabilities.to(self.embeddings.dtype) @ self.embeddings
        if not torch.isfinite(result).all():
            raise ValueError('nonfinite output readout')
        return result
