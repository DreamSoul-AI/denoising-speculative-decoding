"""Reproduce ESP Figure 2 / Section 3.2: mask-vs-true-token hidden-state
cosine similarity across decoder layers, split by acceptance.

Protocol (Appendix A, done in hindsight on the model's own greedy rollout):
for each context x_{1:s} along the rollout, compare per-layer hidden states of

    x_{s+1}  (the true next token)      and      m  (mask in the x_{s+1} slot)

both of which predict position s+2. A step counts as *accepted* when the
mask's Top-1 prediction equals the true x_{s+2}. The paper uses 100
Dolly-Databricks creative-writing samples with LLaMA3.2-3B-Instruct and finds
accepted steps reach ~0.45 cosine in later layers vs ~0.35 for rejected.
"""

import torch

from studies._shared.structure import csv_logger
from studies._shared.structure.inputs import legacy_inputs, select_prompts
from spec_mtp.decode import ar_generate
from spec_mtp.masks import build_mask_provider


@torch.no_grad()
def _hidden_states_of_last(model, embeds, device):
    """Per-layer hidden states of the last position; embeds [n, d]."""
    position_ids = torch.arange(embeds.size(0), device=device).unsqueeze(0)
    out = model(embeds.unsqueeze(0), position_ids, attention_mask_4d=None,
                past_key_values=None, output_hidden_states=True)
    # hidden_states: (embeddings, layer_1, ..., layer_L)
    layers = [h[0, -1].float() for h in out.hidden_states[1:]]
    return layers, out.logits[0, -1]


def evaluate(model, prompts, eos_ids, spec_cfg, eval_cfg, save_dir=None, *, aggregate=True):
    prompts, eos_ids = select_prompts(prompts, eos_ids, eval_cfg)
    device = model.device
    max_steps = int(eval_cfg.get('probe_steps_per_prompt', 16))

    num_layers = model.config.num_hidden_layers
    sums = {'accept': torch.zeros(num_layers), 'reject': torch.zeros(num_layers)}
    counts = {'accept': 0, 'reject': 0}

    for item in prompts:
        prompt_ids = item['input_ids'].to(device)
        # Greedy rollout supplies the "true" continuation for hindsight probing.
        rollout, _ = ar_generate(model, prompt_ids, eval_cfg, eos_ids=eos_ids)
        rollout = rollout[0]
        t = prompt_ids.size(1)

        provider = build_mask_provider(spec_cfg, model.embedding_table)
        steps = min(max_steps, rollout.size(0) - t - 2)
        for s in range(steps):
            context = rollout[:t + s]                     # x_{1:s'}
            true_next = rollout[t + s].item()             # x_{s'+1}
            true_next_next = rollout[t + s + 1].item()    # x_{s'+2}

            context_embeds = model.embed(context.unsqueeze(0))[0]
            provider.init_from_prompt(context_embeds)
            mask = provider.masks()[:1].to(model.dtype)   # single mask token

            true_embed = model.embed(
                torch.tensor([[true_next]], device=device))[0]
            h_true, _ = _hidden_states_of_last(
                model, torch.cat([context_embeds, true_embed], dim=0), device)
            h_mask, mask_logits = _hidden_states_of_last(
                model, torch.cat([context_embeds, mask], dim=0), device)

            accepted = int(mask_logits.argmax().item()) == true_next_next
            key = 'accept' if accepted else 'reject'
            for layer in range(num_layers):
                sums[key][layer] += torch.nn.functional.cosine_similarity(
                    h_true[layer], h_mask[layer], dim=0).item()
            counts[key] += 1

    if not aggregate:
        return {'sums': {k: v.tolist() for k, v in sums.items()}, 'counts': counts,
                'n_accepted': counts['accept'], 'n_rejected': counts['reject']}
    logger = csv_logger.Logger(save_dir, filename='alignment_metrics.csv')
    for layer in range(num_layers):
        logger.log({
            'layer': layer + 1,
            'cos_accepted': (sums['accept'][layer] / max(counts['accept'], 1)).item(),
            'cos_rejected': (sums['reject'][layer] / max(counts['reject'], 1)).item(),
            'n_accepted': counts['accept'],
            'n_rejected': counts['reject'],
        })
    return {'n_accepted': counts['accept'], 'n_rejected': counts['reject']}


def test(cfg, model=None):
    model, prompts, eos_ids, _device = legacy_inputs(cfg, model)
    return evaluate(model, prompts, eos_ids, cfg.spec, cfg.eval,
                    cfg.run.get('save_dir', None))
