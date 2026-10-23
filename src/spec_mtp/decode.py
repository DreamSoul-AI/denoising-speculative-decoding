"""ESP-style speculative decoding engine (generation + parallel verification).

One `generate` call reproduces the ESP loop (Fig. 1):

  prefill   prompt + k mask tokens in one causal forward pass. The last
            prompt position yields the next committed token (the first tree
            root); the k mask positions yield the distributions that seed the
            first draft tree. Mask KV entries are evicted afterwards.

  decode    each pass processes one block: [root | draft tree | masks], with
            a tree attention mask and tree position IDs. The same forward
            pass (i) verifies draft tokens by exact match against the base
            model's own prediction chain (lossless, "sample matching"),
            (ii) yields the bonus/correction token, and (iii) produces the
            mask logits — at the masks of the deepest accepted node — that
            seed the next tree. Cache keeps only root + accepted path.

Verification draws one sample per committed token in chain order, so with a
shared torch.Generator the mathematical verifier follows the same sampling
chain. Independent AR equality is audited separately: BF16 forward shapes
can differ numerically. At T0, every committed token records its own forward
argmax. tau = committed tokens (accepted + bonus) per decode call.
"""

from typing import Any, cast
import time
import hashlib
import json
import math

from spec_mtp.contracts import DecodeRequest, DecodeResult, MaskPolicy, HiddenMaskPolicy, ModelSession, ObservationSink, Cancel

import torch

from spec_mtp.tree import build_tree, node_budget_for
from spec_mtp.tree import (DynamicTreeAttentionCache, TreeAttentionCache,
                                          build_block_attention, mask_indices_of)


def _fork_generator(generator: torch.Generator | None) -> torch.Generator | None:
    """Independent Generator with the same current state (None stays None)."""
    if generator is None:
        return None
    fork = torch.Generator(device=generator.device)
    fork.set_state(generator.get_state())
    return fork


def select_token(logits_row, temperature, generator):
    """Greedy at temperature 0, else one multinomial draw ("sample matching")."""
    if temperature <= 0.0:
        return int(logits_row.argmax(dim=-1).item())
    probs = torch.softmax(logits_row.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def tree_probs(mask_logits, temperature):
    """Candidate-ranking distribution per mask; matches the verifier at T>0."""
    if temperature <= 0.0:
        return torch.softmax(mask_logits.float(), dim=-1)
    return torch.softmax(mask_logits.float() / temperature, dim=-1)


class SpecDecoder:

    def __init__(self, model: ModelSession, spec_cfg, eval_cfg, eos_ids=(), generator=None, *,
                 mask_policy: MaskPolicy, sink: ObservationSink | None = None, cancel: Cancel | None = None):
        self.model = model
        self.mask_policy = mask_policy
        self.sink = sink
        self.cancel = cancel or (lambda: False)
        self.was_cancelled = False
        self.committed: list[int] = []
        self.stats: dict[str, Any] = {}
        self.spec_cfg = spec_cfg
        self.eval_cfg = eval_cfg
        self.eos_ids = set(int(e) for e in eos_ids)
        self.generator = generator
        raw_temperature = eval_cfg.get('temperature', 0.0)
        raw_budget = eval_cfg.get('max_new_tokens', 100)
        if type(raw_budget) is not int or raw_budget < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer')
        if isinstance(raw_temperature, bool) or not math.isfinite(raw_temperature) or raw_temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        self.temperature = float(raw_temperature)
        self.max_new_tokens = int(eval_cfg.get('max_new_tokens', 100))
        self.stop_on_eos = bool(eval_cfg.get('stop_on_eos', True))
        raw_masks = spec_cfg.get('num_masks', 1)
        if type(raw_masks) is not int or raw_masks < 1:
            raise ValueError('num_masks must be a positive integer')
        self.num_masks = raw_masks

        self.impl = str(spec_cfg.get('impl', 'efficient')).lower()
        tree_mode = str(spec_cfg.get('tree', {}).get('mode', 'dynamic')).lower()
        if self.impl == 'efficient' and tree_mode != 'static':
            # Appendix E: the cached-mask fast path assumes a fixed tree
            # structure; dynamic trees change shape every step.
            raise ValueError('spec.impl=efficient requires spec.tree.mode=static; '
                             'use spec.impl=naive with dynamic trees')
        if self.impl not in ('naive', 'efficient'):
            raise ValueError(f'Unsupported spec.impl: {self.impl}')
        self.output_hidden = bool(mask_policy.needs_hidden)
        if self.output_hidden and not model.tied_embeddings:
            raise ValueError('output_hidden requires actually tied input/output weights')
        self._attn_cache: Any = TreeAttentionCache() if self.impl == 'efficient' else None
        # Dynamic trees: memoised, vectorised build of the same naive mask (perf only).
        # Static + naive keeps the literal builder for the Table 4 timing ablation.
        self._dynamic_attn: Any = DynamicTreeAttentionCache() if tree_mode == 'dynamic' else None

        # Log the realized block complexity (Eq 7) for bookkeeping.
        if tree_mode == 'static':
            widths = [int(b) for b in spec_cfg['tree']['branches']]
            self.nominal_block_complexity = (self.num_masks + 1) * (1 + sum(widths))
        else:
            budget = node_budget_for(int(spec_cfg['block_complexity']), self.num_masks)
            self.nominal_block_complexity = (self.num_masks + 1) * (1 + budget)

    def _poll_cancel(self):
        if not self.was_cancelled:
            self.was_cancelled = bool(self.cancel())
        return self.was_cancelled

    def _emit(self, observation):
        if self.sink is not None:
            self.sink.write(observation)

    def _is_eos(self, token):
        return self.stop_on_eos and token in self.eos_ids

    @torch.no_grad()
    def generate(self, prompt_ids):
        """prompt_ids: [1, t]. Returns (output_ids [1, t+n], stats dict)."""
        model = self.model
        _validate_prompt(prompt_ids, model.vocab_size)
        device, dtype = model.device, model.dtype
        min_value = model.min_mask_value()
        prompt_ids = prompt_ids.to(device)
        t = prompt_ids.size(1)
        k = self.num_masks

        # Mask init gets its own Generator, a copy of the sampling generator's
        # current state: sample_embedding noise is the same as when it drew from
        # the shared generator, but token sampling no longer loses those draws,
        # so temperature > 0 stays token-identical to ar_generate.
        provider = self.mask_policy
        stats: dict[str, Any] = {'prefill_calls': 0, 'decode_calls': 0, 'accepted': 0,
                 'committed': 0, 'hit_context_limit': 0, 'trace': [],
                 # Draft attempts/accepts at depth 1..3. An attempt is a
                 # sampled token that had tree children to match; the bonus
                 # token past the last child is not an attempt. Counted before
                 # EOS/budget truncation, same as `accepted`.
                 'attempt_by_depth': [0, 0, 0, 0],
                 'accept_by_depth': [0, 0, 0, 0]}

        self.stats = stats
        self.committed = []
        if self.max_new_tokens == 0 or self._poll_cancel():
            stats.update(new_tokens=0, prefill_time=0., decode_time=0.,
                         termination_reason='cancelled' if self.was_cancelled else 'budget',
                         block_complexity=self.nominal_block_complexity)
            return prompt_ids.clone(), stats
        if t + k > model.max_positions:
            raise ValueError('prompt plus mask slots exceeds context capacity')

        # ---- Prefill: [prompt | m_1..m_k], plain causal mask. -------------
        start = time.perf_counter()
        prompt_embeds = model.embed(prompt_ids)[0]                    # [t, d]
        provider.init_from_prompt(prompt_embeds, generator=_fork_generator(self.generator))
        masks = provider.masks().to(dtype)                            # [k, d]
        prefill_embeds = torch.cat([prompt_embeds, masks], dim=0).unsqueeze(0)
        position_ids = torch.arange(t + k, device=device).unsqueeze(0)
        cache = model.new_cache()
        out = model(prefill_embeds, position_ids, attention_mask_4d=None,
                    past_key_values=cache, **({'last_hidden_only': True} if self.output_hidden else {}))
        stats['prefill_calls'] = 1
        cache = out.past_key_values
        logits = out.logits[0]                                        # [t+k, V]

        choices = []
        def select_verified(row, forward):
            chosen = select_token(logits[row], self.temperature, self.generator)
            if self.temperature <= 0:
                choices.append({'forward': forward, 'row': row, 'chosen': chosen,
                                'argmax': int(logits[row].argmax(dim=-1).item())})
            return chosen
        root_token = select_verified(t - 1, 0)
        if self.output_hidden:
            cast(HiddenMaskPolicy, provider).init_hidden(out.last_hidden_state[0, :t])
            proxy_hiddens = out.last_hidden_state[0, t:t + k].clone()
            stats['output_hidden_prefill'] = {'observed_positions': t, 'scale': float(cast(HiddenMaskPolicy, provider).scale),
                                              'state_dtype': 'float32', 'final_normalized': True}
        mask_logits = logits[t:t + k].clone()                         # [k, V]
        model.crop_cache(cache, t)                                       # evict masks

        committed = [root_token]
        self.committed = committed
        self._emit({'event': 'commit', 'index': 0, 'token': root_token,
                    'forward': 0, 'row': t - 1,
                    'argmax': int(logits[t - 1].argmax().item()) if self.temperature <= 0 else None})
        prev_token_embed = prompt_embeds[-1]
        root_embed = model.embed(torch.tensor([[root_token]], device=device))[0, 0]
        if not self.output_hidden:
            provider.on_token_committed(root_embed, prev_token_embed)
        prev_token_embed = root_embed
        stats['prefill_time'] = time.perf_counter() - start

        # ---- Decode: simultaneous verification + generation. --------------
        decode_start = time.perf_counter()
        while len(committed) < self.max_new_tokens and not self._is_eos(committed[-1]) and not self._poll_cancel():
            remaining = self.max_new_tokens - len(committed)
            probs = tree_probs(mask_logits, self.temperature)
            tree = build_tree(self.spec_cfg, probs, committed[-1])
            num_nodes = tree.num_nodes
            cache_len = model.cache_len(cache)
            root_position = t + len(committed) - 1
            assert cache_len == root_position, 'cache must hold exactly the prefix'
            if root_position + 1 + max(tree.depths or [0]) + k > model.max_positions:
                stats['hit_context_limit'] = 1
                break

            # Block embeddings: root, node tokens, k masks per owner.
            node_ids = torch.tensor([tree.tokens], dtype=torch.long, device=device)
            node_embeds = model.embed(node_ids)[0] if num_nodes else \
                torch.empty(0, prompt_embeds.size(-1), device=device, dtype=dtype)
            if self.output_hidden:
                depth_masks = cast(HiddenMaskPolicy, provider).masks_per_owner(proxy_hiddens).to(dtype)
                owner_masks = depth_masks[[0] + tree.depths]
            else:
                masks = provider.masks().to(dtype)
                owner_masks = masks.unsqueeze(0).expand(1 + num_nodes, k, -1)
            block_embeds = torch.cat([
                prev_token_embed.unsqueeze(0),         # root = last committed token
                node_embeds,
                owner_masks.reshape(-1, prompt_embeds.size(-1)),
            ], dim=0).unsqueeze(0)

            if self.impl == 'efficient':
                attn, position_ids = self._attn_cache.get(
                    tree, k, cache_len, root_position, dtype, device, min_value)
            elif self._dynamic_attn is not None:
                attn, position_ids = self._dynamic_attn.get(
                    tree, k, cache_len, root_position, dtype, device, min_value)
            else:
                attn, position_ids = build_block_attention(
                    tree, k, cache_len, root_position, dtype, device, min_value)

            out = model(block_embeds, position_ids, attention_mask_4d=attn,
                        past_key_values=cache, **({'last_hidden_only': True} if self.output_hidden else {}))
            cache = out.past_key_values
            logits = out.logits[0]                                    # [block, V]
            stats['decode_calls'] += 1

            # Verify: walk the model's own prediction chain through the tree.
            cur = 0
            accepted_nodes: list[int] = []
            committed_now = []
            while True:
                y = select_verified(cur, stats['decode_calls'])
                committed_now.append(y)
                children = tree.children_of(cur)
                match = None
                for child in children:
                    if tree.tokens[child - 1] == y:
                        match = child
                        break
                depth = len(accepted_nodes) + 1
                if children and depth < len(stats['attempt_by_depth']):
                    stats['attempt_by_depth'][depth] += 1
                    if match is not None:
                        stats['accept_by_depth'][depth] += 1
                if match is None:
                    break
                accepted_nodes.append(match)
                cur = match

            choice_offset = len(choices) - len(committed_now)
            stats['accepted'] += len(accepted_nodes)

            # EOS / budget truncation both end generation after this pass.
            stop = False
            for idx, token in enumerate(committed_now):
                if self._is_eos(token):
                    committed_now = committed_now[:idx + 1]
                    stop = True
                    break
            if len(committed_now) > remaining:
                committed_now = committed_now[:remaining]
                stop = True

            for token in committed_now:
                new_embed = model.embed(torch.tensor([[token]], device=device))[0, 0]
                if not self.output_hidden:
                    provider.on_token_committed(new_embed, prev_token_embed)
                prev_token_embed = new_embed
            committed.extend(committed_now)
            stats['committed'] += len(committed_now)
            for index, token in enumerate(committed_now):
                record = choices[choice_offset + index] if self.temperature <= 0 else {
                    'forward': stats['decode_calls'], 'row': ([0] + accepted_nodes)[index], 'argmax': None}
                self._emit({'event': 'commit', 'index': len(committed) - len(committed_now) + index,
                            'token': token, 'forward': record['forward'],
                            'row': record['row'], 'argmax': record['argmax']})

            stats['trace'].append({
                'root_position': root_position, 'cache_before': cache_len,
                'accepted_nodes': list(accepted_nodes), 'committed_tokens': list(committed_now),
                'stopped': stop, 'cache_after': None,
            })
            if self.output_hidden:
                observed = [] if stop else [0] + accepted_nodes
                for block_row in observed:
                    cast(HiddenMaskPolicy, provider).observe_committed(out.last_hidden_state[0, block_row])
                stats['trace'][-1]['output_hidden'] = {
                    'observed_block_rows': observed,
                    'owner_count': int(owner_masks.shape[0]),
                    'slot_count': int(owner_masks.shape[1]),
                    'mask_norm_min': float(owner_masks.float().norm(dim=-1).min()),
                    'mask_norm_max': float(owner_masks.float().norm(dim=-1).max()),
                    'last_owner_diff_from_root': float((owner_masks[-1] - owner_masks[0]).float().norm()),
                }
            if stop:
                break

            # Cache keeps root + accepted path; masks/rejects are evicted.
            model.keep_cache(cache, cache_len, [0] + accepted_nodes)
            stats['trace'][-1]['cache_after'] = model.cache_len(cache)

            # Next tree seeds: masks of the deepest accepted node (or root).
            n_star = accepted_nodes[-1] if accepted_nodes else 0
            mask_rows = mask_indices_of(tree, k, n_star)
            mask_logits = logits[mask_rows].clone()
            if self.output_hidden:
                proxy_hiddens = out.last_hidden_state[0, mask_rows].clone()

        stats['decode_time'] = time.perf_counter() - decode_start
        stats['new_tokens'] = len(committed)
        stats['termination_reason'] = ('cancelled' if self.was_cancelled else
            'context' if stats['hit_context_limit'] else 'eos' if self._is_eos(committed[-1]) else 'budget')
        stats['block_complexity'] = self.nominal_block_complexity
        output = torch.cat([
            prompt_ids,
            torch.tensor([committed], dtype=torch.long, device=device),
        ], dim=1)
        if self.temperature <= 0:
            records, offset = choices[:1], 1
            for trace in stats['trace']:
                records.extend(choices[offset:offset + len(trace['committed_tokens'])])
                offset += len(trace['accepted_nodes']) + 1
            token_hash = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            stats['verification_argmax_audit'] = {
                'passed': all(c['chosen'] == c['argmax'] for c in choices),
                'checked_committed_tokens': len(records),
                'selected_tokens_including_trimmed_tail': len(choices),
                'input_sha256': token_hash(prompt_ids[0].tolist()),
                'output_sha256': token_hash(output[0].tolist()), 'committed': records}
        return output, stats


@torch.no_grad()
def ar_generate(model, prompt_ids, eval_cfg, eos_ids=(), generator=None):
    """Plain KV-cached autoregressive decoding: the tau/speedup reference and
    the lossless audit target (identical sampling semantics as SpecDecoder)."""
    temperature = float(eval_cfg.get('temperature', 0.0))
    max_new_tokens = int(eval_cfg.get('max_new_tokens', 100))
    stop_on_eos = bool(eval_cfg.get('stop_on_eos', True))
    eos = set(int(e) for e in eos_ids)

    _validate_prompt(prompt_ids, model.vocab_size)
    if type(eval_cfg.get('max_new_tokens', 100)) is not int or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a nonnegative integer')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if max_new_tokens == 0:
        return prompt_ids.clone(), {'ar_calls': 0, 'ar_time': 0., 'ar_new_tokens': 0}
    device = model.device
    prompt_ids = prompt_ids.to(device)
    t = prompt_ids.size(1)
    start = time.perf_counter()

    cache = model.new_cache()
    position_ids = torch.arange(t, device=device).unsqueeze(0)
    out = model(model.embed(prompt_ids), position_ids, attention_mask_4d=None,
                past_key_values=cache)
    cache = out.past_key_values
    committed = [select_token(out.logits[0, -1], temperature, generator)]
    calls = 1

    while len(committed) < max_new_tokens:
        if stop_on_eos and committed[-1] in eos:
            break
        pos = t + len(committed) - 1
        if pos + 1 > model.max_positions:
            break
        ids = torch.tensor([[committed[-1]]], dtype=torch.long, device=device)
        out = model(model.embed(ids),
                    torch.tensor([[pos]], dtype=torch.long, device=device),
                    attention_mask_4d=None, past_key_values=cache)
        cache = out.past_key_values
        committed.append(select_token(out.logits[0, -1], temperature, generator))
        calls += 1

    stats = {
        'ar_calls': calls,
        'ar_time': time.perf_counter() - start,
        'ar_new_tokens': len(committed),
    }
    output = torch.cat([
        prompt_ids,
        torch.tensor([committed], dtype=torch.long, device=device),
    ], dim=1)
    return output, stats


def _validate_prompt(prompt_ids, vocab_size):
    if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1
            or prompt_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError('prompt must be a nonempty integer token tensor [1, length]')
    if (prompt_ids < 0).any() or (prompt_ids >= vocab_size).any():
        raise ValueError('prompt token outside model vocabulary')


class DecodeFailure(RuntimeError):
    def __init__(self, result):
        super().__init__('decode failed; committed facts are available in result')
        self.result = result


def generate(request: DecodeRequest, model_session: ModelSession, mask_policy: MaskPolicy,
             sink: ObservationSink, cancel: Cancel | None = None) -> DecodeResult:
    """One scientific prompt; no model construction, Run scheduling or retry."""
    generator = torch.Generator(device=model_session.device).manual_seed(request.seed)
    decoder = SpecDecoder(model_session, dict(request.spec),
                          {'max_new_tokens': request.max_new_tokens, 'temperature': request.temperature,
                           'stop_on_eos': request.stop_on_eos}, request.eos_ids, generator,
                          mask_policy=mask_policy, sink=sink, cancel=cancel)
    prompt = torch.tensor([request.prompt.input_ids], dtype=torch.long, device=model_session.device)
    try:
        sink.write({'event': 'header', 'schema': 1, 'run_id': request.run_id,
                    'qid': request.prompt.qid, 'input_ids': list(request.prompt.input_ids),
                    'sources': dict(request.sources), 'temperature': request.temperature})
        output, stats = decoder.generate(prompt)
        reason = stats['termination_reason']
        result = DecodeResult('cancelled' if reason == 'cancelled' else 'completed', reason,
                              tuple(output[0, len(request.prompt.input_ids):].tolist()), stats, True)
        sink.write({'event': 'end', 'execution': result.execution, 'termination_reason': reason,
                    'committed': list(result.committed), 'stats': dict(stats)})
        return result
    except Exception as error:
        partial = DecodeResult('failed', 'error', tuple(decoder.committed), dict(decoder.stats), False)
        raise DecodeFailure(partial) from error
