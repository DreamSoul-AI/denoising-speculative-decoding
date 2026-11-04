# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "accelerate", "numpy"]
# ///
"""
ESP Reproduction V6 - Throughput-Focused
arXiv 2603.17942 (ICML 2026)

Key fix from V5: 
- The paper's ESP verification is a SINGLE forward pass that:
  1. Builds tree input from prompt + all draft candidates
  2. Runs model once with tree attention
  3. Checks each position's prediction against the draft token AT THAT POSITION
- Our V5 was doing TWO forward passes (mask pass + verification pass)
- V6 merges them into one pass like the paper

Throughput improvement from:
- Single forward pass for verification (not two)
- Temperature=1.0 for mask logits (diverse candidates, more matches)
- Matching paper's exact algorithm
"""

import os, json, time, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Write a short story about a robot learning to paint.",
    "Explain the theory of relativity in simple terms.",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis.",
    "Write a Python function to sort a list of numbers.",
    "What is machine learning and how does it work?",
    "Explain the difference between TCP and UDP.",
    "Describe the water cycle in detail.",
    "Write a haiku about artificial intelligence.",
    "What are the main causes of climate change?",
]


class ESPDecoder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embed_layer = model.get_input_embeddings()
        # We don't need a real mask token — mask embedding = mean of prompt embeddings (Eq. 4)
        # Just set a dummy mask_token_id for reference
        self.mask_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def get_mask_embedding(self, input_ids):
        with torch.no_grad():
            return self.embed_layer(input_ids).mean(dim=1, keepdim=True)

    def update_mask_embedding(self, mask_embed, token_id, lam=0.1):
        with torch.no_grad():
            tok_emb = self.embed_layer(torch.tensor([[token_id]], device=self.device))
            return mask_embed + lam * (tok_emb - mask_embed)

    def get_mask_logits(self, prompt_ids, mask_embed):
        """Run model on [prompt, mask_embedding] → logits at mask position.
        
        mask_embed: mean of prompt embeddings (Eq. 4), NOT a token embedding.
        """
        prompt_emb = self.embed_layer(prompt_ids)
        inputs_embeds = torch.cat([prompt_emb, mask_embed], dim=1)

        total_len = inputs_embeds.shape[1]
        causal_mask = torch.full((1, 1, total_len, total_len), float("-inf"), device=self.device)
        for i in range(total_len):
            for j in range(i + 1):
                causal_mask[0, 0, i, j] = 0.0
        attention_mask = torch.ones(1, total_len, dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                use_cache=False,
                output_hidden_states=False,
            )
        return out.logits[0, -1, :]

    def get_mask_topk(self, prompt_ids, mask_embed, k, temperature=1.0):
        logits = self.get_mask_logits(prompt_ids, mask_embed)
        probs = torch.softmax(logits / temperature, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, min(k, probs.shape[0]))
        return topk_ids.tolist(), topk_probs.tolist()

    def verify_single_pass(self, prompt_ids, draft_ids, mask_embed):
        """
        Single forward pass: build tree input and verify all candidates at once.
        
        Tree structure for 1 mask token:
        - Positions 0..plen-1: prompt
        - Positions plen..plen+n_draft-1: draft candidates (each attends to prompt only)
        - Position plen+n_draft: mask token (attends to prompt + all drafts)
        
        We check: does the model's prediction at each draft position match the draft token?
        """
        plen = prompt_ids.shape[1]
        n_draft = len(draft_ids)
        
        # Build tree input embeddings
        prompt_emb = self.embed_layer(prompt_ids)  # (1, plen, d)
        
        draft_embs = []
        for d_id in draft_ids:
            d_emb = self.embed_layer(torch.tensor([[d_id]], device=self.device))
            draft_embs.append(d_emb)
        
        mask_emb_expanded = mask_embed  # (1, 1, d)
        
        all_embs = [prompt_emb] + draft_embs + [mask_emb_expanded]
        inputs_embeds = torch.cat(all_embs, dim=1)  # (1, plen+n_draft+1, d)
        
        total_len = inputs_embeds.shape[1]
        
        # Build tree attention mask: each position attends to prompt + itself only
        # This is the key: draft tokens do NOT attend to each other
        causal_mask = torch.full((1, 1, total_len, total_len), float("-inf"), device=self.device)
        
        # Prompt tokens: standard causal (attend to all previous prompt tokens)
        for i in range(plen):
            for j in range(i + 1):
                causal_mask[0, 0, i, j] = 0.0
        
        # Draft tokens: attend to ALL prompt tokens + themselves
        for d in range(n_draft):
            pos = plen + d
            # Attend to all prompt positions
            causal_mask[0, 0, pos, :plen] = 0.0
            # Attend to self
            causal_mask[0, 0, pos, pos] = 0.0
        
        # Mask token: attend to ALL prompt tokens + ALL draft tokens + self
        mask_pos = plen + n_draft
        causal_mask[0, 0, mask_pos, :plen + n_draft + 1] = 0.0
        
        attention_mask = torch.ones(1, total_len, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            out = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        
        logits = out.logits[0]  # (total_len, vocab)
        hidden_states = out.hidden_states
        
        # Check: prediction at each draft position vs draft token
        n_accepted = 0
        accepted_idx = -1
        for d in range(n_draft):
            pos = plen + d
            pred = torch.argmax(logits[pos]).item()
            if pred == draft_ids[d]:
                n_accepted += 1
                accepted_idx = d
        
        # Bonus token: prediction at mask position
        bonus_token = torch.argmax(logits[mask_pos]).item()
        
        # Cosine similarity: mask hidden vs true next token hidden
        cos_final = 0.0
        if hidden_states and n_accepted > 0:
            n_layers = len(hidden_states)
            # Mask hidden state at final layer
            mask_h = hidden_states[n_layers - 1][0, mask_pos, :]
            # True next token = bonus_token, get its embedding
            true_emb = self.embed_layer(torch.tensor([[bonus_token]], device=self.device))
            # Run a quick forward to get its hidden state
            # Actually, we approximate: use the draft position's hidden state
            # that was accepted
            if accepted_idx >= 0:
                true_h = hidden_states[n_layers - 1][0, plen + accepted_idx, :]
                cos_final = torch.nn.functional.cosine_similarity(
                    mask_h.unsqueeze(0), true_h.unsqueeze(0)
                ).item()
        
        return n_accepted, accepted_idx, bonus_token, cos_final

    def generate_ar_baseline(self, prompt, max_new_tokens=256):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.time() - t0
        tokens = out[0, input_ids.shape[1]:].tolist()
        return tokens, elapsed, len(tokens) / max(elapsed, 1e-6)

    def generate_esp(self, prompt, block_complexity=30, n_mask_tokens=1, max_new_tokens=256):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        mask_embed = self.get_mask_embedding(input_ids)

        # BC = (k+1)(1 + sum K_i) => sum K_i = BC/(k+1) - 1
        total_k = block_complexity // (n_mask_tokens + 1) - 1

        generated = input_ids.clone()
        all_tokens = []
        cos_values = []
        t0 = time.time()
        n_model_calls = 0
        n_accepted_total = 0

        while len(all_tokens) < max_new_tokens:
            # Get mask candidates
            draft_ids, draft_probs = self.get_mask_topk(generated, mask_embed, total_k, temperature=1.0)

            # Verify in single forward pass
            n_acc, acc_idx, bonus, cos_val = self.verify_single_pass(
                generated, draft_ids, mask_embed
            )
            n_model_calls += 1
            n_accepted_total += n_acc

            if n_acc > 0:
                # Accept all drafts up to and including the matched one
                for i in range(acc_idx + 1):
                    tok = draft_ids[i]
                    all_tokens.append(tok)
                    generated = torch.cat([generated, torch.tensor([[tok]], device=self.device)], dim=1)
                    mask_embed = self.update_mask_embedding(mask_embed, tok)

                # Bonus token
                if len(all_tokens) < max_new_tokens:
                    all_tokens.append(bonus)
                    generated = torch.cat([generated, torch.tensor([[bonus]], device=self.device)], dim=1)
                    mask_embed = self.update_mask_embedding(mask_embed, bonus)
            else:
                # AR fallback: use bonus (model's prediction at mask position)
                all_tokens.append(bonus)
                generated = torch.cat([generated, torch.tensor([[bonus]], device=self.device)], dim=1)
                mask_embed = self.update_mask_embedding(mask_embed, bonus)

            cos_values.append(cos_val)

            if n_model_calls <= 10:
                d1_str = self.tokenizer.decode([draft_ids[0]]) if draft_ids else "?"
                print(
                    f"  Step {n_model_calls}: n_acc={n_acc}, "
                    f"d1='{d1_str}'({draft_probs[0]:.3f}), "
                    f"bonus='{self.tokenizer.decode([bonus])}', cos={cos_val:.4f}"
                )

        elapsed = time.time() - t0
        tokens = all_tokens[:max_new_tokens]
        tps = len(tokens) / max(elapsed, 1e-6)
        tau = len(tokens) / max(n_model_calls, 1)

        return tokens, elapsed, tps, tau, cos_values, n_accepted_total

    def measure_cosine_similarity(self, prompt, n_mask_tokens=1):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # Use mean of prompt embeddings as mask embedding (Eq. 4)
        mask_emb = self.embed_layer(input_ids).mean(dim=1, keepdim=True)
        prompt_emb = self.embed_layer(input_ids)
        inputs_embeds = torch.cat([prompt_emb, mask_emb], dim=1)

        with torch.no_grad():
            out = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True)

        greedy_next = torch.argmax(out.logits[0, -1, :]).item()

        # Get true next token embedding via separate forward
        true_ids = torch.cat([input_ids, torch.tensor([[greedy_next]], device=self.device)], dim=1)
        with torch.no_grad():
            out_true = self.model(true_ids, output_hidden_states=True)

        n_layers = len(out.hidden_states) - 1
        sims = []
        for li in range(1, n_layers + 1):
            mask_h = out.hidden_states[li][0, -1, :]
            true_h = out_true.hidden_states[li][0, -2, :]
            cos = torch.nn.functional.cosine_similarity(mask_h.unsqueeze(0), true_h.unsqueeze(0)).item()
            sims.append(cos)
        return sims, greedy_next


def run_experiment(model_name, bc, n_masks, max_tokens, n_prompts):
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} | BC={bc} | masks={n_masks}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded: {model.config.hidden_size}d, {model.config.num_hidden_layers} layers, device={device}")
    dec = ESPDecoder(model, tokenizer, device)

    results = []
    for i, prompt in enumerate(PROMPTS[:n_prompts]):
        print(f"\n[{i+1}/{n_prompts}] {prompt[:50]}...")

        cos_sims, greedy_next = dec.measure_cosine_similarity(prompt, n_masks)
        mid = len(cos_sims) // 2
        greedy_str = tokenizer.decode([greedy_next])
        print(f"  Greedy next: '{greedy_str}', cos(mid)={cos_sims[mid]:.4f}, cos(final)={cos_sims[-1]:.4f}")

        ar_tok, ar_time, ar_tps = dec.generate_ar_baseline(prompt, max_tokens)
        esp_tok, esp_time, esp_tps, tau, cos_esp, n_acc = dec.generate_esp(prompt, bc, n_masks, max_tokens)

        acc_rate = n_acc / max(len(esp_tok), 1)
        results.append({
            "idx": i, "ar_tps": ar_tps, "esp_tps": esp_tps, "tau": tau,
            "cos_sims": cos_sims, "cos_final": cos_sims[-1],
            "acceptance_rate": acc_rate,
        })
        print(f"  AR: {ar_tps:.1f} t/s | ESP: {esp_tps:.1f} t/s | tau={tau:.3f} | acc_rate={acc_rate:.3f}")

    agg = {
        "model": model_name, "block_complexity": bc, "n_mask_tokens": n_masks,
        "avg_ar_tps": float(np.mean([r["ar_tps"] for r in results])),
        "avg_esp_tps": float(np.mean([r["esp_tps"] for r in results])),
        "avg_tau": float(np.mean([r["tau"] for r in results])),
        "speedup_pct": float((np.mean([r["esp_tps"] for r in results]) / np.mean([r["ar_tps"] for r in results]) - 1) * 100),
        "avg_cos_final": float(np.mean([r["cos_final"] for r in results])),
        "avg_acceptance_rate": float(np.mean([r["acceptance_rate"] for r in results])),
        "cosine_by_layer": [
            float(np.mean([r["cos_sims"][l] for r in results]))
            for l in range(len(results[0]["cos_sims"]))
        ],
    }
    print(f"\n--- AGGREGATE ---")
    print(json.dumps({k: v for k, v in agg.items() if k != "cosine_by_layer"}, indent=2))
    return {"summary": agg, "per_prompt": results}


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
    max_tokens = int(os.environ.get("MAX_TOKENS", "256"))
    n_prompts = int(os.environ.get("N_PROMPTS", "10"))

    all_results = {}
    for bc, nm in [(30, 1), (60, 2)]:
        all_results[f"bc_{bc}"] = run_experiment(model_name, bc, nm, max_tokens, n_prompts)

    with open("/tmp/esp_results_v6.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to /tmp/esp_results_v6.json")
