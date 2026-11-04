# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "accelerate", "numpy"]
# ///
"""
ESP Reproduction V5 - Correct Implementation
arXiv 2603.17942 (ICML 2026)

Key correction from V2/V4:
- Cosine similarity must compare mask hidden state vs TRUE NEXT TOKEN hidden state
  (not vs last prompt position hidden state)
- The true next token x_{t+1} must be included in the input to get its hidden state
- Verification: in tree-attention forward pass, model predicts at each position;
  we check if draft tokens match model's greedy predictions
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

    def get_mask_embedding(self, input_ids):
        """Mean of prompt embeddings (Eq. 4)."""
        with torch.no_grad():
            return self.embed_layer(input_ids).mean(dim=1, keepdim=True)

    def update_mask_embedding(self, mask_embed, token_id, lam=0.1):
        """Update mask embedding (Eq. 5)."""
        with torch.no_grad():
            tok_emb = self.embed_layer(torch.tensor([[token_id]], device=self.device))
            return mask_embed + lam * (tok_emb - mask_embed)

    def measure_cosine_similarity(self, prompt, n_mask_tokens=1):
        """
        Correct cosine similarity measurement per paper:
        Compare hidden state of mask token m_{t+1} vs true next token x_{t+1},
        both at the last decoder layer, both used to predict token at position t+2.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Get greedy next token
        with torch.no_grad():
            out = self.model(input_ids)
            greedy_next = torch.argmax(out.logits[0, -1]).item()

        # Build input: [prompt, true_next_token]
        true_token = torch.tensor([[greedy_next]], device=self.device)
        ids_with_true = torch.cat([input_ids, true_token], dim=1)

        # Get mask embedding
        mask_embed = self.get_mask_embedding(input_ids)

        # Run 1: [prompt, mask] to get mask hidden state at position t+1
        prompt_embeds = self.embed_layer(input_ids)
        mask_embeds = mask_embed.expand(-1, n_mask_tokens, -1)
        full_embeds_m = torch.cat([prompt_embeds, mask_embeds], dim=1)
        L = input_ids.shape[1]
        seq_m = L + n_mask_tokens
        attn_m = torch.ones(1, seq_m, device=self.device, dtype=torch.long)
        pos_m = torch.arange(seq_m, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out_m = self.model(
                inputs_embeds=full_embeds_m, attention_mask=attn_m,
                position_ids=pos_m, output_hidden_states=True,
            )

        # Run 2: [prompt, true_next_token] to get true hidden state at position t+1
        true_embeds = self.embed_layer(true_token)
        full_embeds_v = torch.cat([prompt_embeds, true_embeds], dim=1)
        seq_v = L + 1
        attn_v = torch.ones(1, seq_v, device=self.device, dtype=torch.long)
        pos_v = torch.arange(seq_v, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out_v = self.model(
                inputs_embeds=full_embeds_v, attention_mask=attn_v,
                position_ids=pos_v, output_hidden_states=True,
            )

        # Compute cosine similarity at each layer
        layer_sims = []
        n_layers = len(out_m.hidden_states)
        for li in range(n_layers):
            # Mask hidden state at position t+1 (last mask position)
            h_m = out_m.hidden_states[li][0, -1, :]
            # True next-token hidden state at position t+1 (last position)
            h_v = out_v.hidden_states[li][0, -1, :]
            cos = torch.nn.functional.cosine_similarity(
                h_m.unsqueeze(0), h_v.unsqueeze(0)
            ).item()
            layer_sims.append(cos)

        return layer_sims, greedy_next

    def get_mask_topk(self, input_ids, mask_embed, k):
        """Get top-K candidates from mask logits."""
        prompt_embeds = self.embed_layer(input_ids)
        full_embeds = torch.cat([prompt_embeds, mask_embed], dim=1)
        L = input_ids.shape[1]
        seq_len = L + 1
        attn = torch.ones(1, seq_len, device=self.device, dtype=torch.long)
        pos = torch.arange(seq_len, device=self.device).unsqueeze(0)

        with torch.no_grad():
            out = self.model(inputs_embeds=full_embeds, attention_mask=attn, position_ids=pos)
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits / 0.01, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, min(k, probs.shape[0]))
        return topk_ids.tolist(), topk_probs.tolist()

    def verify_and_accept(self, input_ids, mask_embed, draft_ids, n_mask_tokens):
        """
        Verify draft tokens in a single tree-attention forward pass.

        Token layout: [prompt, draft_0, draft_1, ..., draft_{K-1}, mask_0, ..., mask_{M-1}]

        Tree attention: draft_i attends to prompt + self only (not other drafts).
        This means each draft is evaluated independently against the prompt.

        Verification logic:
        - Model logits at position L-1 (last prompt pos) predict the next token
        - Check if this prediction matches any draft → accept it + bonus

        Returns: n_accepted, accepted_token, bonus_token, cos_sims
        """
        prompt_len = input_ids.shape[1]
        n_draft = len(draft_ids)
        n_mask = n_mask_tokens

        # Build embeddings
        prompt_embeds = self.embed_layer(input_ids)
        draft_tensor = torch.tensor([draft_ids], device=self.device)
        draft_embeds = self.embed_layer(draft_tensor)
        mask_embeds = mask_embed.expand(-1, n_mask, -1)
        full_embeds = torch.cat([prompt_embeds, draft_embeds, mask_embeds], dim=1)
        total_len = prompt_len + n_draft + n_mask

        # Build tree attention as 4D causal_mask: (batch, 1, q_len, kv_len)
        # 0.0 = attend, -inf = don't attend
        causal_mask = torch.full((1, 1, total_len, total_len), float("-inf"), device=self.device)

        # Prompt: standard causal
        for i in range(prompt_len):
            causal_mask[0, 0, i, : i + 1] = 0.0

        # Drafts: each attends to prompt + self only (NOT other drafts)
        for i in range(n_draft):
            pos = prompt_len + i
            causal_mask[0, 0, pos, :prompt_len] = 0.0
            causal_mask[0, 0, pos, pos] = 0.0

        # Masks: attend to everything before them
        for i in range(n_mask):
            pos = prompt_len + n_draft + i
            causal_mask[0, 0, pos, : pos + 1] = 0.0

        position_ids = torch.arange(total_len, device=self.device).unsqueeze(0)

        # Use full attention mask (all 1s) since we handle masking via causal_mask
        attention_mask = torch.ones(1, total_len, device=self.device, dtype=torch.long)

        with torch.no_grad():
            out = self.model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                output_hidden_states=True,
            )

        logits = out.logits[0]

        # === VERIFICATION ===
        # Logits at position L-1 predict what should be at position L
        pred_token = torch.argmax(logits[prompt_len - 1]).item()

        n_accepted = 0
        accepted_token = None

        # Check if prediction matches any draft
        if pred_token in draft_ids:
            n_accepted = 1
            accepted_token = pred_token

        # Bonus token: logits at the position after the accepted token
        # If accepted, bonus = logits at position of accepted draft → predicts next
        if n_accepted > 0:
            # Find position of accepted draft
            draft_pos = prompt_len + draft_ids.index(accepted_token)
            bonus_token = torch.argmax(logits[draft_pos]).item()
        else:
            # No acceptance: bonus = model's own next prediction (AR fallback)
            bonus_token = pred_token

        # Cosine similarity from hidden states
        cos_sims = []
        if out.hidden_states and n_mask > 0:
            for li in range(len(out.hidden_states)):
                hs = out.hidden_states[li]
                # Mask hidden state
                mask_h = hs[0, prompt_len + n_draft + n_mask - 1, :]
                # True next-token hidden state: run separately
                cos_sims.append(0.0)  # placeholder

        return n_accepted, accepted_token, bonus_token

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

        # Branch config from BC
        # BC = (k+1)(1 + sum K_i) => sum K_i = BC/(k+1) - 1
        total_k = block_complexity // (n_mask_tokens + 1) - 1
        n_candidates = total_k  # for depth-1, all candidates at depth 1

        generated = input_ids.clone()
        all_tokens = []
        accepted_counts = []
        cos_final_layer = []
        t0 = time.time()
        n_model_calls = 0

        while len(all_tokens) < max_new_tokens:
            # Step 1: Get mask candidates
            draft_ids, draft_probs = self.get_mask_topk(generated, mask_embed, n_candidates)

            # Step 2: Verify in single forward pass
            n_acc, acc_token, bonus = self.verify_and_accept(
                generated, mask_embed, draft_ids, n_mask_tokens
            )
            n_model_calls += 1

            # Step 3: Measure cosine similarity for accepted token
            if n_acc > 0 and len(all_tokens) < max_new_tokens:
                # Reconstruct prompt up to this point
                cos_sims, _ = self.measure_cosine_similarity(
                    self.tokenizer.decode(generated[0].tolist()), n_mask_tokens
                )
                cos_final_layer.append(cos_sims[-1] if cos_sims else 0.0)

            # Step 4: Update generated sequence
            if n_acc > 0:
                # Accept the matched draft
                all_tokens.append(acc_token)
                generated = torch.cat([generated, torch.tensor([[acc_token]], device=self.device)], dim=1)
                mask_embed = self.update_mask_embedding(mask_embed, acc_token)
                accepted_counts.append(1)

                # Bonus token
                if len(all_tokens) < max_new_tokens:
                    all_tokens.append(bonus)
                    generated = torch.cat([generated, torch.tensor([[bonus]], device=self.device)], dim=1)
                    mask_embed = self.update_mask_embedding(mask_embed, bonus)
                    accepted_counts.append(1)
            else:
                # AR fallback
                all_tokens.append(bonus)
                generated = torch.cat([generated, torch.tensor([[bonus]], device=self.device)], dim=1)
                mask_embed = self.update_mask_embedding(mask_embed, bonus)
                accepted_counts.append(0)

            # Debug
            if n_model_calls <= 5:
                d1_str = self.tokenizer.decode([draft_ids[0]]) if draft_ids else "?"
                acc_str = self.tokenizer.decode([acc_token]) if acc_token else "none"
                bonus_str = self.tokenizer.decode([bonus])
                print(
                    f"  Step {n_model_calls}: acc={n_acc}, "
                    f"d1='{d1_str}'({draft_probs[0]:.3f}), "
                    f"accepted='{acc_str}', bonus='{bonus_str}'"
                )

        elapsed = time.time() - t0
        tokens = all_tokens[:max_new_tokens]
        tps = len(tokens) / max(elapsed, 1e-6)
        tau = len(tokens) / max(n_model_calls, 1)

        return tokens, elapsed, tps, tau, cos_final_layer

    def measure_all_cosine(self, prompt, n_mask_tokens=1):
        """Measure cosine similarity for a prompt."""
        sims, greedy = self.measure_cosine_similarity(prompt, n_mask_tokens)
        greedy_str = self.tokenizer.decode([greedy])
        return sims, greedy_str


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
    n_layers = model.config.num_hidden_layers
    print(f"  Model loaded: {model.config.hidden_size}d, {n_layers} layers, device={device}")
    dec = ESPDecoder(model, tokenizer, device)

    results = []
    for i, prompt in enumerate(PROMPTS[:n_prompts]):
        print(f"\n[{i+1}/{n_prompts}] {prompt[:50]}...")

        # Cosine similarity measurement
        cos_sims, greedy_str = dec.measure_all_cosine(prompt, n_masks)
        mid = len(cos_sims) // 2
        print(f"  Greedy next: '{greedy_str}', cos(mid-L{mid})={cos_sims[mid]:.4f}, cos(final-L{len(cos_sims)-1})={cos_sims[-1]:.4f}")

        # AR baseline
        ar_tok, ar_time, ar_tps = dec.generate_ar_baseline(prompt, max_tokens)

        # ESP
        esp_tok, esp_time, esp_tps, tau, cos_esp = dec.generate_esp(prompt, bc, n_masks, max_tokens)

        results.append({
            "idx": i, "ar_tps": ar_tps, "esp_tps": esp_tps, "tau": tau,
            "cos_sims": cos_sims, "cos_final": cos_sims[-1], "cos_45": cos_sims[45] if len(cos_sims) > 45 else cos_sims[-1],
        })
        print(f"  AR: {ar_tps:.1f} t/s | ESP: {esp_tps:.1f} t/s | tau={tau:.3f}")

    agg = {
        "model": model_name, "block_complexity": bc, "n_mask_tokens": n_masks,
        "avg_ar_tps": float(np.mean([r["ar_tps"] for r in results])),
        "avg_esp_tps": float(np.mean([r["esp_tps"] for r in results])),
        "avg_tau": float(np.mean([r["tau"] for r in results])),
        "speedup": float(np.mean([r["esp_tps"] for r in results]) / np.mean([r["ar_tps"] for r in results])),
        "avg_cos_final": float(np.mean([r["cos_final"] for r in results])),
        "avg_cos_45": float(np.mean([r["cos_45"] for r in results])),
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

    with open("/tmp/esp_results_v5.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to /tmp/esp_results_v5.json")
