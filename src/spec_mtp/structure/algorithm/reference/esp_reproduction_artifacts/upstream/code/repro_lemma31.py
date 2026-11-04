"""
Numerical Audit of Lemma 3.1 from ESP paper (arXiv 2603.17942)

Lemma 3.1: Let h_m, h_v ∈ R^d be hidden states for the mask token and the 
next-true token after the last decoder layer and let W ∈ R^{d×V} be the LM head 
with columns w_r ∈ R^d. 

Assume ||h_m||_2, ||h_v||_2 ≤ c_h and ||w_r||_2 ≤ c_w, ∀ r.

Define i* = argmax_r w_r^T h_v as the next-next true token (under greedy sampling)
and S_m = argtopK_r w_r^T h_m as the top-K tokens under the mask token.

Then, i* ∈ S_m if cos(h_m, h_v) ≥ δ* for some δ* > 0.

The proof bounds |w_r^T(h_m - h_v)| ≤ c_w ||h_m - h_v||_2 ≤ c_w * sqrt(2) * c_h * sqrt(1-δ)
and shows that the margin Δ_j^m = w_{i*}^T h_m - w_j^T h_m satisfies
Δ_j^m ≥ Δ_j^v - 2c*sqrt(1-δ) where c = sqrt(2)*c_w*c_h.

This means i* remains in Top-K when δ (cosine similarity) is high enough.

We verify this numerically with random LM heads and hidden states.
"""

import numpy as np
import json
from typing import Tuple, List

def verify_lemma_3_1(
    d: int = 4096,      # hidden dimension (LLaMA3.1-8B)
    V: int = 128256,     # vocab size (LLaMA3.1-8B)
    K: int = 10,         # top-K
    c_h: float = 1.0,    # norm bound for hidden states
    c_w: float = 1.0,    # norm bound for weight columns
    n_trials: int = 100,  # number of random trials
    delta_values: List[float] = None,
    seed: int = 42
) -> dict:
    """
    Numerically verify Lemma 3.1 by:
    1. Creating random weight matrix W and hidden states h_m, h_v
    2. For various cosine similarity values (delta), checking if i* ∈ S_m
    """
    rng = np.random.RandomState(seed)
    
    if delta_values is None:
        delta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Generate random weight matrix with bounded norms
    W_raw = rng.randn(d, V).astype(np.float64)
    # Scale to have bounded column norms
    col_norms = np.linalg.norm(W_raw, axis=0, keepdims=True)
    W = W_raw / col_norms * c_w  # Now ||w_r||_2 = c_w for all r
    
    results = {}
    
    for delta_target in delta_values:
        successes = 0
        delta_theoretical_threshold = []
        
        for trial in range(n_trials):
            # Generate h_v with bounded norm
            h_v_raw = rng.randn(d).astype(np.float64)
            h_v = h_v_raw / np.linalg.norm(h_v_raw) * c_h
            
            # Generate h_m with specified cosine similarity to h_v
            # h_m = alpha * h_v + beta * orthogonal_component
            # cos(h_m, h_v) = alpha * ||h_v|| / ||h_m|| 
            # We want cos = delta_target
            
            # Random orthogonal direction
            ortho_raw = rng.randn(d).astype(np.float64)
            # Make orthogonal to h_v
            ortho_raw = ortho_raw - np.dot(ortho_raw, h_v) / np.dot(h_v, h_v) * h_v
            ortho = ortho_raw / np.linalg.norm(ortho_raw)
            
            # h_m = delta * h_v_unit + sqrt(1-delta^2) * ortho_unit (both scaled by c_h)
            h_m = c_h * (delta_target * h_v / c_h + np.sqrt(max(0, 1 - delta_target**2)) * ortho)
            
            # Verify cosine similarity
            cos_sim = np.dot(h_m, h_v) / (np.linalg.norm(h_m) * np.linalg.norm(h_v))
            
            # Compute logits
            logits_m = W.T @ h_m  # shape (V,)
            logits_v = W.T @ h_v  # shape (V,)
            
            # i* = argmax under h_v (true next token)
            i_star = np.argmax(logits_v)
            
            # S_m = top-K under h_m (mask token predictions)
            top_k_indices = np.argpartition(logits_m, -K)[-K:]
            
            # Check if i* ∈ S_m
            if i_star in top_k_indices:
                successes += 1
        
        success_rate = successes / n_trials
        
        # Theoretical analysis: compute the threshold delta*
        # From the proof: Δ_j^m ≥ Δ_j^v - 2c*sqrt(1-δ)
        # For i* to be in top-K, we need: for all j in top-K (excluding i*),
        # Δ_j^m > 0, i.e., w_{i*}^T h_m > w_j^T h_m
        # A sufficient condition is: min_j Δ_j^v > 2c*sqrt(1-δ)
        
        results[delta_target] = {
            "success_rate": success_rate,
            "cos_sim_actual": float(cos_sim),
            "n_trials": n_trials
        }
        
        print(f"δ = {delta_target:.2f}: success rate = {success_rate:.3f} ({successes}/{n_trials})")
    
    return results


def verify_proof_inequality(
    d: int = 4096,
    V: int = 128256,
    c_h: float = 1.0,
    c_w: float = 1.0,
    n_trials: int = 100,
    seed: int = 42
) -> dict:
    """
    Verify the key inequality from the proof:
    |w_r^T(h_m - h_v)| ≤ c_w * ||h_m - h_v||_2 ≤ sqrt(2) * c_w * c_h * sqrt(1-δ)
    """
    rng = np.random.RandomState(seed)
    
    W_raw = rng.randn(d, V).astype(np.float64)
    col_norms = np.linalg.norm(W_raw, axis=0, keepdims=True)
    W = W_raw / col_norms * c_w
    
    violations = 0
    max_ratio = 0.0
    
    for trial in range(n_trials):
        h_v_raw = rng.randn(d).astype(np.float64)
        h_v = h_v_raw / np.linalg.norm(h_v_raw) * c_h
        
        delta = rng.uniform(0.1, 0.9)
        ortho_raw = rng.randn(d).astype(np.float64)
        ortho_raw = ortho_raw - np.dot(ortho_raw, h_v) / np.dot(h_v, h_v) * h_v
        ortho = ortho_raw / np.linalg.norm(ortho_raw)
        
        h_m = c_h * (delta * h_v / c_h + np.sqrt(max(0, 1 - delta**2)) * ortho)
        
        cos_sim = np.dot(h_m, h_v) / (np.linalg.norm(h_m) * np.linalg.norm(h_v))
        
        # Compute ||h_m - h_v||_2
        diff_norm = np.linalg.norm(h_m - h_v)
        
        # Compute |w_r^T(h_m - h_v)| for all r
        diff = h_m - h_v
        inner_products = np.abs(W.T @ diff)  # shape (V,)
        max_inner = np.max(inner_products)
        
        # Theoretical bound
        theoretical_bound = c_w * np.sqrt(2) * c_h * np.sqrt(max(0, 1 - cos_sim))
        
        # Check: |w_r^T(h_m - h_v)| ≤ c_w * ||h_m - h_v||_2
        bound_1_satisfied = max_inner <= c_w * diff_norm + 1e-10
        
        # Check: c_w * ||h_m - h_v||_2 ≤ sqrt(2) * c_w * c_h * sqrt(1-δ)
        bound_2_satisfied = c_w * diff_norm <= theoretical_bound + 1e-10
        
        if not (bound_1_satisfied and bound_2_satisfied):
            violations += 1
        
        ratio = max_inner / theoretical_bound if theoretical_bound > 1e-15 else float('inf')
        max_ratio = max(max_ratio, ratio)
    
    return {
        "violations": violations,
        "n_trials": n_trials,
        "max_ratio_to_bound": max_ratio,
        "all_bounds_satisfied": violations == 0
    }


def analyze_margin_based_guarantee(
    d: int = 4096,
    V: int = 128256,
    K: int = 10,
    c_h: float = 1.0,
    c_w: float = 1.0,
    n_trials: int = 100,
    seed: int = 42
) -> dict:
    """
    Analyze the margin-based guarantee more carefully:
    For i* to be in Top-K, we need:
    Δ_j^m = w_{i*}^T h_m - w_j^T h_m > 0 for all j in top-K(m) \ {i*}
    
    The sufficient condition from the proof is:
    min_j Δ_j^v > 2c*sqrt(1-δ) where c = sqrt(2)*c_w*c_h
    """
    rng = np.random.RandomState(seed)
    
    W_raw = rng.randn(d, V).astype(np.float64)
    col_norms = np.linalg.norm(W_raw, axis=0, keepdims=True)
    W = W_raw / col_norms * c_w
    
    delta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    results = {}
    
    for delta_target in delta_values:
        guaranteed = 0
        actual_in_topk = 0
        
        for trial in range(n_trials):
            h_v_raw = rng.randn(d).astype(np.float64)
            h_v = h_v_raw / np.linalg.norm(h_v_raw) * c_h
            
            ortho_raw = rng.randn(d).astype(np.float64)
            ortho_raw = ortho_raw - np.dot(ortho_raw, h_v) / np.dot(h_v, h_v) * h_v
            ortho = ortho_raw / np.linalg.norm(ortho_raw)
            
            h_m = c_h * (delta_target * h_v / c_h + np.sqrt(max(0, 1 - delta_target**2)) * ortho)
            
            logits_m = W.T @ h_m
            logits_v = W.T @ h_v
            
            i_star = np.argmax(logits_v)
            
            # Compute margin Δ_j^v for j ≠ i*
            logits_v_no_istar = np.delete(logits_v, i_star)
            margin_v = logits_v[i_star] - logits_v_no_istar
            min_margin_v = np.min(margin_v)
            
            # Sufficient condition from proof
            c = np.sqrt(2) * c_w * c_h
            sufficient_threshold = 2 * c * np.sqrt(max(0, 1 - delta_target))
            
            if min_margin_v > sufficient_threshold:
                guaranteed += 1
            
            # Check actual Top-K membership
            top_k_indices = np.argpartition(logits_m, -K)[-K:]
            if i_star in top_k_indices:
                actual_in_topk += 1
        
        results[delta_target] = {
            "guaranteed_rate": guaranteed / n_trials,
            "actual_topk_rate": actual_in_topk / n_trials
        }
        print(f"δ = {delta_target:.2f}: guaranteed = {guaranteed/n_trials:.3f}, actual = {actual_in_topk/n_trials:.3f}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("NUMERICAL AUDIT OF LEMMA 3.1")
    print("=" * 60)
    
    # Use smaller dimension for speed, but enough to be meaningful
    d = 512
    V = 10000
    K = 10
    n_trials = 200
    
    print(f"\nParameters: d={d}, V={V}, K={K}, n_trials={n_trials}")
    
    print("\n--- Test 1: Bound inequality verification ---")
    bound_results = verify_proof_inequality(d=d, V=V, n_trials=n_trials)
    print(f"Violations: {bound_results['violations']}/{bound_results['n_trials']}")
    print(f"Max ratio to theoretical bound: {bound_results['max_ratio_to_bound']:.6f}")
    print(f"All bounds satisfied: {bound_results['all_bounds_satisfied']}")
    
    print("\n--- Test 2: Success rate vs cosine similarity ---")
    success_results = verify_lemma_3_1(d=d, V=V, K=K, n_trials=n_trials)
    
    print("\n--- Test 3: Margin-based guarantee analysis ---")
    margin_results = analyze_margin_based_guarantee(d=d, V=V, K=K, n_trials=n_trials)
    
    # Save results
    all_results = {
        "bound_verification": bound_results,
        "success_rate": {str(k): v for k, v in success_results.items()},
        "margin_analysis": {str(k): v for k, v in margin_results.items()},
        "parameters": {"d": d, "V": V, "K": K, "n_trials": n_trials}
    }
    
    with open("/tmp/lemma31_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\nResults saved to /tmp/lemma31_results.json")
