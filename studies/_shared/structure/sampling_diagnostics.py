"""Library helpers folded from one-shot structure scripts."""
def conditional_acceptance(verifier_logits, candidates, temperature, seeds):
    import torch
    from spec_mtp.decode import select_token
    assert temperature > 0 and verifier_logits.ndim == 1 and torch.isfinite(verifier_logits).all()
    probs = torch.softmax(verifier_logits.float() / temperature, dim=-1)
    draws = []
    for seed in seeds:
        generator = torch.Generator(device=verifier_logits.device).manual_seed(seed)
        before = generator.get_state().clone()
        draws.append(select_token(verifier_logits, temperature, generator))
        assert not torch.equal(before, generator.get_state())
    return {'temperature': temperature, 'seeds': list(seeds), 'verifier_token_ids': draws, 'verifier_draws': len(draws), 'RNG_consumed': True, 'same_verifier_draws_for_all_cells': True, 'cells': [{'label': label, 'candidate': token, 'exact_candidate_probability': float(probs[token]), 'first_slot_hits': sum((draw == token for draw in draws))} for label, token in candidates.items()]}
import math

def detectability(probability, trials):
    if not 0 <= probability <= 1 or trials <= 0:
        raise ValueError('probability must be in [0,1] and trials positive')
    if probability == 0:
        detection, required = (0.0, None)
    elif probability == 1:
        detection, required = (1.0, 1)
    else:
        log_miss = math.log1p(-probability)
        detection = -math.expm1(trials * log_miss)
        required = math.ceil(math.log(0.05) / log_miss)
    return {'ideal_Bernoulli_probability_at_least_one': detection, 'ideal_trials_for_95pct_detection': required, 'one_sided_95pct_upper_after_zero': -math.expm1(math.log(0.05) / trials)}

def event_summary(draws, seeds, candidate, probability):
    from scipy.stats import beta
    if len(draws) != len(seeds) or not draws or (not 0 <= probability <= 1):
        raise ValueError('nonempty aligned draws/seeds and valid probability required')
    hits = [seed for seed, draw in zip(seeds, draws) if draw == candidate]
    n, k = (len(draws), len(hits))
    return {'hits': k, 'first_hit_seed': hits[0] if hits else None, 'hit_seeds': hits, 'observed_rate': k / n, 'Clopper_Pearson_95pct': [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))], 'ideal_expected_hits': n * probability, **detectability(probability, n)}
