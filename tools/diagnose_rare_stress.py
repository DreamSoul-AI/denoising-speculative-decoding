"""Fixed owner1 native tail control and explicitly changed-temperature stress, no model forwards.

    python tools/diagnose_rare_stress.py --out .tmp/rare-stress-<new-id>
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_output_windows import file_hash
from diagnose_sampling_paths import conditional_acceptance
from diagnose_rare_events import detectability


def event_summary(draws, seeds, candidate, probability):
    from scipy.stats import beta
    if len(draws) != len(seeds) or not draws or not 0 <= probability <= 1:
        raise ValueError('nonempty aligned draws/seeds and valid probability required')
    hits = [seed for seed, draw in zip(seeds, draws) if draw == candidate]
    n, k = len(draws), len(hits)
    return {'hits': k, 'first_hit_seed': hits[0] if hits else None,
        'hit_seeds': hits, 'observed_rate': k/n,
        'Clopper_Pearson_95pct': [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                                 1. if k == n else float(beta.ppf(.975, k+1, n-k))],
        'ideal_expected_hits': n*probability, **detectability(probability, n)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.evidence import runtime
    root = Path(__file__).resolve().parents[1]
    paths = {'tool': Path(__file__), 'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'),
        'helper': root / 'tools/diagnose_sampling_paths.py',
        'owner_report': root / '.tmp/q241-owner1-20261015-v1/diagnosis.json',
        'owner_tensor': root / '.tmp/q241-owner1-20261015-v1/selected_tensors.pt',
        'local_report': root / '.tmp/q241-local-factors-20261015-v1/diagnosis.json',
        'local_tensor': root / '.tmp/q241-local-factors-20261015-v1/selected_tensors.pt'}
    binding = {k: file_hash(p) for k, p in paths.items()}
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    owner, local = read(paths['owner_report']), read(paths['local_report'])
    assert owner['quality_passed'] and local['quality_passed']
    assert owner['raw_tensor_sha256'] == binding['owner_tensor']
    assert local['raw_tensor_sha256'] == binding['local_tensor']
    assert manifest()['digest'] == '670c744e0535dcad'
    torch.set_num_threads(1)
    a = torch.load(paths['owner_tensor'], weights_only=True, map_location='cpu', mmap=True)
    b = torch.load(paths['local_tensor'], weights_only=True, map_location='cpu', mmap=True)
    candidates = {f'S{s}P{p}': int(a[f'r717:S{s}P{p}']['logits'][0, 5].argmax())
                  for s in (0, 1) for p in (0, 1)}
    logits = b['frame:719']['logits'][0, 0].cuda()
    report = {'version': 'rare-stress-20261015-v1', 'source_binding': binding, 'source_digest': manifest()['digest'],
        'runtime': runtime(), 'model_forwards': 0, 'new_generation_tokens': 0,
        'groups': [], 'quality_passed': False}
    try:
        for temperature, count in ((.8, 16384), (1., 16384), (2., 131072)):
            masses = torch.softmax(logits.float() / temperature, dim=-1)
            if temperature == 2.:
                assert all(detectability(float(masses[c]), count)['ideal_Bernoulli_probability_at_least_one'] >= .98 for c in candidates.values())
            group = conditional_acceptance(logits, candidates, temperature, range(10000, 10000+count))
            group['role'] = 'changed_temperature_stress' if temperature == 2. else 'original_temperature_control'
            reference64 = torch.softmax(logits.cpu().double() / temperature, dim=-1)
            for cell in group['cells']:
                cell['CPU_FP64_mass'] = float(reference64[cell['candidate']])
                cell.update(event_summary(group['verifier_token_ids'], group['seeds'], cell['candidate'], cell['exact_candidate_probability']))
                assert cell['hits'] == cell['first_slot_hits']
                cell['zero_upper_bound_applies'] = cell['first_slot_hits'] == 0
            report['groups'].append(group)
            print(json.dumps({'temperature': temperature, 'draws': count, 'hits': [(c['candidate'], c['hits']) for c in group['cells']]}), flush=True)
        assert all(file_hash(p) == binding[k] for k, p in paths.items())
        assert manifest()['digest'] == report['source_digest'] and runtime() == report['runtime']
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'native_verifier_draws': 163840,
        'results': [(g['temperature'], [(c['candidate'], c['first_slot_hits']) for c in g['cells']]) for g in report['groups']]}))


if __name__ == '__main__':
    main()
