"""Native owner1 rare-event draws and stable detectability calculations, no model forwards.

    python tools/diagnose_rare_events.py --out .tmp/rare-events-<new-id>
"""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_output_windows import file_hash
from diagnose_sampling_paths import conditional_acceptance


def detectability(probability, trials):
    if not 0 <= probability <= 1 or trials <= 0:
        raise ValueError('probability must be in [0,1] and trials positive')
    if probability == 0:
        detection, required = 0., None
    elif probability == 1:
        detection, required = 1., 1
    else:
        log_miss = math.log1p(-probability)
        detection = -math.expm1(trials * log_miss)
        required = math.ceil(math.log(.05) / log_miss)
    return {'ideal_Bernoulli_probability_at_least_one': detection,
        'ideal_trials_for_95pct_detection': required,
        'one_sided_95pct_upper_after_zero': -math.expm1(math.log(.05) / trials)}


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
    report = {'version': 'rare-events-20261015-v1', 'source_binding': binding, 'source_digest': manifest()['digest'],
        'runtime': runtime(), 'model_forwards': 0, 'new_generation_tokens': 0,
        'groups': [], 'quality_passed': False}
    try:
        for temperature in (.8, 1.):
            group = conditional_acceptance(logits, candidates, temperature, range(300, 4396))
            reference64 = torch.softmax(logits.cpu().double() / temperature, dim=-1)
            for cell in group['cells']:
                cell['CPU_FP64_mass'] = float(reference64[cell['candidate']])
                cell.update(detectability(cell['exact_candidate_probability'], group['verifier_draws']))
                cell['zero_upper_bound_applies'] = cell['first_slot_hits'] == 0
            report['groups'].append(group)
        assert all(file_hash(p) == binding[k] for k, p in paths.items())
        assert manifest()['digest'] == report['source_digest'] and runtime() == report['runtime']
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'native_verifier_draws': 8192,
        'results': [(g['temperature'], [(c['candidate'], c['first_slot_hits']) for c in g['cells']]) for g in report['groups']]}))


if __name__ == '__main__':
    main()
