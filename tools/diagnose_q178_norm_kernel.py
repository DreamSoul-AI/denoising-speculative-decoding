"""Compare same-GPU norm graphs at entry and subsequent provider calls.

    python tools/diagnose_q178_norm_kernel.py --out .tmp/q178-norm-kernel-<new-id>
"""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies.output_space_extension.kernel_common import load, verify, compare_control, profile
from diagnose_output_windows import file_hash
from diagnose_q178_numerics import synchronized_transfer
from spec_mtp.masks import OutputHiddenMaskProvider

NATIVE = OutputHiddenMaskProvider.masks_per_owner


def mapped(provider, proxies, alternative=True):
    import torch
    anchor, velocity = provider.anchor.clone(), provider.velocity.clone()
    owners, details = [], []
    for owner in range(provider.num_masks + 1):
        if owner < provider.num_masks:
            y = proxies[owner].detach().float()
            if provider.order:
                velocity = provider.beta_prime * velocity + (1 - provider.beta_prime) * (y - anchor)
            anchor = y
        else:
            anchor = anchor + velocity
        slots = torch.stack([anchor + i * velocity for i in range(provider.num_masks)])
        native = slots.norm(dim=-1, keepdim=True)
        alt = (slots * slots).sum(dim=-1, keepdim=True).sqrt()
        reference = slots.cpu().double().norm(dim=-1, keepdim=True).float().to(slots.device)
        norms = alt if alternative else native
        assert torch.isfinite(norms).all() and (norms > 0).all()
        owners.append(provider.scale * slots / norms)
        details.append({'owner': owner, 'native_norm': native.cpu().flatten().tolist(),
            'alternative_norm': alt.cpu().flatten().tolist(), 'reference64_rounded_norm': reference.cpu().flatten().tolist(),
            'native_reference_max_abs': float((native - reference).abs().max()),
            'alternative_reference_max_abs': float((alt - reference).abs().max())})
    result = torch.stack(owners)
    assert torch.isfinite(result).all()
    return result, details


def run(c, donor, post):
    import torch
    calls, deltas = 0, []
    def substitute(provider, proxies):
        nonlocal calls
        calls += 1
        original = NATIVE(provider, proxies)
        if not post or calls <= c.call:
            return original
        state = [provider.anchor.clone(), provider.velocity.clone(), proxies.clone()]
        result, _ = mapped(provider, proxies)
        assert all(torch.equal(a, b) for a, b in zip(state, [provider.anchor, provider.velocity, proxies]))
        deltas.append(float((result - original).abs().max()))
        return result
    with patch.object(OutputHiddenMaskProvider, 'masks_per_owner', substitute):
        samples, evidence, _, records = synchronized_transfer(c.model, c.states, donor=donor, **c.keywords)
    assert OutputHiddenMaskProvider.masks_per_owner is NATIVE
    return samples, evidence, records, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    root = Path(__file__).resolve().parents[1]
    report, tensors = {'quality_passed': False, 'cases': [], 'frozen_comparisons': []}, {}
    try:
        c = load(args.out, (root / 'docs/研究计划 Denoising Speculative Decoding.md'), Path(__file__))
        report.update(source_binding=c.binding, source_digest=c.native['source_digest'],
            model_digest=c.native['model_digest'], runtime=c.previous['runtime'], version='q178-norm-kernel-20261014-v1',
            generation_budget=800, auxiliary_model_forwards=0)
        donors = {}
        for case in ('output_order0', 'output_order1'):
            s = c.old[f'baseline:{case}:85']['current_state']
            p = OutputHiddenMaskProvider(2, c.model.embedding_table, order=0 if case.endswith('0') else 1)
            p.anchor, p.velocity = s['anchor'].cuda(), s['velocity'].cuda()
            proxies = s['proxies'].cuda()
            original = NATIVE(p, proxies)
            assert torch.equal(original.cpu(), c.old[f'baseline:{case}:85']['current_masks'])
            reproduced, _ = mapped(p, proxies, alternative=False)
            assert torch.equal(reproduced, original)
            alt, detail = mapped(p, proxies)
            donors[case] = {'native': original[0, 1].cpu(), 'entry': alt[0, 1].cpu()}
            report['frozen_comparisons'].append({'case': case, 'root': 85, 'norms': detail,
                'mask_max_abs': float((alt - original).abs().max()),
                'entry_vector_bit_equal': torch.equal(alt[0, 1], original[0, 1])})
        for label in ('canonical_one_self', 'canonical_one_from_zero'):
            s = c.old[f'{label}:86']['current_state']
            p = OutputHiddenMaskProvider(2, c.model.embedding_table, order=1)
            p.anchor, p.velocity = s['anchor'].cuda(), s['velocity'].cuda()
            proxies = s['proxies'].cuda()
            original = NATIVE(p, proxies)
            assert torch.equal(original.cpu(), c.old[f'{label}:86']['current_masks'])
            alt, detail = mapped(p, proxies)
            report['frozen_comparisons'].append({'case': label, 'root': 86, 'norms': detail,
                'mask_max_abs': float((alt - original).abs().max())})
            if label == 'canonical_one_self':
                slots = proxies[0].clone().expand(2, -1).clone()
                # Profile real slot values from owner0, not a fabricated proxy pair.
                v = p.beta_prime * p.velocity + (1 - p.beta_prime) * (proxies[0] - p.anchor)
                slots[0], slots[1] = proxies[0], proxies[0] + v
                report['native_profile'] = profile(lambda: slots.norm(dim=-1, keepdim=True),
                    args.out / 'norm-native.chrome.json', {'aten::linalg_vector_norm'})
                report['alternative_profile'] = profile(lambda: (slots * slots).sum(dim=-1, keepdim=True).sqrt(),
                    args.out / 'norm-alternative.chrome.json', {'aten::mul', 'aten::sum', 'aten::sqrt'})
        for stage, entry, post in [('native', False, False), ('entry', True, False),
                                   ('post', False, True), ('entry_post', True, True)]:
            for graft in (False, True):
                label = stage + ('_graft' if graft else '_self')
                donor_case = 'output_order0' if graft else 'output_order1'
                donor = donors[donor_case]['entry' if entry else 'native']
                samples, replay, records, deltas = run(c, donor, post)
                if stage == 'native':
                    compare_control(c, samples, replay, graft)
                report['cases'].append({'label': label, 'stage': stage, 'graft': graft, 'replay': replay,
                    'canonical_replacements': records, 'post_mapping_max_abs': max(deltas, default=0),
                    'entry_vector_changed': not torch.equal(donor, donors[donor_case]['native'])})
                tensors.update({f'{label}:{p}': s for p, s in samples.items()})
        verify(c)
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'cases': [(c['label'], c['replay']['stats']['accepted']) for c in report['cases']]}))


if __name__ == '__main__':
    main()
