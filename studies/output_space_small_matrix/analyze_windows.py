"""Reproduce offline slot1 structure checks from preserved observer tensors.

    python studies/output_space_small_matrix/analyze_windows.py --raw .tmp/output-windows-20261014-v1/selected_tensors.pt
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from spec_mtp.masks import OutputHiddenMaskProvider
    from studies._shared.structure.sources import manifest

    study = Path(__file__).resolve().parent
    observer = json.loads((study / 'docs/window_diagnosis_v1.json').read_text(encoding='utf-8'))
    matrix = json.loads((study / 'docs/execution_v1.json').read_text(encoding='utf-8'))
    with args.raw.open('rb') as handle:
        raw_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert raw_sha == observer['raw_tensor_sha256'] and manifest()['digest'] == '670c744e0535dcad'
    raw = torch.load(args.raw, weights_only=True)
    rows = {row['case']: row for row in matrix['runs']}
    findings = []
    for case in ['output_order0', 'output_order1']:
        scale = rows[case]['raw']['evidence'][1]['stats']['output_hidden_prefill']['scale']
        order = rows[case]['spec']['output']['order']
        for root in [83, 94, 97]:
            sample = raw[f'{case}:{root}']
            state = sample['source_state']
            assert sample['source_owner'] == 0
            table = torch.zeros(1, state['anchor'].numel())
            table[0, 0] = scale
            provider = OutputHiddenMaskProvider(2, table, order=order, beta=.9, beta_prime=.99)
            provider.anchor, provider.velocity = state['anchor'].clone(), state['velocity'].clone()
            actual = provider.masks_per_owner(state['proxies'])[0, 0]
            provider.velocity = torch.zeros_like(provider.velocity)
            zero = provider.masks_per_owner(state['proxies'])[0, 0]
            assert torch.equal(actual, zero)
            saved = sample['seed_mask_inputs'][0]
            assert torch.allclose(actual, saved, rtol=1e-6, atol=1e-7)
            findings.append({'case': case, 'root_position': root, 'source_block_root': sample['source_root'],
                             'source_owner': 0, 'source_velocity_norm': float(state['velocity'].norm()),
                             'slot1_identical_with_source_velocity_zeroed': True,
                             'saved_seed_reconstruction_max_abs_error': float((actual - saved).abs().max())})
    result = {'analysis_type': 'post hoc offline structural check, no additional model forward',
              'tensor_sha256': raw_sha, 'source_digest': '670c744e0535dcad',
              'formula': 'owner0 first slot j=0 is Phi(proxy[0]); committed velocity enters later slots, not this direct slot',
              'findings': findings,
              'interpretation': 'At these recorded owner0 seeds, different first-mask directions arise from different prior proxy hiddens. The gain is not a direct current-velocity addition to slot1; its earlier history propagation is not independently isolated.'}
    target = study / 'docs/structural_diagnosis_v1.json'
    if target.exists():
        assert json.loads(target.read_text(encoding='utf-8')) == result
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2) + '\n',encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
