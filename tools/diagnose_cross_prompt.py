"""Fixed q171/q172/q174 replication using the existing observer/transfer functions.

    python tools/diagnose_cross_prompt.py --out .tmp/output-cross-prompt-<new-id>
"""

import argparse
import json
from pathlib import Path

from diagnose_output_windows import cosine, file_hash, margin, observe
from diagnose_proxy_transfer import transfer


def select_window(zero, one):
    by_zero = {t['root_position']: t for t in zero}
    by_one = {t['root_position']: t for t in one}
    common = sorted(set(by_zero) & set(by_one))
    hit = next(r for r in common if by_zero[r]['accepted_nodes'] and by_one[r]['accepted_nodes'])
    index = common.index(hit)
    assert index > 0
    root = common[index - 1]
    assert not by_zero[root]['accepted_nodes'] and not by_one[root]['accepted_nodes']
    return root, hit, sorted((set(by_zero) | set(by_one)) & set(range(root, max(common) + 1)))


def compare_window(sample, baseline, target):
    import torch
    states = {}
    for key in ('anchor', 'velocity', 'proxies'):
        a, b = sample['current_state'][key], baseline['current_state'][key]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        states[key] = {'bit_equal': torch.equal(a, b), 'max_abs_delta': float((a - b).abs().max()),
                       'L2_delta': float((a - b).norm())}
    return {'root_position': sample['seed_positions'][0], 'source_block_root': sample['source_root'],
            'source_owner': sample['source_owner'], 'first_slot': margin(sample['seed_logits'][0], target),
            'baseline_first_slot': margin(baseline['seed_logits'][0], target),
            'first_seed_bit_equal': torch.equal(sample['seed_mask_inputs'][0], baseline['seed_mask_inputs'][0]),
            'first_seed_cosine': cosine(sample['seed_mask_inputs'][0], baseline['seed_mask_inputs'][0]),
            'committed_velocity_norm': float(sample['current_state']['velocity'].norm()), 'states': states}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new; prior evidence is preserved')
    args.out.mkdir(parents=True)
    import torch
    import yaml
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM

    root = Path(__file__).resolve().parents[1]
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    studies = {name: root / 'studies' / name for name in ('output_space_prerequisite', 'output_space_small_matrix')}
    paths = {'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'),
             'tool': Path(__file__), 'observer_helper': Path(__file__).with_name('diagnose_output_windows.py'),
             'transfer_helper': Path(__file__).with_name('diagnose_proxy_transfer.py')}
    datasets, rows = {}, {}
    for name, study in studies.items():
        paths[name + ':execution'] = study / 'docs/execution_v1.json'
        paths[name + ':snapshot'] = study / 'docs/modelscope_snapshot.json'
        datasets[name] = read(paths[name + ':execution'])
        assert datasets[name]['study_quality_passed'] and datasets[name]['source_digest'] == '670c744e0535dcad'
        rows[name] = {r['case']: r for r in datasets[name].get('runs', datasets[name].get('rows'))}
        for case in ('output_order0', 'output_order1', 'AR'):
            folder = study / 'runs' / rows[name][case]['run_id']
            for filename in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
                paths[name + ':' + case + ':' + filename] = folder / filename
            assert file_hash(folder / 'assets/decode_observations.json') == rows[name][case]['raw_sha256']
    binding = {k + '_sha256': file_hash(p) for k, p in paths.items()}
    report = {'version': 'output-space-cross-q171-172-174-20261014-v1', 'source_digest': '670c744e0535dcad',
              'source_binding': binding, 'selection_rule': 'root immediately before first shared acceptance',
              'generation_budget': 1500, 'auxiliary_forwards': 0, 'prompts': [], 'quality_passed': False}
    tensors = {}
    try:
        assert manifest()['digest'] == report['source_digest']
        cache = studies['output_space_small_matrix'] / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        snapshot = read(paths['output_space_small_matrix:snapshot'])
        assert read(paths['output_space_prerequisite:snapshot'])['files'] == snapshot['files']
        for filename, metadata in snapshot['files'].items():
            assert file_hash(cache / filename) == metadata['sha256']
        torch.set_num_threads(1)
        apply_runtime(123, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                          torch_dtype='float32', attn_implementation='eager').to('cuda')
        for name in studies:
            assert model_digest(model) == datasets[name]['model_digest']
            assert runtime() == rows[name]['output_order1']['runtime']
        for qid, name, index, seed, fixed_root in [(171, 'output_space_prerequisite', 0, 123, 63),
                (172, 'output_space_small_matrix', 0, 123, 42), (174, 'output_space_small_matrix', 2, 125, 41)]:
            data, native = datasets[name], rows[name]
            assert (data['question_id'] if name == 'output_space_prerequisite' else data['question_ids'][index]) == qid
            ar = native['AR']['raw']['evidence'][index]
            refs = {c: native[c]['raw']['evidence'][index] for c in ('output_order0', 'output_order1')}
            intervention_root, hit, targets = select_window(refs['output_order0']['stats']['trace'], refs['output_order1']['stats']['trace'])
            assert intervention_root == fixed_root
            prompt = torch.tensor([ar['input_ids']], device=model.device)
            entry = {'question_id': qid, 'generator_seed': seed, 'native_study': name,
                     'intervention_root': fixed_root, 'first_shared_hit': hit, 'targets': targets,
                     'baselines': [], 'cases': []}
            report['prompts'].append(entry)
            configs, inputs, baseline = {}, {}, {}
            for case in ('output_order0', 'output_order1'):
                prefix = name + ':' + case + ':'
                inputs[case] = read(paths[prefix + 'assets/decode_inputs.json'])
                cfg = yaml.safe_load(paths[prefix + 'config.yaml'].read_text(encoding='utf-8'))
                configs[case] = _convert(cfg['algorithm']['config']['eval'])
                assert inputs[case]['prompts'][index]['input_ids'] == ar['input_ids'] == refs[case]['input_ids']
                assert refs[case]['output_ids'] == ar['output_ids']
                samples, replay = observe(model, prompt, _convert(native[case]['spec']), configs[case],
                    inputs[case]['eos_ids'], refs[case], targets, seed=seed)
                baseline[case] = samples
                tensors.update({f'q{qid}:baseline:{case}:{r}': s for r, s in samples.items()})
                entry['baselines'].append({'case': case, 'native_run_id': native[case]['run_id'],
                    'replay': replay, 'stats': {k: refs[case]['stats'][k] for k in ('accepted', 'new_tokens', 'decode_calls')},
                    'trace': refs[case]['stats']['trace']})
            assert torch.equal(baseline['output_order0'][fixed_root]['current_masks'][0, 0],
                               baseline['output_order1'][fixed_root]['current_masks'][0, 0])
            for recipient, donor in [('output_order1', 'output_order0'), ('output_order0', 'output_order1'),
                                     ('output_order1', 'output_order1')]:
                label = recipient + '_from_' + donor
                samples, replay, intervention = transfer(model, prompt, _convert(native[recipient]['spec']),
                    configs[recipient], inputs[recipient]['eos_ids'], refs[recipient],
                    baseline[donor][fixed_root]['current_masks'][0, 1],
                    baseline[recipient][fixed_root]['current_masks'], fixed_root, targets,
                    placebo=recipient == donor, seed=seed)
                tensors[f'q{qid}:{label}:intervention'] = intervention
                tensors.update({f'q{qid}:{label}:{r}': s for r, s in samples.items()})
                assert torch.equal(samples[fixed_root]['seed_mask_inputs'], baseline[recipient][fixed_root]['seed_mask_inputs'])
                windows = [compare_window(s, baseline[recipient][r], ar['output_ids'][r + 1]) for r, s in sorted(samples.items())]
                if recipient == 'output_order0':
                    assert all(w['committed_velocity_norm'] == 0 for w in windows)
                prior = {t['root_position']: t for t in refs[recipient]['stats']['trace']}
                actual = {t['root_position']: t for t in replay['trace']}
                differences = [{'root_position': r, 'baseline_accepted_nodes': prior[r]['accepted_nodes'],
                                'actual_accepted_nodes': actual[r]['accepted_nodes']}
                               for r in sorted(set(prior) & set(actual))
                               if prior[r]['accepted_nodes'] != actual[r]['accepted_nodes']]
                entry['cases'].append({'label': label, 'recipient': recipient, 'donor': donor,
                    'replay': replay, 'windows': windows, 'matched_acceptance_changes': differences,
                    'actual_only_roots': sorted(set(actual) - set(prior)), 'baseline_only_roots': sorted(set(prior) - set(actual)),
                    'second_slot_vector_L2_delta': float((intervention['donor_vector'] - intervention['original_vector']).norm())})
        assert all(file_hash(p) == binding[k + '_sha256'] for k, p in paths.items())
        assert manifest()['digest'] == report['source_digest'] and model_digest(model) == datasets['output_space_small_matrix']['model_digest']
        assert runtime() == rows['output_space_small_matrix']['output_order1']['runtime']
        for filename, metadata in snapshot['files'].items():
            assert file_hash(cache / filename) == metadata['sha256']
        report.update(model_digest=datasets['output_space_small_matrix']['model_digest'], runtime=runtime(), quality_passed=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'prompts': [{'question_id': p['question_id'],
          'accepted': [c['replay']['stats']['accepted'] for c in p['cases']]} for p in report['prompts']]}))


if __name__ == '__main__':
    main()
