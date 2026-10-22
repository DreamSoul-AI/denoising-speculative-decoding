"""Manifest-bound four-prompt window transfers using unchanged diagnostic helpers.

    python tools/diagnose_study_windows.py --study studies/output_space_followup --out .tmp/followup-transfer-<new-id>
"""

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies.output_space_extension.analyze_results import slots
from diagnose_extension import choose_window
from diagnose_cross_prompt import compare_window
from diagnose_output_windows import file_hash, margin, observe
from diagnose_proxy_transfer import transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
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

    study = args.study.resolve()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    data = read(study / 'docs/execution_v1.json')
    selection = read(study / 'docs/window_selection_v1.json')
    assert data['study_quality_passed'] and len(data['question_ids']) == 4
    assert data['question_ids'] == [p['question_id'] for p in read(study / 'docs/prompts_manifest.json')['rendered_prompts']]
    assert selection['matrix_sha256'] == file_hash(study / 'docs/execution_v1.json')
    assert selection['runner_sha256'] == file_hash(Path(__file__))
    paths = {'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'), 'matrix': study / 'docs/execution_v1.json',
        'selection': study / 'docs/window_selection_v1.json', 'tool': Path(__file__),
        'snapshot': study / 'docs/modelscope_snapshot.json',
        'observer_helper': Path(__file__).with_name('diagnose_output_windows.py'),
        'transfer_helper': Path(__file__).with_name('diagnose_proxy_transfer.py'),
        'compare_helper': Path(__file__).with_name('diagnose_cross_prompt.py'),
        'selection_helper': Path(__file__).with_name('diagnose_extension.py'),
        'slots_helper': Path(__file__).resolve().parents[1] / 'studies/output_space_extension/analyze_results.py'}
    native = {r['case']: r for r in data['runs']}
    for case in ('output_order0', 'output_order1', 'AR'):
        folder = study / 'runs' / native[case]['run_id']
        for filename in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
            paths[case + ':' + filename] = folder / filename
        assert file_hash(folder / 'assets/decode_observations.json') == native[case]['raw_sha256']
    binding = {k + '_sha256': file_hash(p) for k, p in paths.items()}
    report = {'version': data['version'] + '-transfer', 'source_digest': data['source_digest'],
              'source_binding': binding, 'generation_budget': 2000, 'auxiliary_forwards': 0,
              'prompts': [], 'quality_passed': False}
    tensors = {}
    try:
        assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        snapshot = read(paths['snapshot'])
        for filename, metadata in snapshot['files'].items():
            assert file_hash(cache / filename) == metadata['sha256']
        torch.set_num_threads(1)
        apply_runtime(123, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                          torch_dtype='float32', attn_implementation='eager').to('cuda')
        assert model_digest(model) == data['model_digest'] and runtime() == native['output_order1']['runtime']
        for index, frozen in enumerate(selection['prompts']):
            qid, intervention_root, targets = frozen['question_id'], frozen['intervention_root'], frozen['targets']
            assert qid == data['question_ids'][index]
            refs = {c: native[c]['raw']['evidence'][index] for c in ('output_order0', 'output_order1')}
            assert choose_window(refs['output_order0']['stats']['trace'], refs['output_order1']['stats']['trace']) == {k: frozen[k] for k in ('intervention_root', 'selection_reason', 'targets')}
            entry = {**frozen, 'generator_seed': 123 + index, 'baselines': [], 'cases': []}
            report['prompts'].append(entry)
            if intervention_root is None:
                entry['status'] = 'no feasible continuing window; prompt retained'
                continue
            ar = native['AR']['raw']['evidence'][index]
            prompt = torch.tensor([ar['input_ids']], device=model.device)
            baseline, configs, inputs = {}, {}, {}
            for case in ('output_order0', 'output_order1'):
                inputs[case] = read(paths[case + ':assets/decode_inputs.json'])
                cfg = yaml.safe_load(paths[case + ':config.yaml'].read_text(encoding='utf-8'))
                configs[case] = _convert(cfg['algorithm']['config']['eval'])
                assert inputs[case]['prompts'][index]['input_ids'] == ar['input_ids'] == refs[case]['input_ids']
                assert refs[case]['output_ids'] == ar['output_ids']
                visited = {t['root_position'] for t in refs[case]['stats']['trace']}
                samples, replay = observe(model, prompt, _convert(native[case]['spec']), configs[case],
                    inputs[case]['eos_ids'], refs[case], sorted(set(targets) & visited), seed=123 + index)
                baseline[case] = samples
                tensors.update({f'q{qid}:baseline:{case}:{r}': s for r, s in samples.items()})
                entry['baselines'].append({'case': case, 'replay': replay,
                    'stats': {k: refs[case]['stats'][k] for k in ('accepted', 'new_tokens', 'decode_calls')},
                    'trace': refs[case]['stats']['trace']})
            entry['cross_order_first_slot_bit_equal_at_intervention'] = torch.equal(
                baseline['output_order0'][intervention_root]['current_masks'][0, 0],
                baseline['output_order1'][intervention_root]['current_masks'][0, 0])
            natural = []
            for position in sorted(set(baseline['output_order0']) & set(baseline['output_order1'])):
                zero = margin(baseline['output_order0'][position]['seed_logits'][0], ar['output_ids'][position + 1])
                one = margin(baseline['output_order1'][position]['seed_logits'][0], ar['output_ids'][position + 1])
                natural.append({'root_position': position, 'zero_margin': zero['target_margin_vs_best_other'],
                    'one_margin': one['target_margin_vs_best_other'],
                    'opposite_margin_signs': zero['target_margin_vs_best_other'] * one['target_margin_vs_best_other'] < 0,
                    'zero_slots': slots(baseline['output_order0'][position], ar['output_ids'], position),
                    'one_slots': slots(baseline['output_order1'][position], ar['output_ids'], position)})
            entry['natural_baseline_margin_pairs'] = natural
            for recipient, donor in [('output_order1', 'output_order0'), ('output_order0', 'output_order1'), ('output_order1', 'output_order1')]:
                label = recipient + '_from_' + donor
                samples, replay, intervention = transfer(model, prompt, _convert(native[recipient]['spec']),
                    configs[recipient], inputs[recipient]['eos_ids'], refs[recipient],
                    baseline[donor][intervention_root]['current_masks'][0, 1],
                    baseline[recipient][intervention_root]['current_masks'], intervention_root, targets,
                    placebo=recipient == donor, seed=123 + index)
                tensors[f'q{qid}:{label}:intervention'] = intervention
                tensors.update({f'q{qid}:{label}:{r}': s for r, s in samples.items()})
                assert torch.equal(samples[intervention_root]['seed_mask_inputs'], baseline[recipient][intervention_root]['seed_mask_inputs'])
                windows = []
                for position, sample in sorted(samples.items()):
                    if position in baseline[recipient]:
                        w = compare_window(sample, baseline[recipient][position], ar['output_ids'][position + 1])
                        w['recipient_baseline_visited'] = True
                    else:
                        w = {'root_position': position, 'recipient_baseline_visited': False,
                             'first_slot': margin(sample['seed_logits'][0], ar['output_ids'][position + 1]),
                             'committed_velocity_norm': float(sample['current_state']['velocity'].norm())}
                    if recipient == 'output_order0':
                        assert w['committed_velocity_norm'] == 0
                    if position in baseline[donor]:
                        w['first_seed_bit_equal_to_donor_baseline'] = torch.equal(sample['seed_mask_inputs'][0], baseline[donor][position]['seed_mask_inputs'][0])
                        w['first_seed_L2_to_donor_baseline'] = float((sample['seed_mask_inputs'][0] - baseline[donor][position]['seed_mask_inputs'][0]).norm())
                    w['slot_rankings'] = slots(sample, ar['output_ids'], position)
                    windows.append(w)
                prior = {t['root_position']: t for t in refs[recipient]['stats']['trace']}
                actual = {t['root_position']: t for t in replay['trace']}
                changes = [{'root_position': r, 'baseline_accepted_nodes': prior[r]['accepted_nodes'],
                            'actual_accepted_nodes': actual[r]['accepted_nodes']}
                           for r in sorted(set(prior) & set(actual)) if prior[r]['accepted_nodes'] != actual[r]['accepted_nodes']]
                entry['cases'].append({'label': label, 'recipient': recipient, 'donor': donor, 'replay': replay,
                    'windows': windows, 'matched_acceptance_changes': changes,
                    'actual_only_roots': sorted(set(actual) - set(prior)), 'baseline_only_roots': sorted(set(prior) - set(actual)),
                    'second_slot_vector_L2_delta': float((intervention['donor_vector'] - intervention['original_vector']).norm())})
            entry['status'] = 'all fixed transfers passed'
        assert all(file_hash(p) == binding[k + '_sha256'] for k, p in paths.items())
        assert manifest()['digest'] == data['source_digest'] and model_digest(model) == data['model_digest']
        assert runtime() == native['output_order1']['runtime']
        for filename, metadata in snapshot['files'].items():
            assert file_hash(cache / filename) == metadata['sha256']
        report.update(model_digest=data['model_digest'], runtime=runtime(), quality_passed=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if tensors:
            torch.save(tensors, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'prompts': [{'qid': p['question_id'], 'root': p['intervention_root'], 'accepted': [c['replay']['stats']['accepted'] for c in p['cases']]} for p in report['prompts']]}))


if __name__ == '__main__':
    main()
