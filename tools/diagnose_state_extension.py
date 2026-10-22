"""Fixed q242/q243 two-seed anchor/velocity interaction matrix.

    python tools/diagnose_state_extension.py --out .tmp/state-extension-<new-id>
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_output_windows import cosine, file_hash, observe, TRACE_KEYS, SEMANTIC_KEYS
from diagnose_proxy_transfer import transfer
from diagnose_q178_numerics import canonical_states, synchronized_transfer
from diagnose_q241_state_factors import selective_transfer


def frozen_targets(selection):
    entry = selection['intervention_root']
    later = [p for p in selection['targets'] if p > entry]
    return sorted(set([entry, *later[:3], *later[-2:]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    import yaml
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM
    root = Path(__file__).resolve().parents[1]
    study = root / 'studies/output_space_cross_task'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    paths = {'tool': Path(__file__), 'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'),
        'matrix': study / 'docs/execution_v1.json', 'selection': study / 'docs/window_selection_v1.json',
        'snapshot': study / 'docs/modelscope_snapshot.json'}
    for name in ('diagnose_output_windows.py', 'diagnose_proxy_transfer.py', 'diagnose_q178_numerics.py',
                 'diagnose_q241_state_factors.py'):
        paths[name] = root / 'tools' / name
    data = read(paths['matrix'])
    assert data['study_quality_passed']
    rows = {r['case']: r for r in data['runs']}
    for case in ('output_order0', 'output_order1', 'AR'):
        for name in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
            paths[case + ':' + name] = study / 'runs' / rows[case]['run_id'] / name
    binding = {k: file_hash(p) for k, p in paths.items()}
    snapshot = read(paths['snapshot'])
    cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
    assert all(file_hash(cache / n) == m['sha256'] for n, m in snapshot['files'].items())
    assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
    torch.set_num_threads(1)
    apply_runtime(124, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
    model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
        torch_dtype='float32', attn_implementation='eager').to('cuda')
    assert model_digest(model) == data['model_digest'] and runtime() == rows['output_order1']['runtime']
    report = {'version': 'state-extension-20261015-v1', 'source_binding': binding,
        'source_digest': data['source_digest'], 'model_digest': data['model_digest'],
        'runtime': runtime(), 'generation_budget': 3600, 'teacher_forward_budget': 4,
        'units': [], 'quality_passed': False}
    saved = {}
    try:
        for qid, seed in [(242, 124), (242, 127), (243, 125), (243, 128)]:
            apply_runtime(seed, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
            index = data['question_ids'].index(qid)
            selection = next(s for s in read(paths['selection'])['prompts'] if s['question_id'] == qid)
            targets, entry = frozen_targets(selection), selection['intervention_root']
            assert entry == (658 if qid == 242 else 600)
            ar = rows['AR']['raw']['evidence'][index]
            prompt = torch.tensor([ar['input_ids']], device=model.device)
            unit = {'qid': qid, 'seed': seed, 'entry': entry, 'targets': targets, 'baselines': [], 'cases': []}
            baseline, refs, inputs, cfgs = {}, {}, {}, {}
            for case in ('output_order0', 'output_order1'):
                refs[case] = rows[case]['raw']['evidence'][index]
                assert refs[case]['output_ids'] == ar['output_ids']
                inputs[case] = read(paths[case + ':assets/decode_inputs.json'])
                assert inputs[case]['prompts'][index]['input_ids'] == ar['input_ids']
                cfgs[case] = _convert(yaml.safe_load(paths[case + ':config.yaml'].read_text(encoding='utf-8'))['algorithm']['config']['eval'])
                assert cfgs[case]['temperature'] == 0 and cfgs[case]['max_new_tokens'] == 100
                visited = {t['root_position'] for t in refs[case]['stats']['trace']}
                samples, audit = observe(model, prompt, _convert(rows[case]['spec']), cfgs[case],
                    inputs[case]['eos_ids'], refs[case], sorted(set(targets) & visited), seed=seed)
                baseline[case] = samples
                unit['baselines'].append({'case': case, 'replay': audit})
                saved.update({f'q{qid}s{seed}:baseline:{case}:{p}': s for p, s in samples.items()})
            ids = torch.tensor([ar['output_ids']], device=model.device)
            teacher = model(model.embed(ids), torch.arange(ids.shape[1], device=model.device)[None], last_hidden_only=True)
            hidden = teacher.last_hidden_state[0].detach().cpu().clone()
            del teacher
            cos = [cosine(s['exact_root_hidden'], hidden[p]) for values in baseline.values() for p, s in values.items()]
            assert min(cos) >= .99999
            states = canonical_states(hidden, model.embedding_table, 1)
            saved[f'q{qid}s{seed}:canonical_states'] = states
            saved[f'q{qid}s{seed}:teacher_hidden'] = hidden
            unit.update(teacher_positions=len(ar['output_ids']), exact_root_cosine_min=min(cos))
            unit['cases'].append({'label': 'native_self', 'mode': 'native', 'graft': False,
                'replay': {'output_equal_independent_AR': True,
                    'stats': {k: refs['output_order1']['stats'][k] for k in SEMANTIC_KEYS},
                    'trace': [{k: t[k] for k in TRACE_KEYS} for t in refs['output_order1']['stats']['trace']],
                    'visited_targets': sorted(baseline['output_order1']), 'reused_baseline_generation': True},
                'state_replacements': []})
            for mode in ('native', 'anchor', 'velocity', 'both'):
                for graft in (False, True):
                    if mode == 'native' and not graft:
                        continue
                    donor_case = 'output_order0' if graft else 'output_order1'
                    kwargs = dict(prompt=prompt, spec=_convert(rows['output_order1']['spec']), eval_cfg=cfgs['output_order1'],
                        eos=inputs['output_order1']['eos_ids'], reference=refs['output_order1'],
                        original_masks=baseline['output_order1'][entry]['current_masks'], root=entry,
                        targets=targets, placebo=False, seed=seed, donor=baseline[donor_case][entry]['current_masks'][0, 1])
                    if mode == 'native':
                        samples, replay, _ = transfer(model, **kwargs)
                        records = []
                    elif mode == 'both':
                        samples, replay, _, records = synchronized_transfer(model, states, **kwargs)
                    else:
                        samples, replay, _, records = selective_transfer(model, states, mode, **kwargs)
                    for p, s in samples.items():
                        if p > entry and mode != 'native':
                            for coordinate in ('anchor', 'velocity') if mode == 'both' else (mode,):
                                assert torch.equal(s['current_state'][coordinate], states[p][coordinate])
                    label = mode + ('_graft' if graft else '_self')
                    saved.update({f'q{qid}s{seed}:{label}:{p}': s for p, s in samples.items()})
                    unit['cases'].append({'label': label, 'mode': mode, 'graft': graft,
                        'replay': replay, 'state_replacements': records})
            report['units'].append(unit)
        assert all(file_hash(p) == binding[k] for k, p in paths.items())
        assert all(file_hash(cache / n) == m['sha256'] for n, m in snapshot['files'].items())
        assert model_digest(model) == data['model_digest'] and manifest()['digest'] == data['source_digest']
        assert runtime() == report['runtime']
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if saved:
            torch.save(saved, args.out / 'selected_tensors.pt')
            report['raw_tensor_sha256'] = file_hash(args.out / 'selected_tensors.pt')
        (args.out / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'quality_passed': True, 'units': len(report['units'])}))


if __name__ == '__main__':
    main()
