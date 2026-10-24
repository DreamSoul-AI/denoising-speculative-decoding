"""Nonzero-temperature full-path audit and conditional source719 stochastic acceptance.

    python tools/diagnose_sampling_paths.py --out .tmp/sampling-paths-<new-id>
"""

from spec_mtp.masks import build_mask_provider

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_output_windows import file_hash, observe
from diagnose_proxy_transfer import transfer
from diagnose_extension import choose_window
from diagnose_state_extension import frozen_targets


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
    return {'temperature': temperature, 'seeds': list(seeds), 'verifier_token_ids': draws,
        'verifier_draws': len(draws), 'RNG_consumed': True, 'same_verifier_draws_for_all_cells': True,
        'cells': [{'label': label, 'candidate': token, 'exact_candidate_probability': float(probs[token]),
                   'first_slot_hits': sum(draw == token for draw in draws)} for label, token in candidates.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    args.out.mkdir(parents=True)
    import torch
    import yaml
    from spec_mtp.decode import SpecDecoder, ar_generate
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM
    from rpipe.structure.system.runtime import apply_runtime
    from rpipe.structure.system.config import SystemConfig
    root = Path(__file__).resolve().parents[1]
    study = root / 'studies/output_space_cross_task'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    paths = {'tool': Path(__file__), 'plan': (Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md'),
        'matrix': study / 'docs/execution_v1.json', 'snapshot': study / 'docs/modelscope_snapshot.json',
        'local_report': root / '.tmp/q241-local-factors-20261015-v1/diagnosis.json',
        'local_tensor': root / '.tmp/q241-local-factors-20261015-v1/selected_tensors.pt'}
    for name in ('diagnose_output_windows.py', 'diagnose_proxy_transfer.py', 'diagnose_extension.py',
                 'diagnose_state_extension.py'):
        paths[name] = root / 'tools' / name
    native = read(paths['matrix'])
    rows = {r['case']: r for r in native['runs']}
    for case in ('output_order0', 'output_order1'):
        for name in ('config.yaml', 'assets/decode_inputs.json'):
            paths[case + ':' + name] = study / 'runs' / rows[case]['run_id'] / name
    binding = {k: file_hash(p) for k, p in paths.items()}
    local = read(paths['local_report'])
    assert local['quality_passed'] and binding['local_tensor'] == local['raw_tensor_sha256']
    assert manifest()['digest'] == native['source_digest'] == '670c744e0535dcad'
    snapshot = read(paths['snapshot'])
    cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
    assert all(file_hash(cache / k) == v['sha256'] for k, v in snapshot['files'].items())
    torch.set_num_threads(1)
    apply_runtime(123, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
    model = HFCausalLM(name_or_path=str(cache), local_files_only=True, torch_dtype='float32',
        attn_implementation='eager').to('cuda')
    assert model_digest(model) == native['model_digest'] and runtime() == rows['output_order1']['runtime']
    report = {'version': 'sampling-paths-20261015-v1', 'source_binding': binding,
        'source_digest': native['source_digest'], 'model_digest': native['model_digest'], 'runtime': runtime(),
        'generation_budget': 1400, 'teacher_forwards': 0, 'units': [], 'conditional': [],
        'quality_passed': False}
    saved = {}
    try:
        old = torch.load(paths['local_tensor'], weights_only=True, map_location='cpu', mmap=True)
        candidates = {f'S{s}P{p}': int(old[f'r719:S{s}P{p}']['logits'][0, 3].argmax())
                      for s in (0, 1) for p in (0, 1)}
        for temperature in (.8, 1.):
            report['conditional'].append(conditional_acceptance(old['frame:720']['logits'][0, 0].cuda(),
                candidates, temperature, range(200, 264)))
        input_ids = rows['AR']['raw']['evidence'][0]['input_ids']
        prompt = torch.tensor([input_ids], device=model.device)
        configs = {}
        inputs = {}
        for case in ('output_order0', 'output_order1'):
            inputs[case] = read(paths[case + ':assets/decode_inputs.json'])
            assert inputs[case]['prompts'][0]['input_ids'] == input_ids
            cfg = yaml.safe_load(paths[case + ':config.yaml'].read_text(encoding='utf-8'))['algorithm']['config']['eval']
            cfg['temperature'] = .8
            configs[case] = _convert(cfg)
            assert configs[case]['max_new_tokens'] == 100
        for seed in (123, 124):
            apply_runtime(seed, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
            gen = lambda: torch.Generator(device=model.device).manual_seed(seed)
            unit = {'qid': 241, 'seed': seed, 'temperature': .8, 'native': [], 'observed': [], 'cases': []}
            report['units'].append(unit)
            ar, ar_stats = ar_generate(model, prompt, configs['output_order1'],
                eos_ids=inputs['output_order1']['eos_ids'], generator=gen())
            unit['AR'] = {'input_ids': input_ids, 'output_ids': ar[0].tolist(), 'stats': ar_stats}
            refs = {}
            for case in ('output_order0', 'output_order1'):
                output, stats = SpecDecoder(model, _convert(rows[case]['spec']), configs[case],
                    eos_ids=inputs[case]['eos_ids'], generator=gen(), mask_policy=build_mask_provider(_convert(rows[case]['spec']), model.embedding_table)).generate(prompt)
                record = {'case': case, 'input_ids': input_ids, 'output_ids': output[0].tolist(),
                    'stats': stats, 'output_equal_independent_sampled_AR': torch.equal(output, ar)}
                unit['native'].append(record)
                assert record['output_equal_independent_sampled_AR'] and not stats['hit_context_limit']
                refs[case] = record
            selection = choose_window(refs['output_order0']['stats']['trace'], refs['output_order1']['stats']['trace'])
            unit['selection'] = selection
            path = args.out / f'selection-seed{seed}.json'
            assert not path.exists()
            path.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
            unit['selection_sha256'] = file_hash(path)
            entry = selection['intervention_root']
            if entry is None:
                unit['status'] = 'infeasible; retained, no replacement'
                continue
            targets = frozen_targets(selection)
            unit['targets'] = targets
            baseline = {}
            for case in ('output_order0', 'output_order1'):
                visited = {t['root_position'] for t in refs[case]['stats']['trace']}
                samples, audit = observe(model, prompt, _convert(rows[case]['spec']), configs[case],
                    inputs[case]['eos_ids'], refs[case], sorted(set(targets) & visited), seed=seed)
                baseline[case] = samples
                unit['observed'].append({'case': case, 'replay': audit})
                saved.update({f's{seed}:baseline:{case}:{p}': v for p, v in samples.items()})
            for graft in (False, True):
                donor_case = 'output_order0' if graft else 'output_order1'
                samples, replay, _ = transfer(model, prompt=prompt, spec=_convert(rows['output_order1']['spec']),
                    eval_cfg=configs['output_order1'], eos=inputs['output_order1']['eos_ids'],
                    reference=refs['output_order1'], donor=baseline[donor_case][entry]['current_masks'][0, 1],
                    original_masks=baseline['output_order1'][entry]['current_masks'], root=entry,
                    targets=targets, placebo=not graft, seed=seed)
                label = 'graft' if graft else 'self'
                saved.update({f's{seed}:{label}:{p}': v for p, v in samples.items()})
                unit['cases'].append({'label': label, 'replay': replay})
            assert file_hash(path) == unit['selection_sha256']
        assert all(file_hash(p) == binding[k] for k, p in paths.items())
        assert all(file_hash(cache / k) == v['sha256'] for k, v in snapshot['files'].items())
        assert model_digest(model) == native['model_digest'] and manifest()['digest'] == native['source_digest']
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
    print(json.dumps({'quality_passed': True, 'sampled_units': len(report['units']), 'conditional_draws': 128}))


if __name__ == '__main__':
    main()
