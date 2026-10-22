"""Fixed root83 2x2 seed-mask direction/norm counterfactual (no generation).

    python tools/diagnose_mask_scale.py --study studies/output_space_small_matrix --diagnostic .tmp/output-windows-20261014-v1 --out .tmp/output-scale-<new-id>
"""

import argparse
import json
from pathlib import Path

from diagnose_output_windows import file_hash, margin


def crossed_masks(input_vector, output_vector):
    import torch

    vectors = {'input': input_vector.detach().float().cpu(), 'output': output_vector.detach().float().cpu()}
    norms = {name: vector.norm() for name, vector in vectors.items()}
    assert all(torch.isfinite(v).all() and norms[name] > 0 for name, v in vectors.items())
    return [{'direction': direction, 'norm_source': scale,
             'vector': (vector.clone() if direction == scale else vector * (norms[scale] / norms[direction])),
             'norm': float(norms[scale]), 'observed_combination': direction == scale}
            for direction, vector in vectors.items() for scale in norms]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--diagnostic', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new; previous evidence is retained')
    args.out.mkdir(parents=True)
    import torch
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.sources import manifest
    from spec_mtp.model import HFCausalLM

    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    study = args.study.resolve()
    data = read(study / 'docs/execution_v1.json')
    observer = read(args.diagnostic / 'diagnosis.json')
    assert observer['quality_passed'] and observer['windows'] == [83, 94, 97]
    assert file_hash(args.diagnostic / 'selected_tensors.pt') == observer['raw_tensor_sha256']
    assert file_hash(study / 'docs/execution_v1.json') == observer['source_binding']['matrix_sha256']
    assert manifest()['digest'] == data['source_digest'] == '670c744e0535dcad'
    binding = {'plan_sha256': file_hash((Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md')),
               'tool_sha256': file_hash(Path(__file__)),
               'helper_sha256': file_hash(Path(__file__).with_name('diagnose_output_windows.py')),
               'observer_sha256': file_hash(args.diagnostic / 'diagnosis.json'),
               'tensor_sha256': observer['raw_tensor_sha256'], 'source_digest': manifest()['digest']}
    report = {'version': 'output-space-q173-scale-20261014-v1', 'question_id': 173,
              'root_position': 83, 'kind': 'fixed 2x2 isolated-prefix counterfactual; no decoder replay',
              'source_binding': binding, 'forward_budget': 4, 'new_generated_tokens': 0,
              'cells': [], 'quality_passed': False}
    try:
        cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
        snapshot = read(study / 'docs/modelscope_snapshot.json')
        for name, metadata in snapshot['files'].items():
            assert file_hash(cache / name) == metadata['sha256']
        raw = torch.load(args.diagnostic / 'selected_tensors.pt', weights_only=True)
        input_vector = raw['frozen_mean_chain:83']['seed_mask_inputs'][0]
        output_vector = raw['output_order0:83']['seed_mask_inputs'][0]
        assert torch.equal(output_vector, raw['output_order1:83']['seed_mask_inputs'][0])
        cells = crossed_masks(input_vector, output_vector)
        rows = {r['case']: r for r in data['runs']}
        ar = rows['AR']['raw']['evidence'][1]
        torch.set_num_threads(1)
        apply_runtime(124, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': False}))
        model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                          torch_dtype='float32', attn_implementation='eager').to('cuda')
        assert model_digest(model) == data['model_digest'] and runtime() == rows['output_order1']['runtime']
        prefix_ids = torch.tensor([ar['output_ids'][:83]], device=model.device)
        target = int(ar['output_ids'][84])
        for cell in cells:
            vector = cell['vector'].to(model.device)
            inputs = torch.cat([model.embed(prefix_ids), vector[None, None]], dim=1)
            out = model(inputs, torch.arange(84, device=model.device)[None])
            observed = margin(out.logits[0, -1], target)
            result = {key: value for key, value in cell.items() if key != 'vector'}
            result['first_slot'] = observed
            report['cells'].append(result)
            if cell['observed_combination']:
                case = 'frozen_mean_chain' if cell['direction'] == 'input' else 'output_order0'
                expected = next(c for c in observer['cases'] if c['case'] == case)['windows'][0]['isolated_first_slot']
                assert observed['top1_id'] == expected['top1_id']
                assert abs(observed['target_margin_vs_best_other'] - expected['target_margin_vs_best_other']) <= 1e-3
        assert len(report['cells']) == 4 and model_digest(model) == data['model_digest']
        assert runtime() == rows['output_order1']['runtime'] and manifest()['digest'] == binding['source_digest']
        for name, metadata in snapshot['files'].items():
            assert file_hash(cache / name) == metadata['sha256']
        assert file_hash((Path(__file__).resolve().parents[1] / 'docs/研究计划 Denoising Speculative Decoding.md')) == binding['plan_sha256']
        report['model_digest'] = data['model_digest']
        report['runtime'] = runtime()
        report['forward_calls'] = 4
        report['quality_passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.out / 'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False) + '\n',encoding='utf-8')
    print(json.dumps(report['cells']))


if __name__ == '__main__':
    main()
