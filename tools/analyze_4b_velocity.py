"""Decompose saved mask velocities and seed-depth transitions on CPU.

    python tools/analyze_4b_velocity.py --study studies/qwen3_4b_bf16_five_methods --capture docs/research/4b-mask-geometry-observations-2026-10-15.json --vectors .tmp/4b-geometry-capture-v5/vectors.pt --out .tmp/4b-velocity-v1.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

def velocity_geometry(anchor, velocity):
    a, v = anchor.float(), velocity.float()
    assert a.norm() > 0 and v.norm() > 0 and (a + v).norm() > 0
    radial = a * (v @ a) / (a @ a)
    tangent = v - radial
    normalized_move = (a + v) / (a + v).norm() - a / a.norm()
    return {'velocity_over_anchor': float(v.norm() / a.norm()),
            'radial_signed_over_velocity': float((v @ a) / (a.norm() * v.norm())),
            'tangent_over_velocity': float(tangent.norm() / v.norm()),
            'normalized_movement_retention': float(normalized_move.norm() / (v.norm() / a.norm())),
            'raw_slot_norm_ratio': float((a + v).norm() / a.norm())}


def transition_counts(evidence):
    from analyze_4b_geometry import trace_metrics
    result = {str(d): {'transitions': 0, 'next_accept_d1': 0, 'next_accept_d2': 0} for d in (0, 1, 2)}
    for e in evidence:
        trace_metrics(e, len(e['input_ids']))
        for previous, current in zip(e['stats']['trace'], e['stats']['trace'][1:]):
            assert not previous['stopped']
            cell = result[str(len(previous['accepted_nodes']))]
            cell['transitions'] += 1
            cell['next_accept_d1'] += bool(current['accepted_nodes'])
            cell['next_accept_d2'] += len(current['accepted_nodes']) == 2
    for cell in result.values():
        n = cell['transitions']
        cell['next_accept_d1_rate'] = cell['next_accept_d1'] / n if n else None
        cell['next_accept_d2_rate'] = cell['next_accept_d2'] / n if n else None
    return result


def template_metrics(level, velocity, direction, masks):
    fixed = torch.stack([level + i * direction for i in (1, 2)])
    cosine = torch.nn.functional.cosine_similarity
    return {'velocity_initial_cos': float(cosine(velocity.float(), direction.float(), dim=0)),
            'velocity_initial_norm_ratio': float(velocity.float().norm() / direction.float().norm()),
            'velocity_initial_relative_error': float((velocity.float() - direction.float()).norm() / direction.float().norm()),
            'constant_slot0_cos': float(cosine(fixed[0].float(), masks[0].float(), dim=0)),
            'constant_slot1_cos': float(cosine(fixed[1].float(), masks[1].float(), dim=0)),
            'constant_slot0_relative_error': float((fixed[0].float() - masks[0].float()).norm() / masks[0].float().norm()),
            'constant_slot1_relative_error': float((fixed[1].float() - masks[1].float()).norm() / masks[1].float().norm())}


def main():
    from analyze_4b_geometry import distribution, sha
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--vectors', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists()
    capture = json.loads(args.capture.read_text())
    assert capture['quality_passed'] and sha(args.vectors) == capture['raw_vectors_sha256']
    vectors = torch.load(args.vectors, map_location='cpu', weights_only=True)
    report = {'analyzer_sha256': sha(__file__), 'capture_sha256': sha(args.capture),
              'vectors_sha256': sha(args.vectors), 'no_model_forward': True, 'velocity': {}, 'transitions': {}}
    for method in ('input_first', 'output_first'):
        points = []
        for key, case in vectors.items():
            if not key.endswith('/' + method):
                continue
            qid = int(key.split('/')[0])
            length = next(c['new_tokens'] for c in capture['cases'] if c['question_id'] == qid and c['case'] == method)
            for frame in case['frames']:
                for i, depth in enumerate(frame['depths']):
                    if frame['root_index'] + depth + 2 >= length:
                        continue
                    a, b = frame['raw_predictions'][i].float()
                    point = velocity_geometry(a, b - a)
                    point.update(question_id=qid, root_index=frame['root_index'], depth=depth)
                    points.append(point)
        metrics = ('velocity_over_anchor', 'radial_signed_over_velocity', 'tangent_over_velocity',
                   'normalized_movement_retention', 'raw_slot_norm_ratio')
        report['velocity'][method] = {'points': points, 'distribution': {
            k: distribution([p[k] for p in points]) for k in metrics}}
    paired = json.loads((args.study / 'docs/paired_evidence_v1.json').read_text())
    prompts = json.loads((args.study / 'docs/prompts_manifest.json').read_text())['rendered_prompts']
    prefixes = Counter(tuple(p['input_ids'][:2]) for p in prompts)
    assert len(prefixes) == 1
    first, second = next(iter(prefixes))
    cache = args.study / 'shared/model/modelscope/Qwen/Qwen3-4B'
    snapshot = json.loads((args.study / 'docs/modelscope_snapshot.json').read_text())
    index = cache / 'model.safetensors.index.json'
    expected = {f['path']: f['sha256'] for f in snapshot['files']}
    assert sha(index) == expected[index.name]
    shard = json.loads(index.read_text())['weight_map']['model.embed_tokens.weight']
    assert sha(cache / shard) == expected[shard]
    from safetensors import safe_open
    with safe_open(cache / shard, framework='pt', device='cpu') as handle:
        table = handle.get_slice('model.embed_tokens.weight')
        direction = table[second:second + 1][0] - table[first:first + 1][0]
    points = []
    for key, case in vectors.items():
        if key.endswith('/input_first'):
            for frame in case['frames']:
                point = template_metrics(frame['input_level'], frame['input_velocity'], direction, frame['masks'][0])
                point.update(question_id=int(key.split('/')[0]), root_index=frame['root_index'])
                points.append(point)
    report['template_control'] = {'prompt_count': len(prompts), 'unique_first_two_pairs': len(prefixes),
        'embedding_shard_sha256': expected[shard], 'points': points,
        'distribution': {k: distribution([p[k] for p in points]) for k in points[0] if k not in ('question_id', 'root_index')}}
    for bc in (30, 60):
        for method in ('input_first', 'output_zero', 'output_first'):
            cell = paired['cells'][f'{bc}/{method}']
            path = args.study / 'runs' / cell['run_id'] / 'assets/decode_observations.json'
            assert sha(path) == cell['files_sha256']['assets/decode_observations.json']
            raw = json.loads(path.read_text())
            counts = transition_counts(raw['out']['evidence'])
            report['transitions'][f'{bc}/{method}'] = {'run_id': cell['run_id'], 'raw_sha256': sha(path),
                'counts': counts, 'total': sum(c['transitions'] for c in counts.values())}
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS: saved vectors and six full480 Run hashes/seed-depth transitions; zero model forwards')


if __name__ == '__main__':
    main()
