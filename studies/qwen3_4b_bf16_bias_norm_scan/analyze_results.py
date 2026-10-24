"""CPU-only native readout of eight norm cells and the five reused endpoints."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BASE = ROOT / 'studies/qwen3_4b_bf16_five_methods'
sys.path.insert(0, str(ROOT / 'tools'))
import analyze_4b_full_directions as directions  # noqa: E402
_spec = importlib.util.spec_from_file_location('norm_pair_readout', BASE / 'analyze_results.py')
_pair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pair)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def depth_rows(raw, prompts):
    assert len(raw) == len(prompts)
    rows = []
    for item, prompt in zip(raw, prompts):
        assert item['input_ids'] == prompt['input_ids']
        stats = item['stats']
        assert stats['prefill_calls'] == 1 and len(stats['trace']) == stats['decode_calls']
        prefix = item['output_ids'][:len(prompt['input_ids']) + 1]
        assert len(prefix) == len(prompt['input_ids']) + 1
        rows.append({'question_id': prompt['question_id'],
            'initial_prefix_sha256': hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            'attempt_by_depth': stats['attempt_by_depth'], 'accept_by_depth': stats['accept_by_depth'],
            'first_block_accepted': len(stats['trace'][0]['accepted_nodes']) if stats['trace'] else None})
    return rows


def first_block_compare(first, second):
    assert len(first) == len(second)
    result = {'total': len(first), 'same_prefix_with_block': 0, 'different_prefix': 0,
              'missing_block': 0, 'accepted': {'win': 0, 'tie': 0, 'loss': 0}}
    for a, b in zip(first, second):
        assert a['question_id'] == b['question_id']
        if a['first_block_accepted'] is None or b['first_block_accepted'] is None:
            result['missing_block'] += 1
        elif a['initial_prefix_sha256'] != b['initial_prefix_sha256']:
            result['different_prefix'] += 1
        else:
            result['same_prefix_with_block'] += 1
            delta = b['first_block_accepted'] - a['first_block_accepted']
            result['accepted']['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    assert sum(result['accepted'].values()) == result['same_prefix_with_block']
    return result


def plot(cells, means, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [0., .5, 1., 2.]
    zero = cells['zero/0']['totals']['pooled_tau']
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    try:
        for mode in ('template', 'random0', 'random1', 'random2'):
            y = [zero] + [cells[f'{mode}/{scale:g}']['totals']['pooled_tau'] for scale in x[1:]]
            ax.plot(x, y, marker='o', label=mode)
        ax.plot(x, [zero] + [means[f'{scale:g}'] for scale in x[1:]], 'k--', label='three-random mean')
        ax.set(xlabel='Native step_scale / original direction norm multiplier',
               ylabel='Pooled committed / decode calls', xticks=x, title='Fixed480, Qwen3-4B BF16/T0, BC30')
        ax.legend()
        fig.supxlabel('0 and1 reuse frozen evidence; only0.5 and2 are new. Output lengths may differ; no speed claim.')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)

def load_reused(data_root):
    from studies._shared.structure import validation
    data_root = Path(data_root).resolve()
    formal = read(BASE / 'docs/paired_evidence_v1.json')
    original = read(ROOT / 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json')
    prompts, snapshot = (read(BASE / 'docs' / n) for n in ('prompts_manifest.json', 'modelscope_snapshot.json'))
    old_ar = formal['cells']['AR']
    arpath = data_root / 'studies/qwen3_4b_bf16_five_methods/runs' / old_ar['run_id'] / 'assets/decode_observations.json'
    assert sha(arpath) == old_ar['files_sha256']['assets/decode_observations.json']
    raw_ar = read(arpath)['out']['evidence']
    assert len(raw_ar) == len(prompts['rendered_prompts']) == 480
    assert all(item['input_ids'] == prompt['input_ids'] for item, prompt in zip(raw_ar, prompts['rendered_prompts']))
    reference = [{'question_id': p['question_id'], 'output_ids': r['output_ids'][len(p['input_ids']):],
                  'model_calls': r['stats']['ar_calls']} for p, r in zip(prompts['rendered_prompts'], raw_ar)]
    assert len(reference) == len(raw_ar) == 480
    assert sum(len(r['output_ids']) for r in reference) == old_ar['totals']['new_tokens']
    assert sum(r['model_calls'] for r in reference) == old_ar['totals']['model_calls']
    cells = {'zero/0': dict(original['cells']['zero'], reused=True)}
    for mode in ('template', 'random0', 'random1', 'random2'):
        cells[mode + '/1'] = dict(original['cells'][mode], reused=True)
    for cell in cells.values():
        assert cell['contract']['dtype'] == 'torch.bfloat16'
        assert cell['model_digest'] == formal['cells']['30/input_first']['model_digest']
        setting = cell['actual_settings']
        assert setting['spec']['num_masks'] == 2 and setting['spec']['block_complexity'] == 30
        assert setting['eval']['temperature'] == 0 and setting['eval']['max_new_tokens'] == 256
        owner = data_root / cell.get('study_path', 'studies/qwen3_4b_bf16_five_methods')
        path = owner / 'runs' / cell['run_id'] / 'assets/decode_observations.json'
        for name, expected in cell['files_sha256'].items():
            assert sha(owner / 'runs' / cell['run_id'] / name) == expected, name
        raw = read(path)['out']['evidence']
        assert len(raw) == len(cell['rows']) == len(prompts['rendered_prompts']) == 480
        checked = 0
        for item, row, prompt in zip(raw, cell['rows'], prompts['rendered_prompts']):
            assert row['question_id'] == prompt['question_id'] and item['input_ids'] == prompt['input_ids']
            assert all(item['stats'][k] == row[k] for k in ('accepted', 'committed', 'decode_calls', 'new_tokens'))
            checked += validation._check_argmax(item)
            assert row['model_calls'] == item['stats']['decode_calls'] + item['stats']['prefill_calls']
        totals = _pair.summarize(cell['rows'], False)
        assert all(cell['totals'][k] == value for k, value in totals.items())
        assert checked == cell['totals']['argmax_checked_tokens']
        cell.update(raw_argmax_checked_tokens=checked, raw_files_rechecked=len(cell['files_sha256']))
        cell['depth_rows'] = depth_rows(raw, prompts['rendered_prompts'])
    return formal, original, prompts, snapshot, reference, cells


def matrix_settings(decl):
    from rpipe.structure.control.merge import deep_merge
    common = yaml.safe_load((STUDY / 'experiment_config.yaml').read_text(encoding='utf-8'))['algorithm']['config']
    return [deep_merge(common, setting) for setting in decl['axes']['algorithm.config']]


def check_matrix(decl, entries):
    expected = decl['axes']['algorithm.config']
    assert [(c['direction'], c['spec']['ema']['step_scale']) for c in expected] == [
        (mode, scale) for mode in ('template', 'random0', 'random1', 'random2') for scale in (.5, 2.)]
    assert [e['factors']['algorithm.config'] for e in entries] == matrix_settings(decl), 'Expected all eight frozen merged norm configurations in order'
    assert all(len(e['runs']) == 1 for e in entries)


def verify_reused(data_root, output_dir=None):
    from norm_control import check_refs
    from studies._shared.structure import validation
    check_refs()
    formal, _, _, _, reference, cells = load_reused(data_root)
    comparisons = []
    for base in ('zero/0', 'template/1'):
        for name, cell in cells.items():
            if name == base:
                continue
            pair = directions.compare(cells[base]['rows'], cell['rows'])
            pair.pop('rows')
            comparisons.append(dict(first=base, second=name,
                first_block_same_prefix=first_block_compare(cells[base]['depth_rows'], cell['depth_rows']), **pair))
    decl = yaml.safe_load((STUDY / 'study.yaml').read_text(encoding='utf-8'))
    expected = decl['axes']['algorithm.config']
    check_matrix(decl, [{'factors': {'algorithm.config': setting}, 'runs': [{}]} for setting in matrix_settings(decl)])
    body = {'complete': False, 'reused_only': True, 'new_cells_inspected': False, 'fixed_prompts': 480,
        'data_root': str(Path(data_root).resolve()), 'no_timing': True, 'no_model_forward': True,
        'scope': 'Hash and raw argmax recheck of five existing endpoints; no new native Run acceptance or curve',
        'expected_new_cases': [{'direction': c['direction'], 'step_scale': c['spec']['ema']['step_scale']} for c in expected],
        'shared_AR_reference': {'study_path': 'studies/qwen3_4b_bf16_five_methods',
            'run_id': formal['cells']['AR']['run_id'],
            'raw_sha256': formal['cells']['AR']['files_sha256']['assets/decode_observations.json']},
        'shared_AR_reference_tokens': sum(len(row['output_ids']) for row in reference),
        'raw_argmax_checked_tokens': sum(c['raw_argmax_checked_tokens'] for c in cells.values()),
        'source_refs_sha256': sha(STUDY / 'docs/source_refs.json'), 'analyzer_sha256': sha(Path(__file__)),
        'raw_math_checker_sha256': sha(Path(validation.__file__)),
        'formal_sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in
            (BASE / 'docs/paired_evidence_v1.json', ROOT / 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json')},
        'cells': {name: dict({k: cell[k] for k in ('run_id', 'totals', 'raw_argmax_checked_tokens', 'raw_files_rechecked')},
                            study_path=cell.get('study_path', 'studies/qwen3_4b_bf16_five_methods'),
                            files_sha256=cell['files_sha256'])
                  for name, cell in cells.items()}, 'paired_comparisons': comparisons}
    if output_dir is not None:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / 'reused_readiness_v1.json').write_text(json.dumps(body, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return body


def continuation_entries(study, decl, entries, continuation):
    """Map only unexecuted old cells to identical scientific configurations."""
    from rpipe.structure.control.merge import deep_merge
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    from studies._shared.structure.sources import check_frozen, manifest
    new_decl = yaml.safe_load((continuation / 'study.yaml').read_text(encoding='utf-8'))
    assert new_decl['axes'] == {'algorithm.config': decl['axes']['algorithm.config'][1:]}
    old_common = yaml.safe_load((study / 'experiment_config.yaml').read_text(encoding='utf-8'))
    new_common = yaml.safe_load((continuation / 'experiment_config.yaml').read_text(encoding='utf-8'))
    scientific = dict(new_common, experiment=old_common['experiment'])
    scientific['algorithm'] = dict(new_common['algorithm'], config=dict(new_common['algorithm']['config']))
    scientific['algorithm']['config']['version'] = old_common['algorithm']['config']['version']
    assert scientific == old_common, 'Continuation changed scientific settings'
    check_frozen(continuation, new_decl, manifest())
    pcheck(continuation, new_decl)
    newer = read(continuation / 'index.json')['experiments']
    expected = [deep_merge(new_common['algorithm']['config'], a) for a in new_decl['axes']['algorithm.config']]
    assert [e['factors']['algorithm.config'] for e in newer] == expected
    assert len(newer) == 7 and all(len(e['runs']) == 1 for e in newer)
    assert len({e['runs'][0]['id'] for e in newer}) == 7
    for old in entries[1:]:
        folder = study / 'runs' / old['runs'][0]['id']
        assert not (folder / 'result.json').exists() and not (folder / 'assets/decode_observations.json').exists()
    first = study / 'runs' / entries[0]['runs'][0]['id']
    result = read(first / 'result.json')
    recovery = result['spec_mtp']['revalidation']
    assert result['status'] == 'succeeded' and recovery['execute_rerun'] is False
    return [(study, entries[0]), *[(continuation, e) for e in newer]]


def main(data_root=None, study_dir=None, output_dir=None, continuation_study=None):
    from norm_control import check_refs
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    from rpipe.structure.make.expand import load_study_yaml
    from studies._shared.structure.sources import check_frozen, manifest
    data_root = ROOT if data_root is None else Path(data_root).resolve()
    study = STUDY if study_dir is None else Path(study_dir).resolve()
    output = study / 'docs' if output_dir is None else Path(output_dir).resolve()
    assert Path.cwd().resolve() == study.parents[1], 'Run from the frozen execution checkout root for declared relative inputs'
    check_refs()
    decl = load_study_yaml(study)
    check_frozen(study, decl, manifest())
    pcheck(study, decl)
    formal, original, prompts, snapshot, reference, cells = load_reused(data_root)
    entries = read(study / 'index.json')['experiments']
    check_matrix(decl, entries)
    continuation = Path(continuation_study).resolve() if continuation_study else None
    selected = continuation_entries(study, decl, entries, continuation) if continuation else [(study, e) for e in entries]
    for cell_study, exp in selected:
        setting = exp['factors']['algorithm.config']
        mode, scale = setting['direction'], setting['spec']['ema']['step_scale']
        name = f'{mode}/{scale:g}'
        assert name not in cells and len(exp['runs']) == 1
        folder = cell_study / 'runs' / exp['runs'][0]['id']
        result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
        cells[name] = {'run_id': folder.name, 'study_path': str(cell_study), 'native_status': result.get('status', 'pending')}
        if result.get('status') != 'succeeded':
            continue
        cfg = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
        assert cfg['algorithm']['config']['spec'] == setting['spec']
        assert cfg['seed'] == 123 and cfg['algorithm']['config']['direction'] == mode
        assert cfg['algorithm']['config']['spec']['ema']['step_scale'] == scale
        cell, _ = _pair.load_cell(cell_study, folder, cfg, prompts['rendered_prompts'], prompts, snapshot, reference)
        assert cell['model_digest'] == formal['cells']['30/input_first']['model_digest']
        assert cell['runtime'] == formal['cells']['30/input_first']['runtime']
        progress = read(folder / 'assets/norm_progress.json')
        assert progress['completed'] == len(progress['rows']) == 480
        assert progress['direction'] == mode and progress['step_scale'] == scale
        for p, r, canonical in zip(progress['rows'], cell['rows'], prompts['rendered_prompts']):
            assert p['question_id'] == r['question_id'] == canonical['question_id']
            assert all(p[k] == r[k] for k in ('accepted', 'committed', 'decode_calls', 'new_tokens', 'argmax_checked_tokens'))
            assert p['bias']['step_scale'] == scale and p['bias']['full_prompt_mean_preserved']
        cell.update(study_path=str(cell_study), revalidation=result.get('spec_mtp', {}).get('revalidation'), progress_sha256=sha(folder / 'assets/norm_progress.json'),
            bias_and_depth_rows=[{k: v for k, v in r.items() if k != 'gpu_admission'} for r in progress['rows']])
        cell['depth_rows'] = depth_rows(read(folder / 'assets/decode_observations.json')['out']['evidence'], prompts['rendered_prompts'])
        cells[name] = cell
    ready = all('rows' in cell for cell in cells.values())
    pairs = []
    for name, cell in cells.items():
        if 'rows' not in cell:
            continue
        cell['totals']['category_mean_pooled_tau'] = directions.category_mean(cell['rows'])
        if name == 'zero/0':
            continue
        mode = name.split('/')[0]
        for base in ('zero/0', mode + '/1'):
            pairs.append(dict(first=base, second=name,
                first_block_same_prefix=first_block_compare(cells[base]['depth_rows'], cell['depth_rows']),
                **directions.compare(cells[base]['rows'], cell['rows'])))
    means = {}
    for scale in (.5, 1., 2.):
        rows = [cells[f'random{i}/{scale:g}'] for i in range(3)]
        if all('rows' in c for c in rows):
            means[f'{scale:g}'] = sum(c['totals']['pooled_tau'] for c in rows) / 3
    if ready:
        native = read((continuation or study) / 'spec_mtp_study_check.json')
        process = read((continuation or study) / 'process.json')
        assert native['passed'] and native['complete'] and process['complete'] and process['spec_mtp']['passed']
    body = {'complete': ready, 'author_official': False, 'no_timing': True, 'fixed_prompts': 480,
        'budget': 30, 'mask_count': 2, 'new_cells': 8, 'reused_cells': 5,
        'continuation_study': str(continuation) if continuation else None,
        'original_study_fully_complete': ready and continuation is None,
        'source_refs_sha256': sha(study / 'docs/source_refs.json'), 'analyzer_sha256': sha(Path(__file__)),
        'paired_formal_sha256': sha(BASE / 'docs/paired_evidence_v1.json'),
        'direction_formal_sha256': sha(ROOT / 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json'),
        'cells': cells, 'paired_comparisons': pairs, 'random_three_seed_mean_pooled_tau': means}
    output.mkdir(parents=True, exist_ok=True)
    target = output / ('norm_evidence_v1.json' if ready else 'norm_pending.json')
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if ready and target.exists():
        assert target.read_text(encoding='utf-8') == text
    else:
        target.write_text(text, encoding='utf-8')
    if ready:
        plot(cells, means, output / 'figures/bc30_norm_scan.png')
    print(json.dumps({'complete': ready, 'valid_cells': sum('rows' in c for c in cells.values())}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT, help='Checkout containing the original immutable 4B Runs')
    parser.add_argument('--study-dir', type=Path, help='Frozen native norm Study, after the 8B queue exits')
    parser.add_argument('--output-dir', type=Path, help='Separate CPU output directory')
    parser.add_argument('--continuation-study', type=Path, help='Seven-cell operational continuation; first original cell retained')
    parser.add_argument('--check-reused', action='store_true', help='Read-only old endpoints; never make/accept new cells or plot')
    args = parser.parse_args()
    if args.check_reused:
        proof = verify_reused(args.data_root, args.output_dir)
        print(json.dumps({'complete': proof['complete'], 'reused_cells': len(proof['cells']),
                          'raw_argmax_checked_tokens': proof['raw_argmax_checked_tokens'], 'new_cells_inspected': False}))
    else:
        main(args.data_root, args.study_dir, args.output_dir, args.continuation_study)
