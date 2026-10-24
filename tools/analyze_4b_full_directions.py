"""Read native full480 direction Runs and reuse the original five-method readout.

    python tools/analyze_4b_full_directions.py
"""
import hashlib
import importlib.util
import ast
import json
from pathlib import Path
import sys

import torch  # noqa: F401 - Windows torch-first
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'studies/qwen3_4b_bf16_five_methods'
STUDY = ROOT / 'studies/qwen3_4b_bf16_directions'
RECOVERY = ROOT / 'studies/qwen3_4b_bf16_directions_recovery'
sys.path.insert(0, str(BASE))
_spec = importlib.util.spec_from_file_location('original_five_method_readout', BASE / 'analyze_results.py')
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
load_cell, compare_rows = _original.load_cell, _original.compare_rows

NAMES = ('template', 'random0', 'random1', 'random2', 'zero', 'content')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category_mean(rows):
    categories = sorted({r['category'] for r in rows})
    return sum(sum(r['committed'] for r in rows if r['category'] == c) /
               max(sum(r['decode_calls'] for r in rows if r['category'] == c), 1)
               for c in categories) / len(categories)


def compare(first, second):
    result = compare_rows(first, second)
    for a, b, r in zip(first, second, result['rows']):
        diff = b['committed'] * max(a['decode_calls'], 1) - a['committed'] * max(b['decode_calls'], 1)
        r['tau'] = 'win' if diff > 0 else 'loss' if diff < 0 else 'tie'
    result['tau'] = {s: sum(r['tau'] == s for r in result['rows']) for s in ('win', 'tie', 'loss')}
    return result


def require_resource_failure(previous, error):
    assert previous['native_status'] == 'failed', 'Never replace a completed or running factor'
    assert any(text in error for text in ('Foreign GPU work appeared', 'Nonpreemptive GPU admission deferred')), error


def control_source(path):
    source = path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'control')
    return ast.get_source_segment(source, node)


def plot(body, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert body['complete']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    try:
        values = [body['cells'][n]['totals']['pooled_tau'] for n in NAMES]
        axes[0].barh(NAMES, values)
        for i, value in enumerate(values):
            axes[0].text(value, i, f' {value:.6f}', va='center')
        axes[0].set_xlim(0, max(values) * 1.16)
        axes[0].set_xlabel('Pooled decode committed / decode calls')
        axes[0].set_title('BC30, 480 original prompts')
        names = [p['second'] for p in body['vs_template']]
        left = [0] * len(names)
        for key, color in [('win', '#3b8c6e'), ('tie', '#bbbbbb'), ('loss', '#c16b64')]:
            count = [p['accepted'][key] for p in body['vs_template']]
            axes[1].barh(names, count, left=left, label=key, color=color)
            left = [a + b for a, b in zip(left, count)]
        assert left == [480] * len(names)
        axes[1].set_xlabel('Prompts: accepted W/T/L versus fixed e2-e1')
        axes[1].legend()
        fig.supxlabel('Fixed data, BF16/T0. Different output paths/lengths; counts are not timing or significance.')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main():
    from studies._shared.structure.sources import check_frozen, manifest as sources
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    from rpipe.structure.make.expand import load_study_yaml
    owners = [STUDY]
    if (RECOVERY / 'index.json').exists():
        owners.append(RECOVERY)
        assert sha(STUDY / 'docs/source_refs.json') == sha(RECOVERY / 'docs/source_refs.json')
        assert control_source(STUDY / 'direction_control.py') == control_source(RECOVERY / 'direction_control.py')
    for owner in owners:
        decl = load_study_yaml(owner)
        check_frozen(owner, decl, sources())
        pcheck(owner, decl)
    original = read(BASE / 'docs/paired_evidence_v1.json')
    prompts = read(BASE / 'docs/prompts_manifest.json')
    snapshot = read(BASE / 'docs/modelscope_snapshot.json')
    ar_cell = original['cells']['AR']
    ar_path = BASE / 'runs' / ar_cell['run_id'] / 'assets/decode_observations.json'
    assert sha(ar_path) == ar_cell['files_sha256']['assets/decode_observations.json']
    ar = read(ar_path)['out']['evidence']
    reference = [{'question_id': p['question_id'], 'output_ids': r['output_ids'][len(p['input_ids']):],
                  'model_calls': r['stats']['ar_calls']} for p, r in zip(prompts['rendered_prompts'], ar)]
    assert len(reference) == len(ar) == 480
    cells = {name: {'native_status': 'pending'} for name in NAMES}
    cells['zero'] = dict(original['cells']['30/input_zero'], reused_from='30/input_zero',
        reused_file='studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json')
    for owner in owners:
        for experiment in read(owner / 'index.json')['experiments']:
            name = experiment['factors']['algorithm.config.direction']
            assert name in NAMES and name != 'zero' and len(experiment['runs']) == 1
            replacement = None
            if owner == RECOVERY:
                previous = cells[name]
                failed_path = STUDY / 'runs' / previous['run_id'] / 'result.json'
                failed = read(failed_path)
                require_resource_failure(previous, failed.get('error', ''))
                replacement = dict(previous, error=failed['error'], result_sha256=sha(failed_path))
            folder = owner / 'runs' / experiment['runs'][0]['id']
            result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
            cells[name] = {'run_id': folder.name, 'study_path': owner.relative_to(ROOT).as_posix(),
                'native_status': result.get('status', 'pending'), 'replaced_resource_failure': replacement}
            progress_file = folder / 'assets/direction_progress.json'
            if progress_file.exists():
                cells[name]['completed_prompt_rows'] = read(progress_file)['completed']
            if result.get('status') != 'succeeded':
                continue
            cfg = yaml.safe_load((folder / 'config.yaml').read_text())
            assert cfg['algorithm']['config']['direction'] == name and cfg['seed'] == 123
            expected = original['cells']['30/input_first']['actual_settings']
            assert cfg['algorithm']['config']['spec'] == expected['spec']
            assert cfg['algorithm']['config']['eval'] == expected['eval']
            cell, _ = load_cell(owner, folder, cfg, prompts['rendered_prompts'], prompts, snapshot, reference)
            assert cell['model_digest'] == original['cells']['30/input_first']['model_digest']
            assert cell['runtime'] == original['cells']['30/input_first']['runtime']
            progress = read(progress_file)
            assert progress['completed'] == len(progress['rows']) == 480 and progress['direction'] == name
            assert [r['question_id'] for r in progress['rows']] == [p['question_id'] for p in prompts['rendered_prompts']]
            assert all(r['direction']['full_prompt_mean_preserved'] for r in progress['rows'])
            cell.update(direction_rows=[{'question_id': r['question_id'], **r['direction']} for r in progress['rows']],
                progress_sha256=sha(progress_file), study_path=owner.relative_to(ROOT).as_posix(),
                replaced_resource_failure=replacement)
            cells[name] = cell
    for cell in cells.values():
        if 'rows' in cell:
            cell['totals']['category_mean_pooled_tau'] = category_mean(cell['rows'])
    comparisons = []
    if 'rows' in cells['template']:
        for name, cell in cells.items():
            if name != 'template' and 'rows' in cell:
                comparisons.append(dict(first='template', second=name, **compare(cells['template']['rows'], cell['rows'])))
    existing = []
    for name, cell in cells.items():
        if 'rows' in cell:
            for method in ('frozen_mean', 'input_zero', 'input_first', 'output_zero', 'output_first'):
                old = original['cells']['30/' + method]
                pair = compare(old['rows'], cell['rows'])
                # Canonical per-prompt counts in cells and the old evidence reconstruct these pairs.
                pair.pop('rows')
                existing.append(dict(first='30/' + method, second=name, **pair))
    ready = all('rows' in cell for cell in cells.values())
    native_validation = {owner.name: read(owner / 'spec_mtp_study_check.json')
        for owner in owners if (owner / 'spec_mtp_study_check.json').exists()}
    if ready:
        final_owner = owners[-1]
        native, process = read(final_owner / 'spec_mtp_study_check.json'), read(final_owner / 'process.json')
        assert native['passed'] and native['complete'] and process['complete'] and process['spec_mtp']['passed']
        assert len({c['source_digest'] for n, c in cells.items() if n != 'zero'}) == 1
        if final_owner == RECOVERY:
            initial = native_validation[STUDY.name]
            assert not initial['passed']
            assert set(initial['failed_runs']) == {c['replaced_resource_failure']['run_id']
                for c in cells.values() if c.get('replaced_resource_failure')}
    body = {'complete': ready, 'budget': 30, 'fixed_prompts': 480, 'argmax_is_gate_AR_is_diagnostic': True,
        'no_timing': True, 'original_evidence_sha256': sha(BASE / 'docs/paired_evidence_v1.json'),
        'source_refs_sha256': sha(STUDY / 'docs/source_refs.json'), 'analyzer_sha256': sha(Path(__file__)),
        'reused_analyzer_sha256': sha(BASE / 'analyze_results.py'),
        'native_validation': native_validation,
        'cells': cells, 'vs_template': comparisons, 'vs_existing_five': existing}
    target = STUDY / ('docs/direction_evidence_v1.json' if ready else 'docs/direction_pending.json')
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if ready and target.exists():
        assert target.read_text(encoding='utf-8') == text
    else:
        target.write_text(text, encoding='utf-8')
    if ready:
        plot(body, STUDY / 'docs/figures/bc30_directions.png')
    print(json.dumps({'complete': ready, 'full_cells': sum('rows' in c for c in cells.values()),
        'progress': {n: c.get('completed_prompt_rows', 480 if 'rows' in c else 0) for n, c in cells.items()}}))


if __name__ == '__main__':
    main()
