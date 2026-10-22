"""Read the independent paired matrix using original native cell and WTL helpers."""
import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path
import sys

import yaml
from paired_repeat import BASE, STUDY, ROOT, check_refs
sys.path.insert(0, str(ROOT / 'tools'))
import analyze_4b_full_directions as directions  # noqa: E402
spec = importlib.util.spec_from_file_location('paired_repeat_original_reader', BASE / 'analyze_results.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def check_index(entries):
    expected = check_refs()
    assert [e['factors']['algorithm.config'] for e in entries] == expected
    assert all(len(e['runs']) == 1 for e in entries)
    assert [original.case_name(s) for s in expected] == list(original.CASES)
    ids = [e['runs'][0]['id'] for e in entries]
    assert len(set(ids)) == len(ids) == 11
    return dict(zip(original.CASES, ids))


def readout(output_dir):
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    from studies._shared.structure.sources import check_frozen, manifest
    configs = check_refs()
    cells = {name: {'native_status': 'pending'} for name in original.CASES}
    index = STUDY / 'index.json'
    if index.exists():
        assert Path.cwd().resolve() == ROOT, 'Read from the frozen checkout root'
        decl = yaml.safe_load((STUDY / 'study.yaml').read_text(encoding='utf-8'))
        check_frozen(STUDY, decl, manifest())
        pcheck(STUDY, decl)
        entries = check_index(original.read(index)['experiments'])
        prompts = original.read(BASE / 'docs/prompts_manifest.json')
        snapshot = original.read(BASE / 'docs/modelscope_snapshot.json')
        reference = None
        for name in original.CASES:
            folder = STUDY / 'runs' / entries[name]
            result = original.read(folder / 'result.json') if (folder / 'result.json').exists() else {}
            cells[name] = dict(run_id=folder.name, native_status=result.get('status', 'pending'))
            if result.get('status') != 'succeeded' or (name != 'AR' and reference is None):
                continue
            cfg = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
            expected = configs[original.CASES.index(name)]
            assert cfg['algorithm']['config'] == expected and cfg['seed'] == 123
            cell, outputs = original.load_cell(STUDY, folder, cfg, prompts['rendered_prompts'], prompts, snapshot, reference)
            cells[name] = cell
            if name == 'AR':
                reference = outputs
            else:
                assert all(cell[k] == cells['AR'][k] for k in ('model_digest', 'source_digest', 'runtime'))
    pairs = []
    prior = original.read(BASE / 'docs/paired_evidence_v1.json')
    versus_prior = []
    for name, cell in cells.items():
        if name == 'AR' or 'rows' not in cell:
            continue
        cell['category_mean_pooled_tau'] = directions.category_mean(cell['rows'])
        versus_prior.append(dict(first='original/' + name, second=name,
                                 **directions.compare(prior['cells'][name]['rows'], cell['rows'])))
    for bc in (30, 60):
        for a, b in combinations(original.METHODS, 2):
            first, second = f'{bc}/{a}', f'{bc}/{b}'
            if all('rows' in cells[n] for n in (first, second)):
                pairs.append(dict(first=first, second=second, **directions.compare(cells[first]['rows'], cells[second]['rows'])))
    ready = all('rows' in c for c in cells.values())
    if ready:
        quality, process = (original.read(STUDY / name) for name in ('spec_mtp_study_check.json', 'process.json'))
        assert quality['passed'] and quality['complete'] and process['complete'] and process['spec_mtp']['passed']
    body = dict(complete=ready, scope=__doc__, argmax_is_gate_AR_is_diagnostic=True, timing_disabled=True,
                cells=cells, pairs=pairs, versus_original=versus_prior,
                source_refs_sha256=directions.sha(STUDY / 'docs/source_refs.json'),
                analyzer_sha256=directions.sha(Path(__file__)), shared_reader_sha256=directions.sha(BASE / 'analyze_results.py'))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if ready:
        from plot_results import write_plot
        body['plotter_sha256'] = directions.sha(BASE / 'plot_results.py')
        write_plot(body, output / 'figures/paired_outcomes.png')
    target = output / ('paired_evidence_v1.json' if ready else 'paired_pending.json')
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if target.exists() and ready:
        assert target.read_text(encoding='utf-8') == text, 'Never replace formal results'
    else:
        target.write_text(text, encoding='utf-8')
    return body


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    body = readout(args.output_dir)
    print(json.dumps({'complete': body['complete'], 'valid_cells': sum('rows' in c for c in body['cells'].values()), 'pairs': len(body['pairs'])}))
