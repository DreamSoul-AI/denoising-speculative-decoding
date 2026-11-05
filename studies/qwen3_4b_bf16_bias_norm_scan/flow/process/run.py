from studies._shared.flow.process import run
__all__ = ['run']


import json
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[2]
BASE = ROOT / 'studies/qwen3_4b_bf16_five_methods'
from studies._shared.structure import geometry_4b as directions
from studies.qwen3_4b_bf16_five_methods.structure import readout as _pair


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


from spec_mtp.structure.artifact.provenance import file_sha256 as sha


def plot(cells, means, path):
    import matplotlib.pyplot as plt
    from spec_mtp.structure.artifact.paths import validate_output_root
    from studies.qwen3_4b_bf16_bias_norm_scan.structure.readout import norm_figure
    path = Path(path).resolve()
    validate_output_root(path.parent, ROOT)
    if path.exists() or path.is_symlink():
        raise FileExistsError('never replace an existing norm figure')
    fig = norm_figure(cells, means)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            fig.savefig(stream, format='png', dpi=160)
    finally:
        plt.close(fig)


def load_reused(data_root, *, reader=None):
    from studies._shared.structure import validation
    read_json = read if reader is None else reader.read_json
    sha256 = sha if reader is None else reader.sha
    data_root = Path(data_root).resolve()
    base = data_root / 'studies/qwen3_4b_bf16_five_methods'
    formal = read_json(base / 'docs/paired_evidence_v1.json')
    original = read_json(data_root / 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json')
    prompts, snapshot = (read_json(base / 'docs' / n) for n in ('prompts_manifest.json', 'modelscope_snapshot.json'))
    old_ar = formal['cells']['AR']
    arpath = data_root / 'studies/qwen3_4b_bf16_five_methods/runs' / old_ar['run_id'] / 'assets/decode_observations.json'
    assert sha256(arpath) == old_ar['files_sha256']['assets/decode_observations.json']
    raw_ar = read_json(arpath)['out']['evidence']
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
            assert sha256(owner / 'runs' / cell['run_id'] / name) == expected, name
        raw = read_json(path)['out']['evidence']
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
    from spec_mtp.structure.control.merge import deep_merge
    common = yaml.safe_load((STUDY / 'experiment_config.yaml').read_text(encoding='utf-8'))['algorithm']['config']
    return [deep_merge(common, setting) for setting in decl['axes']['algorithm.config']]


def check_matrix(decl, entries):
    expected = decl['axes']['algorithm.config']
    assert [(c['direction'], c['spec']['ema']['step_scale']) for c in expected] == [
        (mode, scale) for mode in ('template', 'random0', 'random1', 'random2') for scale in (.5, 2.)]
    assert [e['factors']['algorithm.config'] for e in entries] == matrix_settings(decl), 'Expected all eight frozen merged norm configurations in order'
    assert all(len(e['runs']) == 1 for e in entries)


def verify_reused(data_root, output_dir):
    data_root = Path(data_root).resolve(strict=True)
    from studies._shared.structure.validation import check_control_refs
    from studies._shared.structure import validation
    session = _native_session(data_root, STUDY, output_dir)
    sha256 = session.sha
    session.sha(STUDY/'docs/source_refs.json')
    check_control_refs(STUDY, ROOT, kind="norm")
    formal, _, _, _, reference, cells = load_reused(data_root, reader=session)
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
        'source_refs_sha256': sha256(STUDY / 'docs/source_refs.json'), 'analyzer_sha256': sha256(Path(__file__)),
        'raw_math_checker_sha256': sha256(Path(validation.__file__)),
        'formal_sources': {p.relative_to(data_root).as_posix(): sha256(p) for p in
            (data_root / 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json', data_root / 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json')},
        'cells': {name: dict({k: cell[k] for k in ('run_id', 'totals', 'raw_argmax_checked_tokens', 'raw_files_rechecked')},
                            study_path=cell.get('study_path', 'studies/qwen3_4b_bf16_five_methods'),
                            files_sha256=cell['files_sha256'])
                  for name, cell in cells.items()}, 'paired_comparisons': comparisons}
    session.write_text(session.output/'reused_readiness_v1.json', json.dumps(body, indent=2, allow_nan=False)+'\n')
    session.finish()
    return body


def continuation_entries(study, decl, entries, continuation):
    """Map only unexecuted old cells to identical scientific configurations."""
    from spec_mtp.structure.control.merge import deep_merge
    from spec_mtp.structure.artifact.provenance import check_frozen as pcheck
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


def native_readout(data_root, study_dir, output_dir, continuation_study=None):
    from studies._shared.structure.validation import check_control_refs, source_study_root
    from spec_mtp.structure.artifact.provenance import check_frozen as pcheck
    from spec_mtp.structure.make.expand import load_study_yaml
    from studies._shared.structure.sources import check_frozen, manifest
    data_root = Path(data_root).resolve(strict=True)
    study = Path(study_dir).resolve(strict=True)
    execution = source_study_root(study).parents[1]
    continuation = Path(continuation_study).resolve(strict=True) if continuation_study else None
    output = _new_output(output_dir, data_root, study, *([continuation] if continuation is not None else []))
    assert Path.cwd().resolve() == execution, 'Run from the frozen execution checkout root for declared relative inputs'
    session = _native_session(data_root, study, output, continuation)
    read_json, sha256 = session.read_json, session.sha
    session.sha(study/'docs/source_refs.json')
    check_control_refs(study, execution, kind="norm")
    decl = load_study_yaml(study)
    check_frozen(study, decl, manifest())
    pcheck(study, decl)
    formal, original, prompts, snapshot, reference, cells = load_reused(data_root, reader=session)
    entries = read_json(study / 'index.json')['experiments']
    check_matrix(decl, entries)
    selected = continuation_entries(study, decl, entries, continuation) if continuation else [(study, e) for e in entries]
    for cell_study, exp in selected:
        setting = exp['factors']['algorithm.config']
        mode, scale = setting['direction'], setting['spec']['ema']['step_scale']
        name = f'{mode}/{scale:g}'
        assert name not in cells and len(exp['runs']) == 1
        folder = cell_study / 'runs' / exp['runs'][0]['id']
        result = read_json(folder / 'result.json') if session.exists(folder / 'result.json') else {}
        cells[name] = {'run_id': folder.name, 'study_path': str(cell_study), 'native_status': result.get('status', 'pending')}
        if result.get('status') != 'succeeded':
            continue
        cfg = yaml.safe_load(session.read_text(folder / 'config.yaml'))
        assert cfg['algorithm']['config']['spec'] == setting['spec']
        assert cfg['seed'] == 123 and cfg['algorithm']['config']['direction'] == mode
        assert cfg['algorithm']['config']['spec']['ema']['step_scale'] == scale
        cell, _ = _pair.load_cell(cell_study, folder, cfg, prompts['rendered_prompts'], prompts, snapshot, reference, reader=session)
        assert cell['model_digest'] == formal['cells']['30/input_first']['model_digest']
        assert cell['runtime'] == formal['cells']['30/input_first']['runtime']
        progress = read_json(folder / 'assets/norm_progress.json')
        assert progress['completed'] == len(progress['rows']) == 480
        assert progress['direction'] == mode and progress['step_scale'] == scale
        for p, r, canonical in zip(progress['rows'], cell['rows'], prompts['rendered_prompts']):
            assert p['question_id'] == r['question_id'] == canonical['question_id']
            assert all(p[k] == r[k] for k in ('accepted', 'committed', 'decode_calls', 'new_tokens', 'argmax_checked_tokens'))
            assert p['bias']['step_scale'] == scale and p['bias']['full_prompt_mean_preserved']
        cell.update(study_path=str(cell_study), revalidation=result.get('spec_mtp', {}).get('revalidation'), progress_sha256=sha256(folder / 'assets/norm_progress.json'),
            bias_and_depth_rows=[{k: v for k, v in r.items() if k != 'gpu_admission'} for r in progress['rows']])
        cell['depth_rows'] = depth_rows(read_json(folder / 'assets/decode_observations.json')['out']['evidence'], prompts['rendered_prompts'])
        cells[name] = cell
    ready, pairs, means = summarize_cells(cells)
    if ready:
        native = read_json((continuation or study) / 'spec_mtp_study_check.json')
        process = read_json((continuation or study) / 'process.json')
        assert native['passed'] and native['complete'] and process['complete'] and process['spec_mtp']['passed']
    body = {'complete': ready, 'author_official': False, 'no_timing': True, 'fixed_prompts': 480,
        'budget': 30, 'mask_count': 2, 'new_cells': 8, 'reused_cells': 5,
        'continuation_study': str(continuation) if continuation else None,
        'original_study_fully_complete': ready and continuation is None,
        'source_refs_sha256': sha256(study / 'docs/source_refs.json'), 'analyzer_sha256': sha256(Path(__file__)),
        'paired_formal_sha256': sha256(data_root / 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json'),
        'direction_formal_sha256': sha256(data_root / 'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json'),
        'cells': cells, 'paired_comparisons': pairs, 'random_three_seed_mean_pooled_tau': means}
    target = output/('norm_evidence_v1.json' if ready else 'norm_pending.json')
    session.write_text(target, json.dumps(body, indent=2, allow_nan=False)+'\n')
    def render(folder):
        target = folder/'figures/bc30_norm_scan.png'
        plot(cells, means, target)
        return [target]
    session.finish(render=render if ready else None)
    return body


from studies.qwen3_4b_bf16_bias_norm_scan.structure.readout import depth_rows, first_block_compare, summarize_cells, verify_reused_published


def _new_output(output, *inputs):
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    result = validate_study_root(validate_output_root(output, *inputs))
    if result.exists() or result.is_symlink():
        raise FileExistsError('norm readout requires a new external output root')
    return result


def _native_session(data_root, study, output, continuation=None):
    from studies.qwen3_4b_bf16_paired_revalidation.flow.process.run import PublishedReadout
    roots = [Path(data_root).resolve(strict=True), BASE, STUDY]
    if continuation is not None:
        roots.append(Path(continuation).resolve(strict=True))
    session = PublishedReadout(Path(study).resolve(strict=True), output, *roots)
    session.bind_declarations(study, STUDY, continuation)
    return session


def published(input_path, directions_path, repository, commit, input_git_sha256, directions_git_sha256,
              output_root, *, execution_root=None):
    from studies.qwen3_4b_bf16_paired_revalidation.flow.process.run import PublishedReadout
    repository = Path(repository).resolve(strict=True)
    roots = [repository, Path(directions_path).resolve(strict=True).parent]
    if execution_root is not None:
        roots.append(Path(execution_root).resolve(strict=True))
    session = PublishedReadout(Path(input_path).resolve(strict=True).parent, output_root, *roots)
    proof = session.read_fixed(input_path, repository, commit,
        'studies/qwen3_4b_bf16_bias_norm_scan/docs/reused_readiness_v1.json', input_git_sha256)
    original = session.read_fixed(directions_path, repository, commit,
        'studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json', directions_git_sha256)
    summary = verify_reused_published(proof, original)
    count = session.check_files(proof['cells'], execution_root, 'studies/qwen3_4b_bf16_five_methods')
    session.finish_report(proof, summary, count, 'reused_counter_readout.json')
    return proof


import os
import subprocess
from typing import Any

import sys
STUDY = ROOT / 'studies/qwen3_4b_bf16_bias_norm_scan'
READER = STUDY / 'flow/process/run.py'

def audit_once(data_root, output_dir, continuation_study=None):
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    continuation = Path(continuation_study).resolve(strict=True) if continuation_study else None
    output = validate_study_root(validate_output_root(output_dir, STUDY, Path(data_root).resolve(),
        *([continuation] if continuation is not None else [])))
    owner = {'schema': 1, 'study': str(STUDY.resolve()), 'data_root': str(Path(data_root).resolve()),
             'reader_sha256': sha(READER),
             'continuation': str(Path(continuation_study).resolve()) if continuation_study else None}
    marker = output / 'audit_owner.json'
    if output.exists():
        if not marker.is_file() or read(marker) != owner:
            raise FileExistsError('refuse to adopt an existing or differently bound audit output')
    else:
        output.mkdir(parents=True, exist_ok=False)
        with marker.open('x', encoding='utf-8') as stream:
            json.dump(owner, stream, indent=2)
    entries = read(STUDY / 'index.json')['experiments']
    assert len(entries) == 8 and all((len(e['runs']) == 1 for e in entries))
    continuation = Path(continuation_study).resolve() if continuation_study else None
    selected = [(STUDY, e) for e in entries]
    if continuation:
        newer = read(continuation / 'index.json')['experiments']
        assert len(newer) == 7
        selected = [(STUDY, entries[0]), *[(continuation, e) for e in newer]]
    done, failed = ([], [])
    for cell_study, entry in selected:
        run = entry['runs'][0]['id']
        cfg = entry['factors']['algorithm.config']
        path = cell_study / 'runs' / run / 'result.json'
        result = read(path) if path.exists() else {}
        item = dict(run_id=run, case=f"{cfg['direction']}/{cfg['spec']['ema']['step_scale']:g}")
        if result.get('status') == 'succeeded':
            done.append(dict(item, result_sha256=sha(path)))
        elif result.get('status') == 'failed':
            failed.append(item)
    state: dict[str, Any] = dict(complete=False, native_completed=len(done), total=8, completed=done, failed=failed, no_model_forward=True, no_timing=True, audited_cells=[])
    signature = hashlib.sha256(json.dumps([done, sha(READER), sha(STUDY / 'docs/source_refs.json'), sha(continuation / 'docs/source_refs.json') if continuation else None], sort_keys=True).encode()).hexdigest()
    latest = output / 'ledger.json'
    previous = read(latest) if latest.exists() else {}
    if previous.get('signature') == signature and previous.get('failed') == failed:
        return previous
    if done and (not failed):
        process = (continuation or STUDY) / 'process.json'
        if len(done) == 8 and (not process.exists() or not read(process).get('complete')):
            state['phase'] = 'waiting_for_native_study_process'
        else:
            target = output / f"{len(done):02d}-{done[-1]['run_id']}-{signature[:8]}"
            target.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
            with (target / 'audit.stdout.log').open('w') as stdout, (target / 'audit.stderr.log').open('w') as stderr:
                subprocess.run([sys.executable, '-m', 'studies.qwen3_4b_bf16_bias_norm_scan.flow.process.run', 'native', '--data-root', str(Path(data_root).resolve()), '--study-dir', str(STUDY), '--output-root', str(target / 'readout'), *(['--continuation-study', str(continuation)] if continuation else [])], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
            body = read(target / 'readout' / ('norm_evidence_v1.json' if len(done) == 8 else 'norm_pending.json'))
            assert body['complete'] == (len(done) == 8)
            for item in done:
                cell = body['cells'][item['case']]
                assert cell['native_status'] == 'succeeded' and cell['run_id'] == item['run_id']
                assert len(cell['rows']) == 480
                totals = cell['totals']
                assert totals['argmax_checked_tokens'] == totals['new_tokens']
                state['audited_cells'].append(dict(item, totals=totals, files_sha256=cell['files_sha256']))
            state.update(complete=body['complete'], phase='accepted_completed_cells', signature=signature, readout_dir=str(target / 'readout'), readout_sha256=sha(target / 'readout' / ('norm_evidence_v1.json' if body['complete'] else 'norm_pending.json')))
    if not done or failed:
        state.update(phase='native_failure' if failed else 'waiting_for_first_complete_cell', signature=signature)
    latest.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return state


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Norm native source gate or fixed saved-counter readout')
    sub = parser.add_subparsers(dest='kind', required=True)
    native = sub.add_parser('native')
    native.add_argument('--data-root', type=Path, required=True)
    native.add_argument('--study-dir', type=Path, required=True)
    native.add_argument('--output-root', type=Path, required=True)
    native.add_argument('--continuation-study', type=Path)
    reused = sub.add_parser('reused')
    reused.add_argument('--data-root', type=Path, required=True)
    reused.add_argument('--output-root', type=Path, required=True)
    saved = sub.add_parser('published')
    for flag in ('input', 'directions', 'repository', 'output-root'):
        saved.add_argument('--'+flag, type=Path, required=True)
    for flag in ('commit', 'input-git-sha256', 'directions-git-sha256'):
        saved.add_argument('--'+flag, required=True)
    saved.add_argument('--execution-root', type=Path)
    args = parser.parse_args(argv)
    if args.kind == 'native':
        native_readout(args.data_root, args.study_dir, args.output_root, args.continuation_study)
    elif args.kind == 'reused':
        verify_reused(args.data_root, args.output_root)
    else:
        published(args.input, args.directions, args.repository, args.commit, args.input_git_sha256,
                  args.directions_git_sha256, args.output_root, execution_root=args.execution_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
