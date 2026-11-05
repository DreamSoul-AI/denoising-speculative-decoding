"""Read the independent paired matrix using original native cell and WTL helpers."""
from studies._shared.flow.process import run
__all__ = ['run']


import json
from itertools import combinations
from pathlib import Path

import yaml
from studies.qwen3_4b_bf16_paired_revalidation.paired_repeat import BASE, STUDY, ROOT, check_refs
from studies._shared.structure import geometry_4b as directions
from studies.qwen3_4b_bf16_five_methods.structure import readout as original
from studies.qwen3_4b_bf16_five_methods.structure import plots as paired_plots
from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout


class PublishedReadout(SavedReadout):
    """Shared paired/norm saved-report binding, without accepting their old native source gate."""
    def _source_files(self):
        import inspect
        from spec_mtp.structure.artifact.provenance import file_sha256
        from studies._shared.structure import algorithm, config, evidence, sources, validation
        root = Path(__file__).resolve().parents[4]
        names = ('qwen3_4b_bf16_paired_revalidation', 'qwen3_4b_bf16_bias_norm_scan', 'qwen3_4b_bf16_five_methods')
        paths = [p for name in names for pillar in ('structure', 'flow') for p in (root/'studies'/name/pillar).rglob('*.py')]
        paths += [Path(directions.__file__), Path(sources.__file__), Path(inspect.getfile(SavedReadout))]
        for module in (algorithm, config, evidence, validation):
            assert module.__file__ is not None
            paths.append(Path(module.__file__))
        paths += [root/'studies'/name/file for name, file in (
            ('qwen3_4b_bf16_paired_revalidation', 'paired_repeat.py'),
            ('qwen3_4b_bf16_bias_norm_scan', 'norm_control.py'),
            ('_shared', 'structure/resource_admission.py'),
            ('qwen3_4b_bf16_five_methods', 'untimed_pairing.py'))]
        return {str(p.resolve()): file_sha256(p) for p in paths}

    def read_fixed(self, path, repository, commit, member, sha256):
        from spec_mtp.structure.artifact.provenance import read_historical_bytes
        value = self.read_json(path)
        if value != json.loads(read_historical_bytes(repository, commit, member, sha256)):
            raise ValueError('saved published report differs from its fixed Git member')
        self.historical.append({'commit': commit, 'path': member, 'sha256': sha256})
        return value

    def bind_declarations(self, *studies):
        for study in studies:
            if study is None:
                continue
            study = Path(study).resolve(strict=True)
            for name in ('study.yaml', 'experiment_config.yaml', 'recipe.py', 'index.json', 'provenance.json',
                         'spec_mtp_sources.json', 'docs/source_refs.json', 'spec_mtp_study_check.json', 'process.json'):
                path = study/name
                if self.exists(path):
                    self.sha(path)
            if self.exists(study/'study.yaml'):
                declaration = yaml.safe_load(self.read_text(study/'study.yaml'))
                for name in (declaration.get('provenance') or {}).get('include', []):
                    path = study/name
                    if self.exists(path):
                        self.sha(path)

    def check_files(self, cells, execution_root, default_study):
        if execution_root is None:
            return 0
        root = Path(execution_root).resolve(strict=True)
        required = {'config.yaml', 'result.json', 'assets/decode_observations.json', 'assets/decode_inputs.json',
                    'assets/pairing_contract.json', 'assets/spec_mtp_sources.json'}
        checked = 0
        owners = set()
        for cell in cells.values():
            owner = (root/cell.get('study_path', default_study)).resolve(strict=True)
            owner.relative_to(root)
            run_id, files = cell['run_id'], cell['files_sha256']
            if (not isinstance(run_id, str) or Path(run_id).name != run_id or run_id in ('.', '..')
                    or not isinstance(files, dict) or not required <= set(files) or (owner, run_id) in owners):
                raise ValueError('original file check needs complete, unique and safe Run identities')
            owners.add((owner, run_id))
            folder = owner/'runs'/run_id
            for name, expected in files.items():
                path = (folder/name).resolve(strict=True)
                path.relative_to(folder.resolve(strict=True))
                if self.sha(path) != expected:
                    raise ValueError('original file differs from its published SHA256')
                checked += 1
        return checked

    def finish_report(self, body, summary, files_checked, filename):
        self.write_text(self.output/filename, json.dumps(body, indent=2, allow_nan=False)+'\n')
        self.write_text(self.output/'counter_scope.json', json.dumps({
            'scope': 'saved published counters and optional recorded file hashes only; no native Run acceptance',
            'native_run_revalidated': False, 'recorded_files_sha256_checked': files_checked,
            'original_source_bytes_verified': False, 'original_plan_bytes_verified': False, **summary}, indent=2)+'\n')
        self.finish()


def _new_output(output, *inputs):
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    result = validate_study_root(validate_output_root(output, *inputs))
    if result.exists() or result.is_symlink():
        raise FileExistsError('paired readout requires a new external output root')
    return result


def published(input_path, prior, repository, commit, input_git_sha256, prior_git_sha256, output_root, *, execution_root=None):
    repository = Path(repository).resolve(strict=True)
    roots = [repository, Path(prior).resolve(strict=True).parent]
    if execution_root is not None:
        roots.append(Path(execution_root).resolve(strict=True))
    session = PublishedReadout(Path(input_path).resolve(strict=True).parent, output_root, *roots)
    body = session.read_fixed(input_path, repository, commit,
        'studies/qwen3_4b_bf16_paired_revalidation/docs/paired_evidence_v1.json', input_git_sha256)
    old = session.read_fixed(prior, repository, commit,
        'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json', prior_git_sha256)
    summary = directions.verify_paired_repeat(body, old)
    count = session.check_files(body['cells'], execution_root, 'studies/qwen3_4b_bf16_paired_revalidation')
    session.finish_report(body, summary, count, 'paired_counter_readout.json')
    return body


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Paired native source gate or fixed saved-counter readout')
    sub = parser.add_subparsers(dest='kind', required=True)
    native = sub.add_parser('native')
    native.add_argument('--study-dir', type=Path, required=True)
    native.add_argument('--output-root', type=Path, required=True)
    saved = sub.add_parser('published')
    for flag in ('input', 'prior', 'repository', 'output-root'):
        saved.add_argument('--'+flag, type=Path, required=True)
    for flag in ('commit', 'input-git-sha256', 'prior-git-sha256'):
        saved.add_argument('--'+flag, required=True)
    saved.add_argument('--execution-root', type=Path)
    args = parser.parse_args(argv)
    if args.kind == 'native':
        readout(args.output_root, study_dir=args.study_dir)
    else:
        published(args.input, args.prior, args.repository, args.commit, args.input_git_sha256,
                  args.prior_git_sha256, args.output_root, execution_root=args.execution_root)
    return 0


def check_index(entries, expected=None):
    expected = check_refs() if expected is None else expected
    assert [e['factors']['algorithm.config'] for e in entries] == expected
    assert all(len(e['runs']) == 1 for e in entries)
    assert [original.case_name(s) for s in expected] == list(original.CASES)
    ids = [e['runs'][0]['id'] for e in entries]
    assert len(set(ids)) == len(ids) == 11
    return dict(zip(original.CASES, ids))


def readout(output_dir, *, study_dir=None):
    from spec_mtp.structure.artifact.provenance import check_frozen as pcheck
    from studies._shared.structure.sources import check_frozen, manifest
    study = STUDY if study_dir is None else Path(study_dir).resolve(strict=True)
    session = PublishedReadout(study, output_dir, BASE)
    session.bind_declarations(study)
    output = session.output
    read_json = session.read_json
    session.sha(study/'docs/source_refs.json')
    configs = check_refs()
    cells = {name: {'native_status': 'pending'} for name in original.CASES}
    index = study / 'index.json'
    if session.exists(index):
        assert Path.cwd().resolve() == ROOT, 'Read from the frozen checkout root'
        decl = yaml.safe_load((study / 'study.yaml').read_text(encoding='utf-8'))
        check_frozen(study, decl, manifest())
        pcheck(study, decl)
        entries = check_index(read_json(index)['experiments'])
        prompts = read_json(BASE / 'docs/prompts_manifest.json')
        snapshot = read_json(BASE / 'docs/modelscope_snapshot.json')
        reference = None
        for name in original.CASES:
            folder = study / 'runs' / entries[name]
            result = read_json(folder / 'result.json') if session.exists(folder / 'result.json') else {}
            cells[name] = dict(run_id=folder.name, native_status=result.get('status', 'pending'))
            if result.get('status') != 'succeeded' or (name != 'AR' and reference is None):
                continue
            cfg = yaml.safe_load(session.read_text(folder / 'config.yaml'))
            expected = configs[original.CASES.index(name)]
            assert cfg['algorithm']['config'] == expected and cfg['seed'] == 123
            cell, outputs = original.load_cell(study, folder, cfg, prompts['rendered_prompts'], prompts, snapshot, reference, reader=session)
            cells[name] = cell
            if name == 'AR':
                reference = outputs
            else:
                assert all(cell[k] == cells['AR'][k] for k in ('model_digest', 'source_digest', 'runtime'))
    pairs = []
    prior = read_json(BASE / 'docs/paired_evidence_v1.json')
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
        quality, process = (read_json(study / name) for name in ('spec_mtp_study_check.json', 'process.json'))
        assert quality['passed'] and quality['complete'] and process['complete'] and process['spec_mtp']['passed']
    body = dict(complete=ready, scope=__doc__, argmax_is_gate_AR_is_diagnostic=True, timing_disabled=True,
                cells=cells, pairs=pairs, versus_original=versus_prior,
                source_refs_sha256=session.sha(study / 'docs/source_refs.json'),
                analyzer_sha256=session.sha(Path(__file__)), shared_reader_sha256=session.sha(original.__file__))
    if ready:
        from studies.qwen3_4b_bf16_five_methods.flow.write.run import write_plot
        body['plotter_sha256'] = session.sha(paired_plots.__file__)
    target = output / ('paired_evidence_v1.json' if ready else 'paired_pending.json')
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    session.write_text(target, text)
    def render(folder):
        path = folder/'figures/paired_outcomes.png'
        write_plot(body, path)
        return [path]
    session.finish(render=render if ready else None)
    return body


if __name__ == '__main__':
    raise SystemExit(main())
