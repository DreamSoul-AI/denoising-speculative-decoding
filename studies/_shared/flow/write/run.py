"""Recheck the same evidence gate before the library writes success."""
from studies._shared.flow.summarize.run import run_gate as run

__all__ = ['run']


def publish_new_completion(path, payload):
    """Write a new receipt; ordinary write failures remove only the file this call created."""
    created = False
    try:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            created = True
            stream.write(payload)
    except BaseException as error:
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup:
                setattr(error, 'completion_cleanup_error', cleanup)
        raise


def write_paired_report(study, bindings_path, repository, output_root):
    """Render either fixed 4B report with independently bound actual reference inputs."""
    import json
    import os
    from pathlib import Path
    from urllib.parse import quote
    from spec_mtp.structure.artifact import layout, paths, provenance
    from studies._shared.structure import paired_report

    if study not in ('output_space_4b_holdout', 'output_space_4b_recheck_v2'):
        raise ValueError('unknown fixed paired-report Study')
    primary = study == 'output_space_4b_recheck_v2'
    prefix = f'studies/{study}/docs/'
    members = {role: prefix + name for role, name in {
        'execution': 'execution_v1.json', 'plan': 'PLAN.md', 'prompts': 'prompts_manifest.json',
        'history': 'history_inventory.json', 'snapshot': 'modelscope_snapshot.json'}.items()}
    if primary:
        members.update(first_failure=prefix + 'first_run_failure.json',
                       holdout_plan='studies/output_space_4b_holdout/docs/PLAN.md')
    else:
        members['primary_report'] = 'studies/output_space_4b_recheck_v2/docs/STUDY_REPORT.md'
    source_paths = [Path(__file__).resolve(), Path(paired_report.__file__).resolve(),
                    Path(provenance.__file__).resolve(), Path(paths.__file__).resolve(), Path(layout.__file__).resolve(),
                    Path(__file__).resolve().parents[3] / study / 'flow/write/run.py']
    sources = {path: provenance.file_sha256(path) for path in source_paths}
    binding_path, repository = Path(bindings_path).resolve(strict=True), Path(repository).resolve(strict=True)
    observed = {binding_path: provenance.file_sha256(binding_path)}
    bindings = json.loads(binding_path.read_bytes())
    if not isinstance(bindings, dict) or 'execution' not in bindings or set(bindings) - members.keys():
        raise ValueError('paired report requires execution and only its declared optional reference roles')
    actual, historical, bodies = {}, [], {}
    for role, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {'path', 'commit', 'sha256'}:
            raise ValueError('each report binding requires path, immutable commit and Git SHA256')
        path = Path(binding['path']).resolve(strict=True)
        observed[path] = provenance.file_sha256(path)
        archived = provenance.read_historical_bytes(repository, binding['commit'], members[role], binding['sha256'])
        raw = path.read_bytes()
        if members[role].endswith('.json'):
            body = json.loads(raw)
            equal = body == json.loads(archived)
        else:
            body, equal = None, raw == archived
        if not equal:
            raise ValueError(f'actual {role} input differs from its fixed Git original')
        actual[role], bodies[role] = path, body
        historical.append({'role': role, 'commit': binding['commit'], 'path': members[role], 'sha256': binding['sha256']})
    output = layout.validate_study_root(paths.validate_output_root(output_root, repository,
                                        *(path.parent for path in observed)))
    if output.exists() or output.is_symlink():
        raise FileExistsError('paired report requires a new external output root')
    data = bodies['execution']
    if (data.get('quality_passed') is not True or data.get('decoding_timing_disabled') is not True
            or data.get('source_digest') != '670c744e0535dcad' or data.get('n') != (26 if primary else 4)):
        raise ValueError('paired report lacks its original quality/source/timing/prompt contract')
    for role, field in (('plan', 'plan_sha256'), ('snapshot', 'snapshot_sha256'), ('prompts', 'manifest_sha256')):
        if role in actual and observed[actual[role]] != data[field]:
            raise ValueError(f'actual {role} input is not the original execution binding')
    links = {}
    for role, path in actual.items():
        try:
            links[role] = quote(Path(os.path.relpath(path, output)).as_posix(), safe='/.:')
        except ValueError:
            links[role] = path.as_uri()

    def verify():
        if any(provenance.file_sha256(path) != digest for path, digest in (observed | sources).items()):
            raise ValueError('paired report input or rendering source changed during execution')

    verify()
    text = paired_report.render_report(study, data, links)
    figure = paired_report.paired_counts_figure(data)
    try:
        verify()
        output.mkdir(parents=True, exist_ok=False)
        from spec_mtp.structure.system.logger import Logger
        Logger(output / 'assets').info('CPU saved-scalar paired report rendering; no tensor or model forward')
        image = output / 'paired_counts_v1.png'
        with image.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=150)
        with (output / 'paired_report.md').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        verify()
        import matplotlib
        result = {'schema': 1, 'study': study, 'scope': 'CPU saved scalars and historical prose; no model/tensor forward',
            'historical_inputs': historical,
            'runtime_inputs': {role: {'path': str(path), 'sha256': observed[path]} for role, path in actual.items()},
            'bindings_input': {'path': str(binding_path), 'sha256': observed[binding_path]},
            'unprovided_historical_references': sorted(set(members) - actual.keys()),
            'rendering_sources': {str(path): digest for path, digest in sources.items()},
            'rendering_environment': {'matplotlib': matplotlib.__version__, 'backend': 'Agg', 'dpi': 150},
            'outputs': {path.name: provenance.file_sha256(path) for path in (image, output / 'paired_report.md')}}
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        publish_new_completion(output / 'report_complete.json', payload)
        return output / 'paired_report.md'
    finally:
        figure.clear()


def paired_report_main(study, argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Render a fixed historical 4B paired report from explicit bindings')
    for name in ('bindings', 'repository', 'output-root'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    print(write_paired_report(study, args.bindings, args.repository, args.output_root))
    return 0
