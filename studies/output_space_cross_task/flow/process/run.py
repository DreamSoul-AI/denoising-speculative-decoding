"""Process saved cross-task sampling evidence; no model or sampling execution."""
from studies._shared.flow.process import run

__all__ = ['run']

import argparse
import json
from pathlib import Path

from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout
from studies.output_space_cross_task.structure import sampling_readout as protocol

PLAN_ROOT = 'studies/output_space_cross_task/docs/'
PRODUCERS = {
    'first-delta': {'new': ('diagnose_sampling_expansion.py', 'SAMPLING_EXPANSION_PLAN.md'),
                    'old': ('diagnose_sampling_paths.py', 'SAMPLING_PLAN.md')},
    'sampling-expansion': {'expansion': ('diagnose_sampling_expansion.py', 'SAMPLING_EXPANSION_PLAN.md'),
                           'old': ('diagnose_sampling_paths.py', 'SAMPLING_PLAN.md'),
                           'rare': ('diagnose_rare_events.py', 'RARE_EVENT_PLAN.md')},
    'continuation': {'expansion': ('diagnose_sampling_continuation.py', 'SAMPLING_CONTINUATION_PLAN.md'),
                     'rare': ('diagnose_rare_stress.py', 'SAMPLING_CONTINUATION_PLAN.md')},
}


class CrossTaskReadout(SavedReadout):
    def _source_files(self):
        from spec_mtp.structure.artifact.provenance import file_sha256
        from studies._shared.structure import algorithm, config, evidence, sources
        from studies._shared.structure import extension_diagnostics, sampling_diagnostics, window_diagnostics

        root = Path(__file__).resolve().parents[2]
        files = [path for pillar in ('structure', 'flow') for path in (root / pillar).rglob('*.py')]
        for module in (extension_diagnostics, sampling_diagnostics, window_diagnostics, algorithm, config, evidence, sources):
            assert module.__file__ is not None
            files.append(Path(module.__file__))
        import inspect
        files.append(Path(inspect.getfile(SavedReadout)))
        return {str(path.resolve()): file_sha256(path) for path in files}


def bind_inputs(kind, inputs, inputs_sha256, output_root, *, roles, producers,
                quality_fields=None, source_roles=None):
    """Bind the shared role index, exact historical producers and actual saved bytes once."""
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    index_path = Path(inputs).resolve(strict=True)
    if file_sha256(index_path) != inputs_sha256:
        raise ValueError('input index SHA256 differs from the explicit binding')
    index = json.loads(index_path.read_bytes())
    if (not isinstance(index, dict) or index.get('schema') != 1 or index.get('kind') != kind
            or not isinstance(index.get('reports'), dict) or set(index['reports']) != set(roles)
            or not isinstance(index.get('files'), dict) or not isinstance(index.get('producer_sources'), dict)):
        raise ValueError('saved readout requires a complete role-indexed input manifest')
    repository = Path(index['repository']).resolve(strict=True)
    entries = [*index['reports'].values(), *index['files'].values()]
    entries += [entry for mapping in index['producer_sources'].values() for entry in mapping.values()]
    for entry in entries:
        value = Path(entry['path'])
        entry['path'] = str((value if value.is_absolute() else index_path.parent / value).resolve(strict=True))
    paths = [Path(entry['path']).resolve(strict=True) for entry in entries]
    session = CrossTaskReadout(index_path.parent, output_root, repository, *(path.parent for path in paths))
    if session.sha(index_path) != inputs_sha256:
        raise ValueError('input index changed during admission')
    def historical(entry):
        commit = entry.get('commit', index['commit'])
        blob = read_historical_bytes(repository, commit, entry['member'], entry['git_sha256'])
        session.historical.append({'commit': commit, 'path': entry['member'], 'sha256': entry['git_sha256']})
        return blob
    reports, report_hashes = {}, {}
    quality_fields = {role: 'quality_passed' for role in roles} if quality_fields is None else quality_fields
    source_roles = set(roles) if source_roles is None else set(source_roles)
    for role, entry in index['reports'].items():
        actual, fixed = session.read_json(entry['path']), json.loads(historical(entry))
        field = entry.get('runtime_sha256_field')
        if field is None:
            if actual != fixed:
                raise ValueError('actual report differs from the pinned historical JSON')
        elif session.sha(entry['path']) != fixed[field]:
            raise ValueError('actual report differs from its pinned runtime byte SHA256')
        quality = quality_fields.get(role)
        if quality is not None and actual.get(quality) is not True:
            raise ValueError('saved report does not pass its original quality gate')
        if ('source_digest' in actual or role in source_roles) and actual.get('source_digest') != protocol.SOURCE_DIGEST:
            raise ValueError('original producer source digest is missing or differs')
        reports[role], report_hashes[role] = actual, session.sha(entry['path'])
    hashes = {}
    for role, entry in index['files'].items():
        hashes[role] = session.sha(entry['path'])
        if hashes[role] != entry['sha256']:
            raise ValueError('saved file SHA256 differs from its explicit identity')
        if 'member' in entry and session.read_bytes(entry['path']) != historical(entry):
            raise ValueError('original runtime bytes differ from pinned Git source; identity remains unknown')
    for role, claims in producers.items():
        binding = reports[role].get('source_binding')
        if not isinstance(binding, dict):
            raise ValueError('original producer source binding is missing')
        for claim, member in claims.items():
            entry = index['producer_sources'].get(role, {}).get(claim)
            allowed = (member,) if isinstance(member, str) else member
            if entry is None or session.sha(entry['path']) != binding[claim] or entry.get('member') not in allowed:
                raise ValueError('original producer runtime source is missing or unbound; identity remains unknown')
            if session.read_bytes(entry['path']) != historical(entry):
                raise ValueError('original runtime source and Git bytes differ; identity remains unknown')
    return session, index, reports, report_hashes, hashes


def analyze(kind, inputs, inputs_sha256, output_root, *, device=None):
    kinds = {'first-delta': {'new', 'old', 'summary'},
             'sampling-expansion': {'expansion', 'rare', 'old', 'context'},
             'continuation': {'expansion', 'rare', 'context'}}
    if kind not in kinds:
        raise ValueError('unknown saved cross-task analysis')
    producers: dict[str, dict[str, str | tuple[str, ...]]] = {role: {'tool': 'studies/_shared/structure/' + names[0], 'plan': PLAN_ROOT + names[1]}
                 for role, names in PRODUCERS[kind].items()}
    if kind == 'continuation':
        producers['expansion']['tool'] = ('studies/_shared/structure/diagnose_sampling_continuation.py',
                                         'tools/diagnose_sampling_continuation.py')
        producers['rare']['tool'] = ('studies/_shared/structure/diagnose_rare_stress.py',
                                    'tools/diagnose_rare_stress.py')
    session, index, reports, report_hashes, hashes = bind_inputs(kind, inputs, inputs_sha256, output_root,
        roles=kinds[kind], producers=producers,
        quality_fields={role: None if role == 'context' else 'quality_passed' for role in kinds[kind]},
        source_roles=kinds[kind] - {'summary', 'context'})
    if kind == 'first-delta':
        plan = index['files'].get('analysis_plan', {})
        if plan.get('member') != PLAN_ROOT + 'FIRST_DELTA_PLAN.md' or 'git_sha256' not in plan:
            raise ValueError('first-delta original analysis plan requires its fixed Git member identity')
    bindings = {'analysis': session.sha(protocol.__file__),
                **{role + '_report': sha for role, sha in report_hashes.items()}}
    if kind == 'first-delta':
        if device is not None:
            raise ValueError('first-delta saved JSON accounting does not use a tensor device')
        if 'analysis_plan' not in hashes:
            raise ValueError('first-delta requires its actual original analysis plan')
        session.verify()
        result = protocol.analyze_first_delta(reports['new'], reports['old'], reports['summary'],
            {'tool': bindings['analysis'], 'plan': hashes['analysis_plan'], 'new': report_hashes['new'],
             'old': report_hashes['old'], 'summary': report_hashes['summary']})
        filename = 'first_delta_analysis_v1.json'
    else:
        import torch
        if device is None:
            raise ValueError('saved tensor audits require an explicit CUDA device')
        device = torch.device(device)
        if device.type != 'cuda' or device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise ValueError('saved tensor audits require an explicit available CUDA device')
        if hashes.get('tensor') != reports['expansion']['raw_tensor_sha256'] or hashes.get('context_tensor') != reports['context']['raw_tensor_sha256']:
            raise ValueError('saved tensors differ from their original report binding')
        if kind == 'continuation':
            for role in ('input_manifest', 'dataset'):
                if hashes.get(role) != reports['expansion']['source_binding'][role]:
                    raise ValueError('continuation input identity differs from its original producer binding')
        selection_hashes = {}
        units = [*reports.get('old', {}).get('units', []), *reports['expansion']['units']]
        for unit in units:
            role = f"selection:{unit['qid']}:{unit['seed']}"
            if role not in hashes or hashes[role] != unit['selection_sha256']:
                raise ValueError('saved unit selection is missing or differs from its original identity')
            selection_hashes[unit['qid'], unit['seed']] = hashes[role]
        session.verify()
        print(f'saved {kind} device={device} name={torch.cuda.get_device_name(device)}', flush=True)
        tensor_path = session._input(index['files']['tensor']['path'])
        context_path = session._input(index['files']['context_tensor']['path'])
        data = torch.load(tensor_path, weights_only=True, map_location=device, mmap=True)
        context = torch.load(context_path, weights_only=True, map_location=device, mmap=True)
        bindings['plan'] = reports['expansion']['source_binding']['plan']
        if kind == 'sampling-expansion':
            result = protocol.analyze_expansion(reports['expansion'], reports['rare'], reports['old'], data,
                context, selection_hashes, bindings, device)
            filename = 'sampling_expansion_analysis_v1.json'
        else:
            result = protocol.analyze_continuation(reports['expansion'], reports['rare'], data, context,
                selection_hashes, bindings, device)
            filename = 'sampling_continuation_analysis_v1.json'
        result['readout_device'] = str(device)
        result['readout_device_name'] = torch.cuda.get_device_name(device)
    result['readout_only'] = True
    result['readout_model_forwards'] = result['readout_sampling_draws'] = 0
    session.write_text(session.output / filename, json.dumps(result, indent=2, allow_nan=False) + '\n')
    session.finish()
    return result


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'native':
        from studies.output_space_cross_task.flow.process.native import main as native_main
        return native_main(argv[1:])
    if argv and argv[0] == 'state':
        from studies.output_space_cross_task.flow.process.state import main as state_main
        return state_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('first-delta', 'sampling-expansion', 'continuation', 'native', 'state'))
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args(argv)
    analyze(args.kind, args.inputs, args.inputs_sha256, args.output_root, device=args.device)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
