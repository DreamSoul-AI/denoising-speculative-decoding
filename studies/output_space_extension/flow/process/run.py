"""Study process and explicit offline audit of saved transfer observations.

    python -m studies.output_space_extension.flow.process.run --help
"""
from studies._shared.flow.process import run

__all__ = ['run']

EXECUTION_MEMBER = 'studies/output_space_extension/docs/execution_v1.json'
SOURCE_DIGEST = '670c744e0535dcad'

SAVED_ROLES = {'numerics': {'numerics', 'native'},
               'kernels': {'norm', 'qproj', 'previous', 'native'},
               'operators': {'operator', 'previous', 'native'}}
SAVED_TENSORS = {'numerics': {'numerics_tensor', 'original_tensor'},
                 'kernels': {'norm_tensor', 'qproj_tensor', 'previous_tensor'},
                 'operators': {'operator_tensor', 'previous_tensor'}}


def analyze_saved(kind, inputs, inputs_sha256, output_root, *, device):
    """Audit only fixed saved q178 evidence; live mechanism dispatch remains stopped."""
    import inspect
    import json
    from pathlib import Path
    import torch
    from spec_mtp.structure.artifact.provenance import file_sha256
    from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout
    from studies.output_space_cross_task.flow.process.run import bind_inputs
    from studies.output_space_extension.structure import saved_diagnostics as protocol
    from studies._shared.structure import extension_diagnostics, window_diagnostics, q178_diagnostics
    from studies.output_space_cross_task.structure import sampling_readout

    if kind not in SAVED_ROLES:
        raise ValueError('unknown saved extension audit')
    input_alias = Path(inputs).absolute()
    if file_sha256(input_alias) != inputs_sha256:
        raise ValueError('saved extension input index SHA256 differs from the explicit binding')
    raw_index = json.loads(input_alias.read_bytes())
    if set(raw_index.get('files', {})) != SAVED_TENSORS[kind]:
        raise ValueError('saved extension requires its exact original tensor roles; no additional tensors')
    entries = [*raw_index['reports'].values(), *raw_index['files'].values()]
    entries += [entry for values in raw_index['producer_sources'].values() for entry in values.values()]
    index_target = input_alias.resolve(strict=True)
    aliases = {input_alias: index_target}
    for entry in entries:
        path = Path(entry['path'])
        alias = (path if path.is_absolute() else index_target.parent / path).absolute()
        aliases[alias] = alias.resolve(strict=True)

    class ExtensionReadout(SavedReadout):
        def _source_files(self):
            root = Path(__file__).resolve().parents[2]
            paths = [path for pillar in ('flow', 'structure') for path in (root / pillar).rglob('*.py')]
            paths += [Path(inspect.getfile(module)) for module in
                      (SavedReadout, extension_diagnostics, window_diagnostics, q178_diagnostics, sampling_readout)]
            paths.append(Path(inspect.getfile(bind_inputs)))
            return {str(path.resolve()): file_sha256(path) for path in paths}

        def verify(self):
            if any(alias.resolve(strict=True) != target for alias, target in aliases.items()):
                raise ValueError('saved extension input alias changed its resolved target')
            super().verify()

    docs = 'studies/output_space_extension/docs/'
    shared = 'studies/_shared/structure/'
    numeric_plan = raw_index['producer_sources'].get('numerics', raw_index['producer_sources'].get('previous', {})).get('plan_sha256', {})
    # The original report's plan byte SHA is authoritative; its exact Git member
    # is supplied explicitly rather than guessed from an absent archive filename.
    plan_member = numeric_plan.get('member')
    if not isinstance(plan_member, str) or not plan_member.startswith(docs) or not plan_member.endswith('.md'):
        raise ValueError('original numerical plan Git member is required')
    numeric = {'tool_sha256': (shared + 'diagnose_q178_numerics.py', 'tools/diagnose_q178_numerics.py'),
        'plan_sha256': plan_member,
        'observer_helper_sha256': (shared + 'diagnose_output_windows.py', 'tools/diagnose_output_windows.py'),
        'transfer_helper_sha256': (shared + 'diagnose_proxy_transfer.py', 'tools/diagnose_proxy_transfer.py')}
    producers = {'numerics': numeric} if kind == 'numerics' else {'previous': numeric}
    if kind == 'kernels':
        for role, tool, plan in (('norm', 'diagnose_q178_norm_kernel.py', docs + 'NORM_KERNEL_PLAN.md'),
                                 ('qproj', 'diagnose_q178_qproj.py', 'studies/output_space_cross_task/docs/KERNEL_EXECUTION_PLAN.md')):
            producers[role] = {'tool': (shared + tool, 'tools/' + tool), 'plan': plan,
                               'common': 'studies/output_space_extension/kernel_common.py'}
    elif kind == 'operators':
        producers['operator'] = {'tool': docs + 'operator_tool_v1.py.txt', 'plan': docs + 'OPERATOR_PLAN.md',
            'canonical_helper': numeric['tool_sha256'], 'transfer_helper': numeric['transfer_helper_sha256'],
            'observer_helper': numeric['observer_helper_sha256']}
    roles = SAVED_ROLES[kind]
    session, index, reports, report_hashes, hashes = bind_inputs('extension-' + kind, inputs, inputs_sha256,
        output_root, roles=roles, producers=producers, source_roles=roles,
        quality_fields={role: None if role == 'native' else 'quality_passed' for role in roles},
        session_type=ExtensionReadout)
    expected = {'native': (EXECUTION_MEMBER, None),
        'numerics': (docs + 'numerical_analysis_v1.json', 'raw_report_sha256'),
        'previous': (docs + 'numerical_analysis_v1.json', 'raw_report_sha256'),
        'norm': (docs + 'norm_kernel_diagnosis_v1.json', None),
        'qproj': (docs + 'qproj_kernel_diagnosis_v1.json', None),
        'operator': (docs + 'operator_diagnosis_v1.json', None)}
    for role, entry in index['reports'].items():
        if (entry['member'], entry.get('runtime_sha256_field')) != expected[role]:
            raise ValueError('saved extension report does not name its exact historical input role')
    for role in roles - {'native'}:
        if hashes.get(role + '_tensor') != reports[role]['raw_tensor_sha256']:
            raise ValueError('saved extension tensor identity differs from its original report')
        binding = reports[role]['source_binding']
        if binding.get('native', binding.get('matrix_sha256')) != report_hashes['native']:
            raise ValueError('saved extension matrix runtime bytes differ from their producer binding')
    if kind == 'numerics':
        if hashes.get('original_tensor') != reports['numerics']['source_binding']['old_transfer_tensor_sha256']:
            raise ValueError('original transfer tensor is missing or unbound')
    else:
        for role in roles - {'native', 'previous'}:
            binding = reports[role]['source_binding']
            old_tensor = binding['old_tensor'] if role == 'operator' else binding['previous_tensor']
            old_report = binding['old_report'] if role == 'operator' else binding['previous_report']
            if old_tensor != hashes['previous_tensor'] or old_report != report_hashes['previous']:
                raise ValueError('previous numerical context differs from its original producer binding')
    actual = torch.device(device)
    if actual != torch.device('cuda:0') or not torch.cuda.is_available():
        raise ValueError('saved extension audit requires explicit available cuda:0; no CPU fallback')
    session.verify()
    print(f'saved extension tensor.device={actual} name={torch.cuda.get_device_name(actual)}; no model/profiler dispatch', flush=True)
    tensors = {role[:-7]: torch.load(session._input(index['files'][role]['path']), map_location=actual, weights_only=True, mmap=True)
               for role in sorted(SAVED_TENSORS[kind])}
    binding = {'analysis': file_sha256(protocol.__file__), **{role + '_report': sha for role, sha in report_hashes.items()}}
    if kind == 'numerics':
        result = protocol.analyze_numerics(reports['numerics'], tensors['numerics'], tensors['original'], reports['native'],
            {**binding, 'report': report_hashes['numerics'], 'historical_helper': reports['numerics']['source_binding']['tool_sha256']})
    elif kind == 'kernels':
        result = protocol.analyze_kernels(reports, tensors, tensors['previous'], reports['native'], binding)
    else:
        result = protocol.analyze_operators(reports['operator'], tensors['operator'], tensors['previous'], reports['native'],
            {**binding, 'report': report_hashes['operator'], 'tensor': hashes['operator_tensor']})
    result.update(readout_only=True, readout_model_forwards=0, readout_profiler_calls=0,
        readout_device=str(actual), readout_device_name=torch.cuda.get_device_name(actual),
        historical_CPU_margin_revalidated=False,
        numerical_scope='Current CUDA saved-tensor diagnostics; historical CPU margins remain independent/unknown')
    session.verify()
    session.write_text(session.output / (kind + '_saved_evidence_v1.json'), json.dumps(result, indent=2, allow_nan=False) + '\n')
    session.finish()
    return result


def analyze(raw_root, execution, repository, commit, execution_git_sha256, output_root, *, device=None):
    """Derive a new audit from pinned inputs; never re-sign the saved Run."""
    if not __debug__:
        raise RuntimeError('saved transfer audit requires assertion checks; Python -O is unsupported')
    import json
    from pathlib import Path
    import torch
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    from studies._shared.structure import extension_diagnostics, window_diagnostics
    actual = torch.device(device) if device is not None else None
    if actual != torch.device('cuda:0') or not torch.cuda.is_available():
        raise ValueError('saved transfer audit requires explicit available cuda:0; no CPU fallback')

    raw_root = Path(raw_root).resolve()
    repository = Path(repository).resolve()
    execution = Path(execution).resolve()
    output_root = validate_output_root(output_root, raw_root, execution.parent, repository)
    target = output_root / 'slot_diagnosis.json'
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    report_path = (raw_root / 'diagnosis.json').resolve(strict=True)
    tensor_path = (raw_root / 'selected_tensors.pt').resolve(strict=True)
    for member in (report_path, tensor_path):
        member.relative_to(raw_root)
    report_sha = file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('quality_passed') is not True or report.get('source_digest') != SOURCE_DIGEST:
        raise ValueError('saved transfer report has unverified or incompatible source identity')
    tensor_sha = file_sha256(tensor_path)
    if tensor_sha != report['raw_tensor_sha256']:
        raise ValueError('saved tensor SHA256 differs from transfer report')
    matrix_sha = file_sha256(execution)
    if matrix_sha != report['source_binding']['matrix_sha256']:
        raise ValueError('runtime execution SHA256 differs from transfer report')
    matrix = read_historical_bytes(repository, commit, EXECUTION_MEMBER, execution_git_sha256)
    data = json.loads(execution.read_bytes())
    if data != json.loads(matrix):
        raise ValueError('runtime execution and pinned Git member have different content')
    if data.get('source_digest') != report['source_digest']:
        raise ValueError('historical execution and saved transfer sources differ')
    tool_path, helper_path = Path(__file__), Path(extension_diagnostics.__file__)
    window_path = Path(window_diagnostics.__file__)
    tool_sha, helper_sha = file_sha256(tool_path), file_sha256(helper_path)
    window_sha = file_sha256(window_path)
    print(f'saved transfer tensor.device={actual} name={torch.cuda.get_device_name(actual)}', flush=True)
    raw = torch.load(tensor_path, map_location=actual, weights_only=True)
    findings = extension_diagnostics.analyze_saved_transfers(report, data, raw, device=actual)
    if (file_sha256(report_path) != report_sha or file_sha256(tensor_path) != tensor_sha
            or file_sha256(execution) != matrix_sha or file_sha256(tool_path) != tool_sha
            or file_sha256(helper_path) != helper_sha or file_sha256(window_path) != window_sha):
        raise ValueError('analysis input changed during reading')
    result = {
        'schema': 1,
        'kind': 'post hoc offline second-slot expansion and propagation audit; no additional model forwards',
        'note': 'First-slot hypotheses were predeclared; second-slot scalar audit is post hoc from saved logits.',
        'source_digest': report['source_digest'],
        'raw_report_sha256': report_sha, 'raw_tensor_sha256': tensor_sha,
        'historical_input': {'commit': commit, 'path': EXECUTION_MEMBER, 'sha256': execution_git_sha256},
        'runtime_execution_sha256': matrix_sha, 'execution_content_equal': True,
        'readout_device': str(actual), 'readout_device_name': torch.cuda.get_device_name(actual),
        'analysis_provenance': {'module': __name__, 'tool_sha256': tool_sha, 'helper_sha256': helper_sha,
                                'window_sha256': window_sha},
        'findings': findings,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output_root.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
    return target


def main(argv=None):
    import argparse
    import sys
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == 'saved':
        parser = argparse.ArgumentParser(description='Saved q178 evidence only; live BF16 mechanism dispatch is permanently stopped')
        parser.add_argument('kind', choices=tuple(SAVED_ROLES))
        for name in ('inputs', 'inputs-sha256', 'output-root', 'device'):
            parser.add_argument('--' + name, required=True)
        args = parser.parse_args(values[1:])
        analyze_saved(args.kind, args.inputs, args.inputs_sha256, args.output_root, device=args.device)
        return 0
    if values and values[0] in ('live', 'profile', 'kernel-load'):
        raise RuntimeError('BF16 operator-mechanism execution is permanently stopped; saved evidence only')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, help='saved diagnosis.json and selected_tensors.pt directory')
    parser.add_argument('--execution', required=True, help='original runtime execution matrix matching the saved report')
    parser.add_argument('--execution-git-sha256', required=True, help='separate SHA256 of the pinned Git execution member')
    parser.add_argument('--repository', required=True, help='local Git checkout containing the pinned input blob')
    parser.add_argument('--commit', required=True, help='full immutable commit of the historical execution matrix')
    parser.add_argument('--output-root', required=True, help='derived output directory separate from all inputs')
    parser.add_argument('--device', required=True, help='explicit available cuda:0; no CPU fallback')
    args = parser.parse_args(values)
    print(analyze(args.raw, args.execution, args.repository, args.commit, args.execution_git_sha256, args.output_root, device=args.device))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
