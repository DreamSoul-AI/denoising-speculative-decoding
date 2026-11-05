"""Study process and pinned read-only transfer/cross-prompt analysis."""
from studies._shared.flow.process import run

__all__ = ['run']

SOURCE_DIGEST = '670c744e0535dcad'
REPORT_MEMBERS = {
    'transfers': 'studies/output_space_small_matrix/docs/transfer_diagnosis_v1.json',
    'cross-prompt': 'studies/output_space_small_matrix/docs/cross_prompt_diagnosis_v1.json',
}
OBSERVER_MEMBER = 'studies/output_space_small_matrix/docs/window_diagnosis_v1.json'


def analyze(kind, raw_root, repository, commit, report_git_sha256, output_root, *,
            device, original_root=None, observer_git_sha256=None):
    """Derive a new report from pinned saved observations, without re-signing them."""
    if not __debug__:
        raise RuntimeError('saved tensor audit requires Python assertions enabled')
    from datetime import datetime, timezone
    import json
    import multiprocessing
    import os
    from pathlib import Path
    import uuid
    import torch
    from spec_mtp import __version__
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    from studies._shared.structure import extension_diagnostics
    from studies._shared.structure.sources import package_files

    if kind not in REPORT_MEMBERS:
        raise ValueError('analysis must be transfers or cross-prompt')
    if kind == 'transfers' and (original_root is None or observer_git_sha256 is None):
        raise ValueError('transfer analysis requires original observer inputs and its pinned Git SHA256')
    if kind == 'cross-prompt' and (original_root is not None or observer_git_sha256 is not None):
        raise ValueError('cross-prompt analysis does not accept unrelated original observer inputs')
    raw_root, repository = Path(raw_root).resolve(), Path(repository).resolve()
    original = Path(original_root).resolve() if original_root is not None else None
    output = validate_output_root(output_root, raw_root, repository, *([original] if original is not None else []))
    output = validate_study_root(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('analysis requires a new external output directory')
    inputs: dict[Path, str] = {}
    historical: list[dict[str, str]] = []

    def member(root, name):
        path = (root / name).resolve(strict=True)
        path.relative_to(root)
        inputs[path] = file_sha256(path)
        return path

    def pinned_report(path, archived_member, expected_sha):
        raw = read_historical_bytes(repository, commit, archived_member, expected_sha)
        report = json.loads(path.read_bytes())
        if report != json.loads(raw):
            raise ValueError('runtime report and pinned Git member have different content')
        if report.get('quality_passed') is not True:
            raise ValueError('saved report does not pass its original quality gate')
        historical.append({'commit': commit, 'path': archived_member, 'sha256': expected_sha})
        return report

    report_path, tensor_path = member(raw_root, 'diagnosis.json'), member(raw_root, 'selected_tensors.pt')
    report = pinned_report(report_path, REPORT_MEMBERS[kind], report_git_sha256)
    if report.get('source_digest') != SOURCE_DIGEST or inputs[tensor_path] != report['raw_tensor_sha256']:
        raise ValueError('saved report source identity or raw tensor SHA256 differs')
    old_path = None
    if original is not None:
        observer_path, old_path = member(original, 'diagnosis.json'), member(original, 'selected_tensors.pt')
        observer = pinned_report(observer_path, OBSERVER_MEMBER, observer_git_sha256)
        binding = report.get('source_binding')
        if not isinstance(binding, dict):
            raise ValueError('transfer report is missing its original observer binding')
        if (inputs[observer_path] != binding['observer_sha256']
                or inputs[old_path] != observer['raw_tensor_sha256'] or inputs[old_path] != binding['tensor_sha256']
                or observer['source_binding']['source_digest'] != report['source_digest']):
            raise ValueError('original observer report/tensor binding differs from the transfer report')
    source_files = {path: file_sha256(path) for path in (Path(__file__), Path(extension_diagnostics.__file__))}
    package_sources = package_files()
    actual_device = torch.device(device)
    if actual_device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('saved tensor analysis requires an available CUDA device')
    if actual_device.index is None or actual_device.index >= torch.cuda.device_count():
        raise ValueError('analysis device must name an available CUDA index')
    print(json.dumps({
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
        'level': 'INFO', 'event': 'small_matrix_analysis.device', 'logger': __name__,
        'message': 'Read-only saved tensor analysis uses CUDA', 'app_name': 'spec_mtp',
        'app_version': __version__, 'runtime_id': uuid.uuid4().hex,
        'process_name': multiprocessing.current_process().name, 'pid': os.getpid(),
        'device': str(actual_device), 'device_name': torch.cuda.get_device_name(actual_device),
    }), flush=True)
    tensors = torch.load(tensor_path, map_location=actual_device, weights_only=True)
    if kind == 'transfers':
        assert old_path is not None
        previous = torch.load(old_path, map_location=actual_device, weights_only=True)
        findings = extension_diagnostics.analyze_small_matrix_transfers(report, tensors, previous)
    else:
        findings = extension_diagnostics.analyze_small_matrix_cross_prompt(report, tensors)
    if (any(file_sha256(path) != sha for path, sha in (inputs | source_files).items())
            or package_files() != package_sources):
        raise ValueError('analysis input or computation source changed during reading')
    result = {
        'schema': 1, 'kind': kind, 'scope': 'read-only derived audit; no new model forwards',
        'source_digest': report['source_digest'],
        'historical_inputs': historical,
        'runtime_inputs': {str(path): sha for path, sha in inputs.items()},
        'analysis_provenance': {'module': __name__, 'files': {str(path): sha for path, sha in source_files.items()},
                                'package_sources': package_sources},
        'analysis_device': str(actual_device), 'findings': findings,
    }
    if 'source_binding' in report:
        result['original_source_binding'] = report['source_binding']
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    target = output / ('transfer_path_diagnosis.json' if kind == 'transfers' else 'cross_prompt_path_diagnosis.json')
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
    return target


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=tuple(REPORT_MEMBERS))
    for name in ('raw', 'repository', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--report-git-sha256', required=True)
    parser.add_argument('--device', required=True, help='explicit CUDA index, for example cuda:0')
    parser.add_argument('--original', type=Path)
    parser.add_argument('--observer-git-sha256')
    args = parser.parse_args(argv)
    print(analyze(args.kind, args.raw, args.repository, args.commit, args.report_git_sha256,
                  args.output_root, device=args.device, original_root=args.original,
                  observer_git_sha256=args.observer_git_sha256))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
