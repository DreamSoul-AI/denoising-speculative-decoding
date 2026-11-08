"""Study process and explicit offline followup transfer audit.

    python -m studies.output_space_followup.flow.process.run --help
"""
from studies._shared.flow.process import run

__all__ = ['run']

EXECUTION_MEMBER = 'studies/output_space_followup/docs/execution_v1.json'
SOURCE_DIGEST = '670c744e0535dcad'


def analyze(raw_root, execution, repository, commit, execution_git_sha256, output_root):
    """Bind original runtime/Git inputs separately and write one new CUDA audit."""
    from datetime import datetime, timezone
    import json
    import multiprocessing
    import os
    from pathlib import Path
    import uuid
    import torch
    from spec_mtp import __version__
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    from studies._shared.structure import extension_diagnostics

    raw_root, repository, execution = (Path(value).resolve() for value in (raw_root, repository, execution))
    output_root = validate_output_root(output_root, raw_root, execution.parent, repository)
    target = output_root / 'followup_analysis.json'
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    report_path, tensor_path = raw_root / 'diagnosis.json', raw_root / 'selected_tensors.pt'
    for member in (report_path, tensor_path):
        member.resolve().relative_to(raw_root)
    report_sha = file_sha256(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('quality_passed') is not True or report.get('source_digest') != SOURCE_DIGEST:
        raise ValueError('saved followup report has unverified or incompatible source identity')
    tensor_sha = file_sha256(tensor_path)
    if tensor_sha != report['raw_tensor_sha256']:
        raise ValueError('saved tensor SHA256 differs from followup report')
    matrix_sha = file_sha256(execution)
    if matrix_sha != report['source_binding']['matrix_sha256']:
        raise ValueError('runtime execution SHA256 differs from followup report')
    matrix = read_historical_bytes(repository, commit, EXECUTION_MEMBER, execution_git_sha256)
    data = json.loads(execution.read_bytes())
    if data != json.loads(matrix):
        raise ValueError('runtime execution and pinned Git member have different content')
    if data.get('source_digest') != report['source_digest']:
        raise ValueError('historical execution and saved followup sources differ')
    tool_path, helper_path = Path(__file__), Path(extension_diagnostics.__file__)
    tool_sha, helper_sha = file_sha256(tool_path), file_sha256(helper_path)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for saved followup tensor analysis')
    device = torch.device('cuda', torch.cuda.current_device())
    device_name = torch.cuda.get_device_name(device)
    print(json.dumps({
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
        'level': 'INFO', 'event': 'followup_analysis.device', 'logger': __name__,
        'message': 'Saved followup tensor analysis uses CUDA',
        'app_name': 'spec_mtp', 'app_version': __version__, 'runtime_id': uuid.uuid4().hex,
        'process_name': multiprocessing.current_process().name, 'pid': os.getpid(),
        'device': str(device), 'device_name': device_name,
    }), flush=True)
    tensors = torch.load(tensor_path, map_location=device, weights_only=True)
    findings = extension_diagnostics.analyze_saved_followup(report, data, tensors)
    if (file_sha256(report_path) != report_sha or file_sha256(tensor_path) != tensor_sha
            or file_sha256(execution) != matrix_sha or file_sha256(tool_path) != tool_sha
            or file_sha256(helper_path) != helper_sha):
        raise ValueError('analysis input changed during reading')
    result = {
        'schema': 1, 'kind': 'offline scope/state/slot summary; no model forward',
        'source_digest': report['source_digest'],
        'raw_report_sha256': report_sha, 'raw_tensor_sha256': tensor_sha,
        'historical_input': {'commit': commit, 'path': EXECUTION_MEMBER, 'sha256': execution_git_sha256},
        'runtime_execution_sha256': matrix_sha, 'execution_content_equal': True,
        'analysis_device': {'device': str(device), 'name': device_name},
        'analysis_provenance': {'module': __name__, 'tool_sha256': tool_sha, 'helper_sha256': helper_sha},
        'findings': findings,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output_root.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
    return target


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True, help='saved diagnosis.json and selected_tensors.pt directory')
    parser.add_argument('--execution', required=True, help='original runtime matrix matching the saved report')
    parser.add_argument('--execution-git-sha256', required=True, help='SHA256 of the pinned Git matrix member')
    parser.add_argument('--repository', required=True, help='local Git repository containing the pinned input')
    parser.add_argument('--commit', required=True, help='full immutable historical matrix commit')
    parser.add_argument('--output-root', required=True, help='new external derived output directory')
    args = parser.parse_args(argv)
    print(analyze(args.raw, args.execution, args.repository, args.commit, args.execution_git_sha256, args.output_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
