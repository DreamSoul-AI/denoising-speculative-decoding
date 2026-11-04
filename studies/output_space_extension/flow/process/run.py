"""Study process and explicit offline audit of saved transfer observations.

    python -m studies.output_space_extension.flow.process.run --help
"""
from studies._shared.flow.process import run

__all__ = ['run']

EXECUTION_MEMBER = 'studies/output_space_extension/docs/execution_v1.json'
SOURCE_DIGEST = '670c744e0535dcad'


def analyze(raw_root, execution, repository, commit, execution_git_sha256, output_root):
    """Derive a new audit from pinned inputs; never re-sign the saved Run."""
    import json
    from pathlib import Path
    import torch
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    from studies._shared.structure import extension_diagnostics

    raw_root = Path(raw_root).resolve()
    repository = Path(repository).resolve()
    execution = Path(execution).resolve()
    output_root = validate_output_root(output_root, raw_root, execution.parent, repository)
    target = output_root / 'slot_diagnosis.json'
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    report_path = raw_root / 'diagnosis.json'
    tensor_path = raw_root / 'selected_tensors.pt'
    for member in (report_path, tensor_path):
        member.resolve().relative_to(raw_root)
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
    tool_sha, helper_sha = file_sha256(tool_path), file_sha256(helper_path)
    raw = torch.load(tensor_path, map_location='cpu', weights_only=True)
    findings = extension_diagnostics.analyze_saved_transfers(report, data, raw)
    if (file_sha256(report_path) != report_sha or file_sha256(tensor_path) != tensor_sha
            or file_sha256(execution) != matrix_sha or file_sha256(tool_path) != tool_sha
            or file_sha256(helper_path) != helper_sha):
        raise ValueError('analysis input changed during reading')
    result = {
        'schema': 1,
        'kind': 'post hoc offline second-slot expansion and propagation audit; no additional model forwards',
        'note': 'First-slot hypotheses were predeclared; second-slot scalar audit is post hoc from saved logits.',
        'source_digest': report['source_digest'],
        'raw_report_sha256': report_sha, 'raw_tensor_sha256': tensor_sha,
        'historical_input': {'commit': commit, 'path': EXECUTION_MEMBER, 'sha256': execution_git_sha256},
        'runtime_execution_sha256': matrix_sha, 'execution_content_equal': True,
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
    parser.add_argument('--execution', required=True, help='original runtime execution matrix matching the saved report')
    parser.add_argument('--execution-git-sha256', required=True, help='separate SHA256 of the pinned Git execution member')
    parser.add_argument('--repository', required=True, help='local Git checkout containing the pinned input blob')
    parser.add_argument('--commit', required=True, help='full immutable commit of the historical execution matrix')
    parser.add_argument('--output-root', required=True, help='derived output directory separate from all inputs')
    args = parser.parse_args(argv)
    print(analyze(args.raw, args.execution, args.repository, args.commit, args.execution_git_sha256, args.output_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
