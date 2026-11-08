from studies._shared.flow.process import run
__all__ = ['run']

MEMBER = 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json'


def read_published(input_path, repository, commit, input_git_sha256, *, original_study=None):
    """Reaggregate saved counters; never certify a native Run or re-sign its source gate."""
    import json
    import re
    from pathlib import Path
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    from studies.qwen3_4b_bf16_five_methods.structure import readout

    path = Path(input_path).resolve(strict=True)
    repository = Path(repository).resolve(strict=True)
    sources = {Path(__file__).resolve(): file_sha256(__file__),
               Path(readout.__file__).resolve(): file_sha256(readout.__file__)}
    bound = {path: file_sha256(path)}
    archived = json.loads(read_historical_bytes(repository, commit, MEMBER, input_git_sha256))
    body = json.loads(path.read_bytes())
    if body != archived:
        raise ValueError('saved counters differ from the explicitly pinned Git report')
    result = readout.verify(path)
    if original_study is not None:
        owner = Path(original_study).resolve(strict=True)
        run_ids = [cell.get('run_id') for cell in body['cells'].values()]
        if (any(not isinstance(item, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', item)
                for item in run_ids) or len(set(run_ids)) != len(run_ids)):
            raise ValueError('original Run IDs must be safe and unique')
        required = {'config.yaml', 'result.json', 'assets/decode_observations.json',
                    'assets/decode_inputs.json', 'assets/pairing_contract.json', 'assets/spec_mtp_sources.json'}
        for cell in body['cells'].values():
            run_id, files = cell.get('run_id'), cell.get('files_sha256')
            if not isinstance(files, dict) or not required <= set(files):
                raise ValueError('original Run identity and all six recorded file hashes are required')
            for member, expected in files.items():
                if (not isinstance(member, str) or not isinstance(expected, str)
                        or not re.fullmatch(r'[0-9a-f]{64}', expected)):
                    raise ValueError('original Run file hashes must be explicit SHA256 mappings')
                relative = Path('runs') / run_id / member
                actual = (owner / relative).resolve(strict=True)
                if relative.is_absolute() or '..' in relative.parts or not actual.is_relative_to(owner):
                    raise ValueError('original Run member escapes its explicit Study root')
                if file_sha256(actual) != expected:
                    raise ValueError('original Run file differs from the saved report')
                bound[actual] = expected
    if any(file_sha256(p) != digest for p, digest in (bound | sources).items()):
        raise ValueError('saved input or readout source changed during reaggregation')
    result.update(scope='Saved counters and optional recorded file hashes only; no native replay, '
                        'model/token/logit audit, or new acceptance of the original Run',
                  native_run_revalidated=False,
                  historical_input={'commit': commit, 'path': MEMBER, 'sha256': input_git_sha256},
                  runtime_inputs={str(p): digest for p, digest in bound.items()},
                  readout_sources={str(p): digest for p, digest in sources.items()},
                  original_report_identity={k: body[k] for k in ('analyzer_sha256', 'plotter_sha256') if k in body})
    return body, result, bound | sources


def write_readout(input_path, repository, commit, input_git_sha256, output_root, *, original_study=None):
    import json
    from pathlib import Path
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256

    inputs = [Path(input_path).resolve(strict=True).parent, Path(repository).resolve(strict=True)]
    if original_study is not None:
        inputs.append(Path(original_study).resolve(strict=True))
    output = validate_study_root(validate_output_root(output_root, *inputs))
    if output.exists() or output.is_symlink():
        raise FileExistsError('counter readout requires a new external output directory')
    _, receipt, bound = read_published(input_path, repository, commit, input_git_sha256,
                                     original_study=original_study)
    if any(file_sha256(p) != digest for p, digest in bound.items()):
        raise ValueError('saved input or readout source changed before output')
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'counter_readout.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if any(file_sha256(p) != digest for p, digest in bound.items()):
        raise ValueError('saved input or readout source changed while writing output')
    return target


def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Reaggregate explicitly pinned saved counters; no native revalidation')
    for flag in ('input', 'repository', 'output-root'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--input-git-sha256', required=True)
    parser.add_argument('--original-study', type=Path)
    args = parser.parse_args(argv)
    print(write_readout(args.input, args.repository, args.commit, args.input_git_sha256,
                        args.output_root, original_study=args.original_study))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
