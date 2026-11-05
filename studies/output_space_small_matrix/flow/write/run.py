"""Study write phase and explicitly bound saved-report figures."""
from studies._shared.flow.write import run

__all__ = ['run']

SOURCE_DIGEST = '670c744e0535dcad'
MEMBERS = {
    'acceptance': 'studies/output_space_small_matrix/docs/execution_v1.json',
    'windows': 'studies/output_space_small_matrix/docs/window_diagnosis_v1.json',
}
SCALE_MEMBER = 'studies/output_space_small_matrix/docs/scale_diagnosis_v1.json'


def write_plots(kind, input_path, repository, commit, input_git_sha256, output_root, *,
                scale_path=None, scale_git_sha256=None):
    """Render pinned historical values into a new external directory, without re-signing inputs."""
    import json
    from pathlib import Path
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    from studies.output_space_small_matrix.structure import plots

    if kind not in MEMBERS:
        raise ValueError('plot kind must be acceptance or windows')
    if (kind == 'windows') != (scale_path is not None and scale_git_sha256 is not None):
        raise ValueError('windows plots require an explicit scale report and its Git SHA256')
    if kind == 'acceptance' and (scale_path is not None or scale_git_sha256 is not None):
        raise ValueError('acceptance plots do not accept a scale report')
    primary, repository = Path(input_path).resolve(strict=True), Path(repository).resolve(strict=True)
    scale = Path(scale_path).resolve(strict=True) if scale_path is not None else None
    output = validate_study_root(validate_output_root(output_root, primary.parent, repository,
                                *([scale.parent] if scale is not None else [])))
    if output.exists() or output.is_symlink():
        raise FileExistsError('plots require a new external output directory')
    inputs: dict[Path, str] = {}
    historical: list[dict[str, str]] = []

    def read(path, member, expected):
        inputs[path] = file_sha256(path)
        archived = read_historical_bytes(repository, commit, member, expected)
        data = json.loads(path.read_bytes())
        if data != json.loads(archived):
            raise ValueError('runtime plot report differs from the pinned Git report')
        historical.append({'commit': commit, 'path': member, 'sha256': expected})
        return data

    data = read(primary, MEMBERS[kind], input_git_sha256)
    companion = read(scale, SCALE_MEMBER, scale_git_sha256) if scale is not None else None
    quality_key = 'study_quality_passed' if kind == 'acceptance' else 'quality_passed'
    if data.get(quality_key) is not True or (companion is not None and companion.get('quality_passed') is not True):
        raise ValueError('saved plot reports must pass their original quality gate')
    binding = data.get('source_binding', {})
    if not isinstance(binding, dict):
        raise ValueError('historical plot source binding must be a mapping')
    if (data.get('source_digest') if kind == 'acceptance' else binding.get('source_digest')) != SOURCE_DIGEST:
        raise ValueError('plot report has incompatible historical source identity')
    if companion is not None:
        other = companion.get('source_binding', {})
        if (not isinstance(other, dict) or other.get('source_digest') != SOURCE_DIGEST
                or other.get('observer_sha256') != inputs[primary]
                or other.get('tensor_sha256') != data.get('raw_tensor_sha256')
                or companion.get('model_digest') != data.get('model_digest')):
            raise ValueError('scale report is not bound to the actual window report and original tensors/model')
    sources = {Path(__file__).resolve(): file_sha256(__file__),
               Path(plots.__file__).resolve(): file_sha256(plots.__file__)}

    def verify():
        if any(file_sha256(path) != digest for path, digest in (inputs | sources).items()):
            raise ValueError('plot input or rendering source changed during execution')

    verify()
    figure = plots.acceptance_figure(data) if kind == 'acceptance' else plots.windows_figure(data, companion)
    try:
        verify()
        output.mkdir(parents=True, exist_ok=False)
        from spec_mtp.structure.system.logger import Logger
        logger = Logger(output / 'assets')
        logger.info(f'render saved {kind} scalars with Agg; no new numerical computation')
        target = output / ('acceptance_v1.png' if kind == 'acceptance' else 'windows_v1.png')
        with target.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=180)
        verify()
        import matplotlib
        result = {
            'schema': 1, 'kind': kind, 'scope': 'saved scalar rendering; no model forwards or numerical recomputation',
            'source_digest': SOURCE_DIGEST, 'historical_inputs': historical,
            'runtime_inputs': {str(path): digest for path, digest in inputs.items()},
            'original_source_bindings': {str(path): report['source_binding']
                for path, report in ((primary, data), (scale, companion))
                if report is not None and 'source_binding' in report},
            'rendering_sources': {str(path): digest for path, digest in sources.items()},
            'rendering_environment': {'canvas': type(figure.canvas).__module__ + '.' + type(figure.canvas).__name__,
                                      'matplotlib': matplotlib.__version__, 'backend': 'Agg', 'dpi': 180},
            'figure': {'path': target.name, 'sha256': file_sha256(target)},
        }
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        with (output / 'plot_complete.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(payload)
        return target
    finally:
        figure.clear()


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=tuple(MEMBERS))
    for name in ('input', 'repository', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--input-git-sha256', required=True)
    parser.add_argument('--scale', type=Path)
    parser.add_argument('--scale-git-sha256')
    args = parser.parse_args(argv)
    print(write_plots(args.kind, args.input, args.repository, args.commit, args.input_git_sha256,
                      args.output_root, scale_path=args.scale, scale_git_sha256=args.scale_git_sha256))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
