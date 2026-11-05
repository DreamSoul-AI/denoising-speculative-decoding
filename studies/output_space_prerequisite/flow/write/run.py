"""Study write phase and bound read-only rendering of the fixed q171 report."""
from studies._shared.flow.write import run

__all__ = ['run']

SOURCE_DIGEST = '670c744e0535dcad'
EXECUTION_MEMBER = 'studies/output_space_prerequisite/docs/execution_v1.json'


def write_plot(input_path, repository, commit, input_git_sha256, output_root):
    """Render a pinned saved report in a new external root; never change an old Run."""
    import json
    from pathlib import Path
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256, read_historical_bytes
    from studies.output_space_prerequisite.structure import plots

    primary, repository = Path(input_path).resolve(strict=True), Path(repository).resolve(strict=True)
    output = validate_study_root(validate_output_root(output_root, primary.parent, repository))
    if output.exists() or output.is_symlink():
        raise FileExistsError('q171 plot requires a new external output directory')
    input_sha = file_sha256(primary)
    archived = read_historical_bytes(repository, commit, EXECUTION_MEMBER, input_git_sha256)
    data = json.loads(primary.read_bytes())
    if data != json.loads(archived):
        raise ValueError('runtime plot report differs from the pinned Git report')
    if data.get('study_quality_passed') is not True or data.get('source_digest') != SOURCE_DIGEST:
        raise ValueError('q171 plot report lacks its original passing source/quality identity')
    sources = {Path(__file__).resolve(): file_sha256(__file__),
               Path(plots.__file__).resolve(): file_sha256(plots.__file__)}

    def verify():
        if file_sha256(primary) != input_sha or any(file_sha256(path) != digest for path, digest in sources.items()):
            raise ValueError('plot input or rendering source changed during execution')

    verify()
    figure = plots.acceptance_figure(data)
    try:
        verify()
        output.mkdir(parents=True, exist_ok=False)
        from spec_mtp.structure.system.logger import Logger
        Logger(output / 'assets').info('render saved q171 acceptance counts with Agg; no new model or statistic')
        target = output / 'acceptance_v1.png'
        with target.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=180)
        verify()
        import matplotlib
        result = {
            'schema': 1, 'scope': 'fixed q171 saved scalar rendering; no model forwards or statistical recomputation',
            'source_digest': data['source_digest'],
            'historical_input': {'commit': commit, 'path': EXECUTION_MEMBER, 'sha256': input_git_sha256},
            'runtime_input': {'path': str(primary), 'sha256': input_sha},
            'rendering_sources': {str(path): digest for path, digest in sources.items()},
            'rendering_environment': {'matplotlib': matplotlib.__version__, 'backend': 'Agg', 'dpi': 180,
                'canvas': type(figure.canvas).__module__ + '.' + type(figure.canvas).__name__},
            'figure': {'path': target.name, 'sha256': file_sha256(target)},
        }
        if 'source_binding' in data:
            result['original_source_binding'] = data['source_binding']
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
    for name in ('input', 'repository', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--input-git-sha256', required=True)
    args = parser.parse_args(argv)
    print(write_plot(args.input, args.repository, args.commit, args.input_git_sha256, args.output_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
