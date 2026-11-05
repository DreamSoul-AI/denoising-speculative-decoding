"""Use the SpecMTP write phase; no Study-local implementation."""
from studies._shared.flow.write import run

__all__ = ['run']

MEMBERS = {'execution': 'studies/output_space_cross_task/docs/execution_v1.json',
           'canonical': 'studies/output_space_cross_task/docs/canonical_analysis_v1.json'}


def write_plot(execution, canonical, repository, commit, execution_git_sha256, canonical_git_sha256, output_root):
    import json
    from pathlib import Path
    from spec_mtp.structure.artifact import layout, paths, provenance
    from studies.output_space_cross_task.structure import plots
    from studies._shared.flow.write.run import publish_new_completion

    sources = {}
    for module in (plots, layout, paths, provenance):
        filename = module.__file__
        assert filename is not None
        sources[Path(filename).resolve()] = provenance.file_sha256(filename)
    sources[Path(__file__).resolve()] = provenance.file_sha256(__file__)
    completion_source = Path(publish_new_completion.__code__.co_filename).resolve()
    sources[completion_source] = provenance.file_sha256(completion_source)
    repository = Path(repository).resolve(strict=True)
    actual = {'execution': Path(execution).resolve(strict=True), 'canonical': Path(canonical).resolve(strict=True)}
    output = layout.validate_study_root(paths.validate_output_root(output_root, repository,
                                       *(path.parent for path in actual.values())))
    if output.exists() or output.is_symlink():
        raise FileExistsError('cross-task plot requires a new external output root')
    digests = {role: provenance.file_sha256(path) for role, path in actual.items()}
    hashes = {'execution': execution_git_sha256, 'canonical': canonical_git_sha256}
    bodies = {}
    for role, path in actual.items():
        archived = provenance.read_historical_bytes(repository, commit, MEMBERS[role], hashes[role])
        bodies[role] = json.loads(path.read_bytes())
        if bodies[role] != json.loads(archived):
            raise ValueError(f'actual {role} report differs from its fixed Git original')
    if bodies['execution'].get('source_digest') != '670c744e0535dcad':
        raise ValueError('cross-task report has a different original source identity')

    def verify():
        if (any(provenance.file_sha256(path) != digests[role] for role, path in actual.items())
                or any(provenance.file_sha256(path) != digest for path, digest in sources.items())):
            raise ValueError('cross-task input or rendering source changed during execution')

    verify()
    figure = plots.acceptance_figure(bodies['execution'], bodies['canonical'])
    try:
        verify()
        output.mkdir(parents=True, exist_ok=False)
        from spec_mtp.structure.system.logger import Logger
        Logger(output / 'assets').info('CPU saved-scalar cross-task rendering; no tensor or model forward')
        target = output / 'acceptance_v1.png'
        with target.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=170)
        verify()
        import matplotlib
        receipt = {'schema': 1, 'scope': 'CPU saved native counts and separate canonical q241 comparison',
            'historical_inputs': [{'role': role, 'commit': commit, 'path': MEMBERS[role], 'sha256': hashes[role]}
                                  for role in actual],
            'runtime_inputs': {role: {'path': str(path), 'sha256': digests[role]} for role, path in actual.items()},
            'original_canonical_binding': {key: bodies['canonical'][key]
                for key in ('raw_report_sha256', 'raw_tensor_sha256', 'analysis_sha256') if key in bodies['canonical']},
            'rendering_sources': {str(path): digest for path, digest in sources.items()},
            'rendering_environment': {'matplotlib': matplotlib.__version__, 'backend': 'Agg', 'dpi': 170},
            'figure': {'path': target.name, 'sha256': provenance.file_sha256(target)}}
        payload = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        publish_new_completion(output / 'plot_complete.json', payload)
        return target
    finally:
        figure.clear()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Render fixed cross-task native and canonical saved counts')
    for name in ('execution', 'canonical', 'repository', 'commit', 'execution-git-sha256', 'canonical-git-sha256', 'output-root'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    print(write_plot(args.execution, args.canonical, args.repository, args.commit, args.execution_git_sha256,
                     args.canonical_git_sha256, args.output_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
