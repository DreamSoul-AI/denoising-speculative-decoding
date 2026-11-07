from studies._shared.flow.write import run
__all__ = ['run']


def write_plot(body, path):
    """Render already supplied counts to an explicit new external PNG; no evidence certification."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from spec_mtp.structure.artifact.paths import validate_output_root
    from studies.qwen3_4b_bf16_five_methods.structure.plots import paired_figure

    path = Path(path).resolve()
    validate_output_root(path.parent, Path(__file__).resolve().parents[4])
    if path.exists() or path.is_symlink():
        raise FileExistsError('never replace a saved paired figure')
    figure = paired_figure(body)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=160)
    finally:
        plt.close(figure)


def write_published(input_path, repository, commit, input_git_sha256, output_root):
    import json
    from pathlib import Path
    import matplotlib
    import matplotlib.pyplot as plt
    from spec_mtp.structure.artifact.layout import validate_study_root
    from spec_mtp.structure.artifact.paths import validate_output_root
    from spec_mtp.structure.artifact.provenance import file_sha256
    from studies.qwen3_4b_bf16_five_methods.flow.process.run import read_published
    from studies.qwen3_4b_bf16_five_methods.structure import plots

    output = validate_study_root(validate_output_root(output_root,
        Path(input_path).resolve(strict=True).parent, Path(repository).resolve(strict=True)))
    if output.exists() or output.is_symlink():
        raise FileExistsError('paired plots require a new external output directory')
    rendering = {Path(__file__).resolve(): file_sha256(__file__),
                 Path(plots.__file__).resolve(): file_sha256(plots.__file__)}
    body, receipt, bound = read_published(input_path, repository, commit, input_git_sha256)
    bound.update(rendering)
    if any(file_sha256(p) != digest for p, digest in bound.items()):
        raise ValueError('saved input or plot source changed before rendering')
    figure = plots.paired_figure(body)
    try:
        if any(file_sha256(p) != digest for p, digest in bound.items()):
            raise ValueError('saved input or plot source changed before rendering')
        output.mkdir(parents=True, exist_ok=False)
        target = output / 'paired_outcomes.png'
        with target.open('xb') as stream:
            figure.savefig(stream, format='png', dpi=160)
        if any(file_sha256(p) != digest for p, digest in bound.items()):
            raise ValueError('saved input or plot source changed during rendering')
        receipt.update(figure={'path': target.name, 'sha256': file_sha256(target)},
                       rendering_sources={str(p): digest for p, digest in bound.items()},
                       rendering_environment={'backend': 'Agg', 'matplotlib': matplotlib.__version__, 'dpi': 160})
        with (output / 'plot_complete.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return target
    finally:
        plt.close(figure)


def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Render explicitly pinned saved paired counters')
    for flag in ('input', 'repository', 'output-root'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--input-git-sha256', required=True)
    args = parser.parse_args(argv)
    print(write_published(args.input, args.repository, args.commit, args.input_git_sha256, args.output_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
