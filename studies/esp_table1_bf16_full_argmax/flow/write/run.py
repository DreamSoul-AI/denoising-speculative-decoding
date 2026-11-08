"""Plot verified full480 tau and accepted counts; no timing or AR success gate."""
from studies._shared.flow.write import run as run

from pathlib import Path
import torch  # noqa: F401 - Windows native import order
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_plot(body, path):
    from spec_mtp.structure.artifact.paths import validate_output_root

    path = validate_output_root(path, Path(__file__).resolve().parents[4])
    cells=body['cells']
    assert body['all_cells_observed'] and body['quality_passed']
    assert [(c['BC'],c['num_masks']) for c in cells]==[(10,1),(30,1),(60,1),(60,2)]
    assert all(c['n']==480 and c['argmax_all_passed'] for c in cells)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    try:
        positions=list(range(len(cells)))
        labels=[f"{c['BC']}/{c['num_masks']}" for c in cells]
        for offset,(key,label) in enumerate((('pooled_tau','Pooled'),('prompt_mean_tau','Prompt mean'),('category_mean_pooled_tau','Category mean'))):
            axes[0].bar([x+(offset-1)*.24 for x in positions],[c[key] for c in cells],width=.24,label=label)
        axes[0].scatter([1,2,3],[1.63,1.74,1.74],marker='D',color='black',label='Paper target',zorder=5)
        axes[0].set_ylabel('Committed tokens / decode call (tau)')
        axes[0].legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8)
        axes[1].bar(positions,[c['accepted'] for c in cells],color='#6854a3')
        axes[1].set_ylabel('Accepted draft tokens\n(before EOS/budget trimming)')
        for axis in axes:
            axis.set_xticks(positions,labels)
            axis.set_xlabel('BC / mask count (60/2 is control)')
            axis.grid(axis='y',alpha=.2)
            axis.set_axisbelow(True)
        fig.suptitle('Qwen3-8B / BF16 / T0 / SpecBench 480; all committed-token argmax checks passed')
        path=Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError('plot output already exists')
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=160)
    finally:
        plt.close(fig)


def main(argv=None):
    import argparse
    from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout

    parser = argparse.ArgumentParser(description='Plot saved four-cell argmax evidence without a model forward')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--report-sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    report = args.report.resolve(strict=True)
    session = SavedReadout(report.parent, args.output_root)
    if session.sha(report) != args.report_sha256:
        raise ValueError('saved evidence SHA256 differs from explicit binding')
    session.finish(figure=session.read_json(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
