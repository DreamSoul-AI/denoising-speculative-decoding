"""Recompute from complete raw latent pairs rather than execute return values."""
from ...structure.affine import ARMS, SPLITS, check_run, write


def run(ctx):
    result = check_run(ctx.layout.assets_dir)
    metrics = {}
    for split in SPLITS:
        for arm in ARMS:
            for name, value in result['splits'][split]['arms'][arm].items():
                metrics[f'{split}_{arm}_{name}'] = value
    ctx.state['collected'] = dict(metrics=metrics, observations=[])
    write(ctx.layout.assets_dir / 'affine/analysis.json', result)
