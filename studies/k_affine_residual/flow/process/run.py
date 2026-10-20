"""Only the declared current index can finalise this single-Run Study."""
from ...structure.affine import check_run, read


def run(ctx):
    if ctx.scope == 'run':
        check_run(ctx.layout.assets_dir)
        return
    index = read(ctx.study_dir / 'index.json')
    runs = [run for experiment in index['experiments'] for run in experiment['runs']]
    if len(runs) != 1 or runs[0]['seed'] != 123:
        raise ValueError('registered Study expects one seed123 Run')
    root = ctx.study_dir / 'runs' / runs[0]['id']
    if read(root / 'result.json')['status'] != 'succeeded':
        raise ValueError('affine Run did not succeed')
    check_run(root / 'assets')
