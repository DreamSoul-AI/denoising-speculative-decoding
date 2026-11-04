"""Publish only complete postcommit measurements; do not infer scientific support."""
from studies.k_affine_residual.structure.evaluation import check_matrix


def run(ctx):
    report = ctx.state['k_affine_residual']
    check_matrix([{'seed': ctx.control.seed, 'status': 'succeeded',
                   'k_affine_residual': report}], [ctx.control.seed])
    ctx.state['result_draft']['k_affine_residual'] = report
