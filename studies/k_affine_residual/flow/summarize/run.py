"""Completeness and frozen provenance are gates; lower error is not a gate."""
from ...structure.affine import check_run


def run(ctx):
    if not check_run(ctx.layout.assets_dir)['valid']:
        raise ValueError('invalid affine observations')
