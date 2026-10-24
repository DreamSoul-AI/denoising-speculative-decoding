"""Check every declared seed using existing results, without launching a run."""
import json
from pathlib import Path

from rpipe.structure.artifact.index import load_index
from rpipe.structure.make.expand import expand_patches, load_study_yaml
from studies.k_affine_residual.structure.evaluation import check_matrix


def run(ctx):
    if ctx.scope != 'study':
        return
    study = Path(ctx.study_dir)
    declaration = load_study_yaml(study)
    seeds = [patch['seed'] for patch in expand_patches(declaration)]
    records = []
    for experiment in load_index(study).get('experiments', []):
        for item in experiment.get('runs', []):
            path = study / 'runs' / (item.get('run_dir') or item['id']) / 'result.json'
            result = json.loads(path.read_text(encoding='utf-8'))
            records.append({'seed': item['seed'], 'status': result.get('status'),
                            'k_affine_residual': result.get('k_affine_residual')})
    check_matrix(records, seeds)
