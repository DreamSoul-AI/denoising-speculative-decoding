import json
import pytest
import yaml
from test_window_observer import ROOT
from studies._shared.structure.config import spec_for


def test_4b_control_uses_historical_recipe_on_same_unpruned_chain():
    cfg=yaml.safe_load((ROOT/'studies/output_space_4b_input_control/experiment_config.yaml').read_text(encoding='utf-8'))
    actual=cfg['algorithm']['config']['spec']
    historical=spec_for('mean_plus_gamma',beta=.9999,lam=.1,gamma=1.)
    assert actual['ema']==historical['ema']
    assert actual['tree']=={'mode':'static','branches':[1,1],'pruning':False}
    assert actual['num_masks']==2 and cfg['algorithm']['config']['eval']['warmup'] is False


@pytest.mark.parametrize('name,n',[('output_space_4b_recheck_v2',26),('output_space_4b_holdout',4)])
def test_4b_control_reuses_fixed_original_manifests(name,n,historical_file):
    study=historical_file('studies/output_space_4b_input_control/docs/prompts_manifest.json').parent
    copied=json.loads((study/'prompts_manifest.json').read_text())[name]
    original=json.loads(historical_file('studies/'+name+'/docs/prompts_manifest.json').read_text())
    assert copied==original and len(copied['rendered_prompts'])==n
