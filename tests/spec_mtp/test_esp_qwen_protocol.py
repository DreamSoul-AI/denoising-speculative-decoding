import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_aggregation_keeps_call_weighting_and_quality():
    report = load('esp_protocol_report',ROOT/'studies/esp_table1_bf16_qwen_protocol/analyze_results.py')
    rows = [{'category':'a','committed':4,'decode_calls':4,'accepted':0,'exact':False},
            {'category':'a','committed':4,'decode_calls':2,'accepted':2,'exact':True},
            {'category':'b','committed':9,'decode_calls':3,'accepted':6,'exact':True}]
    out = report.summarize(rows)
    assert out['pooled_tau']==pytest.approx(17/9)
    assert out['prompt_mean_tau']==2
    assert out['category_mean_pooled_tau']==pytest.approx((8/6+3)/2)
    assert out['exact_count']==2 and out['main_calls']==12


def test_protocol_primary_cannot_be_selected_from_tau():
    report = load('esp_protocol_roles',ROOT/'studies/esp_table1_bf16_qwen_protocol/analyze_results.py')
    cells = report.ordered_cells({(60,1):{'pooled_tau':1.1},(60,2):{'pooled_tau':3.}})
    assert [(c['BC'],c['num_masks']) for c in cells]==[(10,1),(30,1),(60,1),(60,2)]
    assert cells[2]['role']=='primary' and cells[3]['role']=='control'
    assert cells[2]['pooled_tau_target_delta']==pytest.approx(-.64)
    assert cells[3]['pooled_tau_target_delta']==pytest.approx(1.26)
    assert cells[1]['pending'] and cells[1]['pooled_tau_target_delta'] is None


@pytest.mark.parametrize('failure',[False,True])
def test_launch_once_restores_native_scheduler(monkeypatch,failure):
    launch = load('launch_once_test',ROOT/'tools/launch_once.py')
    original = launch.rpipe.flow.cli.launch_jobs
    def native(argv):
        assert argv==['launch','study']
        patched=launch.rpipe.flow.cli.launch_jobs
        assert patched.func is original and patched.keywords=={'retry_failed':False}
        if failure:
            raise RuntimeError('injected launcher failure')
        return 7
    monkeypatch.setattr(launch,'native_main',native)
    if failure:
        with pytest.raises(RuntimeError,match='injected launcher failure'):
            launch.main(['launch','study'])
    else:
        assert launch.main(['launch','study'])==7
    assert launch.rpipe.flow.cli.launch_jobs is original


def test_corrected_studies_keep_full_coverage_and_mask_update():
    import yaml
    from spec_mtp.tree import node_budget_for
    for name,expected in [('esp_table1_bf16_qwen_protocol',[(60,1)]),('esp_table1_bf16_full',[(10,1),(30,1),(60,1),(60,2)])]:
        root=ROOT/'studies'/name
        config=yaml.safe_load((root/'experiment_config.yaml').read_text())
        study=yaml.safe_load((root/'study.yaml').read_text())
        specs=[cell['spec'] for cell in study['axes']['algorithm.config']]
        assert [(s['block_complexity'],s['num_masks']) for s in specs]==expected
        assert all(s['mask']=={'init':'mean_prompt','lam':.1,'update':True} and s['tree']=={'mode':'dynamic','pruning':True} for s in specs)
        assert config['model']['config']['name_or_path']=='Qwen/Qwen3-8B' and config['model']['config']['torch_dtype']=='bfloat16'
        assert config['algorithm']['config']['eval']['max_new_tokens']==256
        assert config['algorithm']['config']['eval']['temperature']==0 and config['algorithm']['config']['eval']['warmup'] is False
        assert study['seeds']==[123]
    assert node_budget_for(60,1)==29 and node_budget_for(60,2)==19


def test_historical_qwen_original_prompt_coverage(historical_file):
    import json
    for name in ('esp_table1_bf16_qwen_protocol','esp_table1_bf16_full'):
        root=historical_file('studies/'+name+'/docs/prompts_manifest.json').parents[1]
        prompts=json.loads((root/'docs/prompts_manifest.json').read_text())['rendered_prompts']
        assert len(prompts)==(480 if name.endswith('_full') else 13)
        assert len({p['question_id'] for p in prompts})==len(prompts)
