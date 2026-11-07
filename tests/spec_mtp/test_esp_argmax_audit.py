
import json

from pathlib import Path

import pytest





ROOT = Path(__file__).resolve().parents[2]

from studies._shared.structure import esp_controls as audit

def test_eos_or_budget_trim_is_not_counted_as_committed_and_wrong_owner_fails():
    selections = [{'forward':0,'row':1,'chosen':7,'argmax':7},
                  {'forward':1,'row':0,'chosen':8,'argmax':8},
                  {'forward':1,'row':2,'chosen':9,'argmax':9},
                  {'forward':1,'row':3,'chosen':10,'argmax':10}]
    stats = {'trace':[{'accepted_nodes':[2,3],'committed_tokens':[8]}], 'new_tokens':2,'committed':1}
    result = audit.bind_committed([1,2],[1,2,7,8],stats,selections)
    assert result['checked_committed_tokens']==2 and result['selected_tokens_including_trimmed_tail']==4
    selections[2]['row']=4
    with pytest.raises(AssertionError):
        audit.bind_committed([1,2],[1,2,7,8],stats,selections)

def test_tau_aggregation_does_not_gate_on_ar_and_keeps_all_weights(monkeypatch):
    from studies.esp_table1_bf16_full_argmax.structure import readout as report
    rows = [{'category':'a','committed':4,'decode_calls':4,'accepted':0,'exact':False,'argmax_passed':True,'argmax_checked_tokens':5},
            {'category':'a','committed':4,'decode_calls':2,'accepted':2,'exact':True,'argmax_passed':True,'argmax_checked_tokens':5},
            {'category':'b','committed':9,'decode_calls':3,'accepted':6,'exact':False,'argmax_passed':True,'argmax_checked_tokens':10}]
    summary = report.summarize(rows)
    assert summary['AR_exact_rate']==pytest.approx(1/3)
    assert summary['argmax_all_passed'] and summary['argmax_checked_tokens']==20
    assert summary['pooled_tau']==pytest.approx(17/9) and summary['prompt_mean_tau']==2
    assert summary['category_mean_pooled_tau']==pytest.approx((8/6+3)/2)
    rows[0]['argmax_passed']=False
    assert not report.summarize(rows)['argmax_all_passed']
    cells = report.ordered_cells({(60,1):summary,(60,2):{'pooled_tau':3}})
    assert [c['role'] for c in cells]==['primary','primary','primary','control']
    assert cells[0]['target'] is None

def test_new_full_contract_keeps_original_settings_and_480_inputs():
    import yaml
    previous = ROOT/'studies/esp_table1_bf16_full'
    current = ROOT/'studies/esp_table1_bf16_full_argmax'
    old = yaml.safe_load((previous/'experiment_config.yaml').read_text())
    new = yaml.safe_load((current/'experiment_config.yaml').read_text())
    for key in ('seed','data','model','system'):
        assert old[key]==new[key]
    for key in ('task','eval'):
        assert old['algorithm']['config'][key]==new['algorithm']['config'][key]
    before = yaml.safe_load((previous/'study.yaml').read_text())
    after = yaml.safe_load((current/'study.yaml').read_text())
    assert before['axes']==after['axes'] and before['seeds']==after['seeds']
    assert 'argmax_audit.py' not in after['provenance']['include']
    assert not (current/'argmax_audit.py').exists()
    assert audit.__name__ == 'studies._shared.structure.esp_controls'

def test_historical_full_argmax_original_inputs(historical_file):
    previous = historical_file('studies/esp_table1_bf16_full/docs/prompts_manifest.json').parents[1]
    current = historical_file('studies/esp_table1_bf16_full_argmax/docs/prompts_manifest.json').parents[1]
    assert (previous/'docs/prompts_manifest.json').read_bytes()==(current/'docs/prompts_manifest.json').read_bytes()
    assert len(json.loads((current/'docs/prompts_manifest.json').read_text())['rendered_prompts'])==480

def test_pending_report_does_not_link_unpublished_local_evidence(tmp_path, monkeypatch):
    import importlib
    report = importlib.import_module('studies.esp_table1_bf16_full_argmax.flow.process.run')
    study, output = tmp_path / 'source', tmp_path / 'readout'
    (study / 'docs').mkdir(parents=True)
    (study / 'docs/prompts_manifest.json').write_text(json.dumps({'rendered_prompts': [{'question_id': 1}]}))
    (study / 'docs/modelscope_snapshot.json').write_text('{}')
    (study / 'index.json').write_text(json.dumps({'experiments': []}))
    report.analyze(study, output, execution_root=study)
    text = (output / 'STUDY_REPORT.md').read_text(encoding='utf-8')
    assert '`protocol_pending.json`' in text and '](protocol_pending.json)' not in text
    assert not json.loads((output / 'protocol_pending.json').read_text())['quality_passed']
    assert '全四格验收=pending' in text and 'token quality passed=False' not in text
    summary = report.summarize([{'category': 'a', 'committed': 2, 'decode_calls': 1, 'accepted': 1,
        'exact': False, 'argmax_passed': True, 'argmax_checked_tokens': 3}])
    summary['model_digest'] = 'synthetic-model'
    cells = report.ordered_cells({(10, 1): summary})
    monkeypatch.setattr(report, 'ordered_cells', lambda observed: cells)
    output = tmp_path / 'readout-second'
    report.analyze(study, output, execution_root=study)
    text = (output / 'STUDY_REPORT.md').read_text(encoding='utf-8')
    assert 'completed=1/4; 全四格验收=pending' in text
    assert '| 10/1 | primary | valid | pending | 0/1 (0.00%) | 3/True |' in text

def test_tau_plot_requires_full_argmax_pass_but_not_ar_exact(tmp_path,monkeypatch):
    from studies.esp_table1_bf16_full_argmax.flow.write.run import write_plot, main
    from spec_mtp.structure.artifact.provenance import file_sha256
    cells=[{'BC':bc,'num_masks':k,'n':480,'argmax_all_passed':True,
        'pooled_tau':1.4,'prompt_mean_tau':1.5,'category_mean_pooled_tau':1.45,'accepted':100}
        for bc,k in [(10,1),(30,1),(60,1),(60,2)]]
    body={'all_cells_observed':False,'quality_passed':True,'AR_all_exact':False,'cells':cells}
    path=tmp_path/'tau.png'
    with pytest.raises(AssertionError):
        write_plot(body,path)
    assert not path.exists()
    body['all_cells_observed']=True
    body['quality_passed']=False
    with pytest.raises(AssertionError):
        write_plot(body,path)
    body['quality_passed']=True
    forbidden = ROOT / 'studies/esp_table1_bf16_full_argmax/_forbidden_plot_for_test.png'
    with pytest.raises(ValueError):
        write_plot(body, forbidden)
    assert not forbidden.exists()
    source, output = tmp_path / 'source', tmp_path / 'readout'
    source.mkdir()
    report = source / 'evidence.json'
    report.write_text(json.dumps(body), encoding='utf-8')
    assert main(['--report', str(report), '--report-sha256', file_sha256(report), '--output-root', str(output)]) == 0
    path = output / 'figures/tau_acceptance.png'
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and path.stat().st_size>10000
    binding = json.loads((output / 'readout_inputs.json').read_text(encoding='utf-8'))
    assert binding['output_sources']['figures/tau_acceptance.png'] == file_sha256(path)
