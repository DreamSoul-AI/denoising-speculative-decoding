
from spec_mtp.masks import build_mask_provider
import importlib.util
import json
from pathlib import Path
import pytest
import torch
import spec_mtp.decode as decoding
from studies._shared.structure.config import _convert
from spec_mtp.model import HFCausalLM

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('esp_argmax_audit', ROOT/'studies/esp_table1_bf16_full_argmax/argmax_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def decoder():
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size':32, 'hidden_size':16, 'intermediate_size':32,
        'num_layers':1, 'num_heads':2, 'num_kv_heads':1, 'max_position_embeddings':64}).to(torch.bfloat16)
    config = _convert({'method':'esp', 'num_masks':1, 'block_complexity':10, 'impl':'naive',
        'tree':{'mode':'dynamic','pruning':True}, 'mask':{'init':'mean_prompt','lam':.1,'update':True}})
    return decoding.SpecDecoder(model, config, {'temperature':0, 'max_new_tokens':7, 'stop_on_eos':False}, mask_policy=build_mask_provider(config, model.embedding_table))


@pytest.mark.parametrize('prefill_only', [False, True])
def test_observer_preserves_bf16_outputs_calls_and_restores(tmp_path, prefill_only):
    dec = decoder()
    if prefill_only:
        dec.max_new_tokens = 1
    prompt = torch.tensor([[1,2,3]])
    expected, base = dec.generate(prompt)
    generate, select = decoding.SpecDecoder.generate, decoding.select_token
    with audit.observe_argmax(tmp_path):
        actual, stats = dec.generate(prompt)
        # Independent AR remains outside the observer.
        decoding.ar_generate(dec.model, prompt, {'temperature':0,'max_new_tokens':1})
    assert torch.equal(actual, expected)
    assert stats['decode_calls'] == base['decode_calls']
    assert stats['trace'] == base['trace']
    assert decoding.SpecDecoder.generate is generate and decoding.select_token is select
    assert not dec.model._forward_hooks
    recorded = json.loads((tmp_path/'argmax_audit.jsonl').read_text())
    assert recorded == stats['verification_argmax_audit']
    assert recorded['passed'] and recorded['checked_committed_tokens'] == stats['new_tokens']
    with pytest.raises(AssertionError, match='old audit'):
        with audit.observe_argmax(tmp_path):
            pass


def test_wrong_selection_is_rejected_and_restored(tmp_path, monkeypatch):
    dec = decoder()
    wrong = lambda row, temperature, generator: (int(row.argmax().item()) + 1) % row.numel()
    monkeypatch.setattr(decoding, 'select_token', wrong)
    generate = decoding.SpecDecoder.generate
    with pytest.raises(AssertionError, match='forward argmax'):
        with audit.observe_argmax(tmp_path):
            dec.generate(torch.tensor([[1,2,3]]))
    assert decoding.select_token is wrong and decoding.SpecDecoder.generate is generate
    assert not dec.model._forward_hooks and not (tmp_path/'argmax_audit.jsonl').read_text()


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
    monkeypatch.syspath_prepend(str(ROOT/'studies/esp_table1_bf16_full_argmax'))
    spec = importlib.util.spec_from_file_location('argmax_report',ROOT/'studies/esp_table1_bf16_full_argmax/analyze_results.py')
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
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
    assert 'argmax_audit.py' in after['provenance']['include']


def test_historical_full_argmax_original_inputs():
    previous = ROOT/'studies/esp_table1_bf16_full'
    current = ROOT/'studies/esp_table1_bf16_full_argmax'
    assert (previous/'docs/prompts_manifest.json').read_bytes()==(current/'docs/prompts_manifest.json').read_bytes()
    assert len(json.loads((current/'docs/prompts_manifest.json').read_text())['rendered_prompts'])==480


def test_pending_report_does_not_link_unpublished_local_evidence(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'studies/esp_table1_bf16_full_argmax'))
    spec=importlib.util.spec_from_file_location('pending_argmax_report',ROOT/'studies/esp_table1_bf16_full_argmax/analyze_results.py')
    report=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/prompts_manifest.json').write_text(json.dumps({'rendered_prompts':[{'question_id':1}]}))
    (tmp_path/'docs/modelscope_snapshot.json').write_text('{}')
    (tmp_path/'index.json').write_text(json.dumps({'experiments':[]}))
    monkeypatch.setattr(report,'__file__',str(tmp_path/'analyze_results.py'))
    report.main()
    text=(tmp_path/'docs/STUDY_REPORT.md').read_text(encoding='utf-8')
    assert '`protocol_pending.json`' in text and '](protocol_pending.json)' not in text
    assert not json.loads((tmp_path/'docs/protocol_pending.json').read_text())['quality_passed']
    assert '全四格验收=pending' in text and 'token quality passed=False' not in text
    summary=report.summarize([{'category':'a','committed':2,'decode_calls':1,'accepted':1,
        'exact':False,'argmax_passed':True,'argmax_checked_tokens':3}])
    summary['model_digest']='synthetic-model'
    cells=report.ordered_cells({(10,1):summary})
    monkeypatch.setattr(report,'ordered_cells',lambda observed:cells)
    report.main()
    text=(tmp_path/'docs/STUDY_REPORT.md').read_text(encoding='utf-8')
    assert 'completed=1/4; 全四格验收=pending' in text
    assert '| 10/1 | primary | valid | pending | 0/1 (0.00%) | 3/True |' in text


def test_tau_plot_requires_full_argmax_pass_but_not_ar_exact(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'studies/esp_table1_bf16_full_argmax'))
    from plot_results import write_plot
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
    write_plot(body,path)
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') and path.stat().st_size>10000
