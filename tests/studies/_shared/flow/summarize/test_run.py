"""Saved Run finalisation, diagnostic tolerance and collector-to-gate contracts."""
import csv
import json
from types import SimpleNamespace

import pytest

from spec_mtp.structure.artifact.observation import SpecMtpGateError
from studies._shared.flow.summarize.run import run_gate
from studies._shared.structure import validation
from studies._shared.structure.evidence import digest


def test_run_gate_raises_and_records(tmp_path, saved_decode_config, saved_decode_assets, corrupt_saved_argmax):
    assets = saved_decode_assets(tmp_path)
    corrupt_saved_argmax(assets, 'passed', False)
    ctx = SimpleNamespace(config=saved_decode_config(), layout=SimpleNamespace(assets_dir=assets),
                          state={'result_draft': {'metrics': {}}})
    with pytest.raises(SpecMtpGateError):
        run_gate(ctx)
    assert ctx.state['result_draft']['spec_mtp']['passed'] is False


def test_run_gate_without_draft_still_gates(tmp_path, saved_decode_config, saved_decode_assets, corrupt_saved_argmax):
    ctx = SimpleNamespace(config=saved_decode_config(), layout=SimpleNamespace(assets_dir=saved_decode_assets(tmp_path)), state={})
    assert run_gate(ctx)['passed']
    assets = saved_decode_assets(tmp_path, name='b')
    corrupt_saved_argmax(assets, 'passed', False)
    bad = SimpleNamespace(config=saved_decode_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(SpecMtpGateError, match='argmax'):
        run_gate(bad)


@pytest.mark.parametrize('name', ['decode_inputs.json', 'decode_observations.json', 'spec_mtp_sources.json'])
def test_final_gate_requires_raw_evidence(tmp_path, name, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    (assets / name).unlink()
    ctx = SimpleNamespace(config=saved_decode_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(SpecMtpGateError):
        run_gate(ctx)


def test_final_gate_reports_ar_mismatch_without_blocking_argmax(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    path = assets / 'decode_observations.json'
    body = json.loads(path.read_text())
    body['out']['evidence'][0]['ar_ids'][-1] = 99
    path.write_text(json.dumps(body))
    ctx = SimpleNamespace(config=saved_decode_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    report=run_gate(ctx)
    assert report['passed'] and report['exact_match_rate']==0
    assert report['reported_exact_match_rate']==1 and report['AR_reported_exact_mismatches']==[0]


def test_final_gate_rejects_source_drift(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    path = assets / 'spec_mtp_sources.json'
    body = json.loads(path.read_text())
    body['files']['spec_mtp/added.py'] = 'bad'
    path.write_text(json.dumps(body))
    ctx = SimpleNamespace(config=saved_decode_config(), layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(SpecMtpGateError, match='sources changed'):
        run_gate(ctx)


def test_final_gate_rejects_config_drift(tmp_path, saved_decode_config, saved_decode_assets):
    assets = saved_decode_assets(tmp_path)
    config = saved_decode_config()
    config['model'] = {'name': 'changed'}
    ctx = SimpleNamespace(config=config, layout=SimpleNamespace(assets_dir=assets), state={})
    with pytest.raises(SpecMtpGateError, match='configuration changed'):
        run_gate(ctx)


def test_final_gate_rejects_live_input_file_drift(tmp_path, saved_decode_config, saved_decode_assets):
    from studies._shared.structure.evidence import file_hash
    assets = saved_decode_assets(tmp_path)
    source = tmp_path / 'prompt.txt'
    source.write_text('before')
    path = assets / 'decode_inputs.json'
    inputs = json.loads(path.read_text())
    inputs['files'] = {str(source): file_hash(source)}
    path.write_text(json.dumps(inputs))
    raw_path = assets / 'decode_observations.json'
    raw = json.loads(raw_path.read_text())
    raw['inputs_digest'] = digest(inputs)
    raw_path.write_text(json.dumps(raw))
    source.write_text('after')
    with pytest.raises(SpecMtpGateError, match='input changed'):
        run_gate(SimpleNamespace(config=saved_decode_config(), layout=SimpleNamespace(assets_dir=assets), state={}))


def test_ar_mismatch_is_diagnostic_and_native_gate_passes(tmp_path, saved_decode_config, saved_decode_assets):
    from spec_mtp.structure.algorithm.counts import _aggregate
    assets = saved_decode_assets(tmp_path,rate='0.0')
    raw = json.loads((assets/'decode_observations.json').read_text())
    flat = [{'category':k,**v} for k,v in _aggregate(raw['out']['rows']).items()]
    with (assets/'summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    ctx=SimpleNamespace(config=saved_decode_config(),layout=SimpleNamespace(assets_dir=assets),state={'result_draft':{}})
    report=run_gate(ctx)
    assert report['passed'] and report['argmax_checked_tokens']==6 and report['exact_match_rate']==0
    assert report['AR_diagnostics']==[{'prompt_index':0,'exact':False,'first_new_difference_index':5}]
    assert ctx.state['result_draft']['spec_mtp']['validation_policy']=='verifier-argmax-v1'
    assert validation.evaluate_run(saved_decode_config(run_ar=False),assets)['passed']


def test_cached_ar_summary_does_not_block_argmax_gate(tmp_path, saved_decode_config, saved_decode_assets):
    assets=saved_decode_assets(tmp_path)
    path=assets/'summary.csv'
    with path.open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        row['exact_match_rate']='0.2'
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ctx=SimpleNamespace(config=saved_decode_config(),layout=SimpleNamespace(assets_dir=assets),state={'result_draft':{'metrics':{}}})
    report=run_gate(ctx)
    assert report['passed'] and report['exact_match_rate']==1 and report['reported_exact_match_rate']==0.2
    assert ctx.state['result_draft']['metrics']['test_exactmatchrate']==1


@pytest.mark.parametrize('damage', ['raw_accept', 'raw_reject', 'raw_boolean', 'csv_accept', 'csv_reject',
                                    'zero_accept', 'zero_reject'])
def test_alignment_gate_rejects_inconsistent_persisted_counts(tmp_path, damage, saved_decode_config, saved_decode_assets, monkeypatch, record_property):
    import torch
    assert torch.cuda.is_available(), 'alignment saved-count verification requires CUDA'
    print('alignment verification device=cuda:0 name=' + torch.cuda.get_device_name(0), flush=True)
    record_property('cuda_device', torch.cuda.get_device_name(0))
    before = torch.get_default_device()
    original_item = torch.Tensor.item
    seen = {'collect': [], 'verify': []}
    phase = 'collect'

    def observed_item(tensor, *args, **kwargs):
        seen[phase].append((str(tensor.device), str(tensor.dtype)))
        return original_item(tensor, *args, **kwargs)

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(torch.Tensor, 'item', observed_item)
            with torch.device('cuda:0'):
                from studies._shared.flow.collect.run import run as collect
                assets = saved_decode_assets(tmp_path)
                cfg = saved_decode_config('alignment_probe')
                sources_path = assets/'spec_mtp_sources.json'
                saved = json.loads(sources_path.read_text())
                saved['config_digest'] = digest(cfg)
                sources_path.write_text(json.dumps(saved))
                raw_path = assets/'decode_observations.json'
                raw = json.loads(raw_path.read_text())
                raw.update(task='alignment_probe', out={'n_accepted':2, 'n_rejected':1,
                    'counts':{'accept':2, 'reject':1}, 'sums':{'accept':[1., .5], 'reject':[-.25, .25]}})
                if damage in ('zero_accept', 'zero_reject'):
                    key, field = ('accept', 'n_accepted') if damage == 'zero_accept' else ('reject', 'n_rejected')
                    raw['out']['counts'][key] = raw['out'][field] = 0
                    raw['out']['sums'][key] = [0., 0.]
                raw_path.write_text(json.dumps(raw))
                tracker = SimpleNamespace(append=lambda *a, **k: None, save=lambda *a: None, flush_state=lambda: None)
                ctx = SimpleNamespace(config=cfg, layout=SimpleNamespace(assets_dir=assets),
                    state={'tracker':tracker, 'collected':{'metrics':{}}, 'result_draft':{}})
                phase = 'collect'
                collect(ctx)
                phase = 'verify'
                assert run_gate(ctx)['passed']
                if damage in ('zero_accept', 'zero_reject'):
                    metric = 'test_accepted' if damage == 'zero_accept' else 'test_rejected'
                    assert ctx.state['collected']['metrics'][metric] == 0
                    return
                csv_path = assets/'alignment_metrics.csv'
                if damage.startswith('raw'):
                    field = 'n_accepted' if damage == 'raw_accept' else 'n_rejected'
                    raw['out'][field] = True if damage == 'raw_boolean' else raw['out'][field] + 1
                    if damage == 'raw_boolean':
                        raw['out']['counts']['reject'] = True
                    raw_path.write_text(json.dumps(raw))
                else:
                    with csv_path.open(newline='') as handle:
                        rows = list(csv.DictReader(handle))
                    field = 'n_accepted' if damage == 'csv_accept' else 'n_rejected'
                    rows[0][field] = '999'
                    with csv_path.open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                previous = raw_path.read_bytes(), csv_path.read_bytes()
                with pytest.raises(SpecMtpGateError, match='alignment.*count'):
                    run_gate(ctx)
                assert ctx.state['result_draft']['spec_mtp']['passed'] is False
                assert (raw_path.read_bytes(), csv_path.read_bytes()) == previous
    finally:
        assert torch.Tensor.item is original_item
        assert torch.get_default_device() == before
        assert all(seen.values()), 'both real collect and verify scalar results must be observed'
        assert all(device == 'cuda:0' and dtype == 'torch.float32'
                   for values in seen.values() for device, dtype in values)
        record_property('alignment_scalar_devices', seen)
