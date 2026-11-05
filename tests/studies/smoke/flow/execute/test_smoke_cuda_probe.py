"""Explicit CPU preflight spies and one bounded complete CUDA engineering matrix."""
from types import SimpleNamespace
import json
import pytest
from spec_mtp.structure.artifact.provenance import file_sha256
from studies.smoke.flow.execute import run as entry


def arguments(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    plan=inputs/'plan.md';plan.write_text('Authorized tiny engineering fixture; no algorithm experiment.\n')
    return SimpleNamespace(plan=plan,plan_sha256=file_sha256(plan),output_root=tmp_path/'new-output',
        version='cuda-smoke-engineering-fixture-v1',device='cuda:0',compare=None,compare_sha256=None)


@pytest.mark.parametrize('fault',['plan','device','CPUcompare','compare_hash','existing'])
def test_probe_preflight_refuses_bad_plan_device_or_legacy_compare_before_loading(tmp_path,monkeypatch,fault):
    args=arguments(tmp_path)
    from spec_mtp.structure.model import hf_causal
    monkeypatch.setattr(hf_causal,'HFCausalLM',lambda *a,**kw:pytest.fail('invalid admission loaded a model'))
    if fault=='plan':args.plan_sha256='0'*64
    elif fault=='device':args.device='cpu'
    elif fault in ('CPUcompare','compare_hash'):
        args.compare=args.plan.parent/'old.json';args.compare.write_text(json.dumps({'device':'cpu','cells':{}}))
        args.compare_sha256=file_sha256(args.compare) if fault=='CPUcompare' else '0'*64
    else:args.output_root.mkdir()
    with pytest.raises((ValueError,FileExistsError)):entry.run(args)
    if fault!='existing':assert not args.output_root.exists()
    else:assert list(args.output_root.iterdir())==[]


def test_current_cuda_smoke_matrix_records_full_t0_and_sampling_scope(tmp_path):
    import torch
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    print('smoke matrix device=cuda:0 name='+torch.cuda.get_device_name(0),flush=True)
    args=arguments(tmp_path);report=entry.run(args)
    assert report['identity']['device']=='cuda:0' and report['identity']['dtype']=='torch.float32'
    assert len(report['cells'])==36 and report['summary']['T0_argmax_checked_tokens']==384
    assert report['summary']['sampling_distribution_validated'] is False
    complete=json.loads((args.output_root/'probe_complete.json').read_bytes())
    assert complete['probe_sha256']==file_sha256(args.output_root/'probe.json')
    assert complete['legacy_CPU_comparison'] is False
    assert all(len(row['trace'])==row['decode_calls'] for row in report['cells'].values())
    assert len((args.output_root/'cells.jsonl').read_text().splitlines())==36


@pytest.mark.parametrize('fault',['missing_complete','bad_complete','unavailable_after_valid_complete'])
def test_probe_comparison_requires_published_bundle_before_cuda_or_model(tmp_path,monkeypatch,fault):
    import torch
    args=arguments(tmp_path)
    baseline=args.plan.parent/'baseline';baseline.mkdir()
    probe=baseline/'probe.json';header=baseline/'header.json';cells=baseline/'cells.jsonl'
    matrix={f'{variant}|T={temperature}|p={i}':{'new_tokens':32,'exact_match':True,
        'argmax_checked_tokens':32 if temperature==0 else None}
        for variant in entry.protocol.VARIANTS for temperature in entry.protocol.TEMPERATURES for i in range(4)}
    summary=entry.protocol.summarize_cells(matrix)
    data={'schema':1,'protocol':entry.protocol.PROTOCOL,'version':'fixture','identity':{},'sources':{},'summary':summary,'cells':matrix}
    probe.write_text(json.dumps(data));header.write_text(json.dumps(data));cells.write_text('fixture\n')
    marker={'schema':1,'protocol':entry.protocol.PROTOCOL,'version':'fixture','summary':summary,
        'probe_sha256':file_sha256(probe),'header_sha256':file_sha256(header),'cells_sha256':file_sha256(cells)}
    if fault=='bad_complete':marker['probe_sha256']='0'*64
    if fault!='missing_complete':(baseline/'probe_complete.json').write_text(json.dumps(marker))
    args.compare=probe;args.compare_sha256=file_sha256(probe)
    def available():
        assert fault=='unavailable_after_valid_complete','invalid published binding reached CUDA admission'
        return False
    monkeypatch.setattr(torch.cuda,'is_available',available)
    with pytest.raises((ValueError,FileNotFoundError)):entry.run(args)
    assert not args.output_root.exists()
