"""Bounded CUDA correctness on a local tiny model; no weights or data downloads."""

import gc
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

from spec_mtp.structure.algorithm.decode.loop import ar_generate, generate
from spec_mtp.structure.algorithm.mask import ESPMaskProvider
from spec_mtp.structure.api.decode_api import DecodeRequest, Prompt
from spec_mtp.structure.artifact.observation import audit_run
from spec_mtp.structure.model.hf_causal import HFCausalLM


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_tiny_cuda_decode_preserves_committed_argmax_and_releases_memory(backend, record_property):
    assert torch.cuda.is_available(), 'CUDA validation requires a real CUDA device'
    device = torch.device('cuda:0')
    # Initialize the process's CUDA BLAS workspace before measuring owned tensors.
    warmup = torch.nn.functional.linear(torch.ones((1, 3, 16), device=device),
                                        torch.ones((16, 16), device=device))
    del warmup
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(7)
    model = HFCausalLM(tiny_config={'vocab_size': 16, 'hidden_size': 16, 'intermediate_size': 32,
        'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1, 'max_position_embeddings': 32,
        'tie_word_embeddings': True}, attn_implementation=backend).to(device)
    assert model.device == device and model.embedding_table.is_cuda
    request = DecodeRequest(Prompt('cuda-tiny', (1, 2, 3)),
        {'num_masks': 1, 'method': 'esp', 'impl': 'efficient',
         'tree': {'mode': 'static', 'branches': [1], 'pruning': False}},
        max_new_tokens=4, sources={'science': 'cuda-engineering-validation'})
    rows = []

    class Sink:
        def write(self, row):
            rows.append(dict(row))

    mask = ESPMaskProvider(1, model.embedding_table)
    native = prompt = None
    try:
        result = generate(request, model, mask, Sink())
        prompt = torch.tensor([request.prompt.input_ids], device=device)
        native, _ = ar_generate(model, prompt, {'max_new_tokens': 4, 'temperature': 0})
        torch.cuda.synchronize(device)
        assert native.is_cuda and result.committed == tuple(native[0, 3:].tolist())
        audit = audit_run(rows, request.sources)
        assert result.evidence_complete
        assert audit.source_integrity and audit.argmax_coverage and audit.argmax_correctness, audit.reasons
        peak = torch.cuda.max_memory_allocated(device) - baseline
        assert 0 < peak < 256 * 1024 * 1024
        record_property('cuda_device', torch.cuda.get_device_name(device))
        record_property('cuda_peak_allocated_bytes', peak)
        record_property('committed_tokens', len(result.committed))
    finally:
        model.close()
        del native, prompt, mask, model
        gc.collect()
        torch.cuda.synchronize(device)
        remaining = torch.cuda.memory_allocated(device) - baseline
        record_property('cuda_remaining_allocated_bytes', remaining)
        assert remaining < 1024 * 1024


def test_cuda_study_pipeline_keeps_source_read_only_from_an_arbitrary_cwd(tmp_path, monkeypatch, record_property):
    import yaml
    from spec_mtp.structure.make.schedule import repo_root_from
    from studies._shared.flow.cli import main

    assert torch.cuda.is_available(), 'CUDA validation requires a real CUDA device'
    repository = Path(__file__).resolve().parents[4]
    template = repository / 'studies/smoke'
    source = tmp_path / 'source-study'
    for member in template.rglob('*'):
        if member.is_file() and member.suffix in ('.py', '.yaml'):
            copied = source / member.relative_to(template)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(member, copied)
    declaration = source / 'study.yaml'
    body = yaml.safe_load(declaration.read_text(encoding='utf-8'))
    body['study'] = 'round4_cuda_engineering'
    body['fixed']['data']['config'].update(vocab_size=16, prompt_len=3, num_prompts=1)
    body['fixed']['model']['config']['tiny'].update(vocab_size=16, hidden_size=16,
        intermediate_size=32, num_layers=1, num_heads=2, num_kv_heads=1, max_position_embeddings=32)
    algorithm = body['fixed']['algorithm']['config']
    algorithm['version'] = 'round4-cuda-engineering'
    algorithm['eval'].update(device='cuda', max_new_tokens=4, num_eval_prompts=1, warmup=False)
    body['fixed']['system']['config']['device'] = 'cuda'
    body['axes'] = {'algorithm.config.variant': ['esp_static']}
    declaration.write_text(yaml.safe_dump(body, sort_keys=False), encoding='utf-8')

    def snapshot(root):
        return ({path.relative_to(root).as_posix(): path.read_bytes()
                 for path in root.rglob('*') if path.is_file()},
                {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_dir()})

    before = snapshot(source)
    runtime = tmp_path / 'runtime'
    report = tmp_path / 'export/report.md'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    assert repo_root_from(source) == repository
    assert main(['run', str(source), '--run-root', str(runtime)]) == 0
    torch.cuda.synchronize()
    for name in ('index.json', 'provenance.json', 'spec_mtp_sources.json', 'study_source.json', 'process.json'):
        assert (runtime / name).is_file()
        assert not (source / name).exists()
    index = json.loads((runtime / 'index.json').read_text(encoding='utf-8'))
    experiments = index['experiments']
    assert len(experiments) == len(experiments[0]['runs']) == 1
    run_id = experiments[0]['runs'][0]['id']
    run = runtime / 'runs' / run_id
    assert (run / 'config.yaml').is_file() and (run / 'result.json').is_file()
    config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    assert config['system']['config']['device'] == 'cuda'
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    assert result['status'] == 'succeeded' and result['spec_mtp']['passed'] is True
    assert str(result['structure']['system']['device']).startswith('cuda')
    inputs = json.loads((run / 'assets/decode_inputs.json').read_text(encoding='utf-8'))
    assert inputs['device'].startswith('cuda')
    assert inputs['runtime']['gpu'] == torch.cuda.get_device_name(0)
    assert inputs['runtime']['cuda'] == torch.version.cuda
    assert len(inputs['prompts']) == 1 and len(inputs['prompts'][0]['input_ids']) == 3
    gate = json.loads((runtime / 'spec_mtp_study_check.json').read_text(encoding='utf-8'))
    process = json.loads((runtime / 'process.json').read_text(encoding='utf-8'))
    assert gate['complete'] is gate['passed'] is process['complete'] is True
    logs = (run / 'assets/logs/events.jsonl').read_text(encoding='utf-8')
    assert 'phases=prepare,execute,collect,summarize,write,process' in logs
    saved_process = (runtime / 'process.json').read_bytes()
    assert main(['status', str(runtime)]) == 0
    assert main(['logs', str(runtime)]) == 0
    assert main(['report', str(runtime), '--out', str(report)]) == 0
    assert report.is_file() and run_id in report.read_text(encoding='utf-8')
    assert (runtime / 'process.json').read_bytes() == saved_process
    assert snapshot(source) == before
    record_property('cuda_device', torch.cuda.get_device_name(0))
    record_property('runtime_run_id', run_id)
