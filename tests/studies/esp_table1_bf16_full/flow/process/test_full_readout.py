"""Original full-AR saved JSON, strict producer identity and external publication; CPU IO only."""
import importlib
import json
from pathlib import Path

import pytest

from spec_mtp.structure.artifact.provenance import file_sha256
from studies._shared.structure.evidence import digest
from studies._shared.structure.sources import files_digest

reader = importlib.import_module('studies.esp_table1_bf16_full.flow.process.run')


@pytest.fixture
def saved_full(tmp_path):
    study, execution, output = tmp_path / 'study', tmp_path / 'execution', tmp_path / 'readout'
    (study / 'docs').mkdir(parents=True)
    source = execution / 'src/spec_mtp/original.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'# original saved producer fixture; never imported\n')
    recipe = study / 'recipe.py'; recipe.write_bytes(b'# frozen original Study recipe\n')
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    prompts = [{'question_id': 81 + i, 'category': category, 'input_ids': [1, 2]}
               for i, category in enumerate(('a', 'b'))]
    write(study / 'docs/prompts_manifest.json', {'rendered_prompts': prompts})
    write(study / 'docs/modelscope_snapshot.json', {'files': [{'path': 'model.bin', 'sha256': 'a' * 64}]})
    source_files = {'spec_mtp/original.py': file_sha256(source)}
    producer = {'package': 'spec_mtp', 'digest': files_digest(source_files), 'files': source_files,
                'study_files': {'recipe.py': file_sha256(recipe),
                    'docs/prompts_manifest.json': file_sha256(study / 'docs/prompts_manifest.json')}}
    entries = []
    for budget, masks, _, _ in reader.protocol.CELLS:
        spec = {'method': 'esp', 'block_complexity': budget, 'num_masks': masks,
                'tree': {'mode': 'dynamic', 'pruning': True}, 'mask': {'init': 'mean_prompt', 'lam': .1, 'update': True}}
        run_id = f'cell-{budget}-{masks}'
        entries.append({'factors': {'algorithm.config': {'spec': spec}}, 'runs': [{'id': run_id}]})
        folder = study / 'runs' / run_id
        assets = folder / 'assets'
        settings = {'run': {'seed': 123}, 'spec': spec, 'eval': {'temperature': 0, 'max_new_tokens': 256, 'warmup': False,
                    'run_ar_baseline': True, 'stop_on_eos': True}}
        inputs = {'dtype': 'torch.bfloat16', 'backend': 'eager', 'model_digest': 'saved-model',
                  'runtime': {'scope': 'controlled scalar JSON fixture, no model'}, 'settings': settings,
                  'files': {'model.bin': 'a' * 64}, 'prompts': prompts}
        rows = [{'category': item['category'], 'decode_calls': 2, 'committed': 3, 'accepted': 1,
                 'new_tokens': 4, 'spec_time': 0, 'ar_time': 0, 'exact_match': 1} for item in prompts]
        evidence = [{'input_ids': item['input_ids'], 'output_ids': [1, 2, 3, 4, 5, 6],
                     'ar_ids': [1, 2, 3, 4, 5, 6], 'stats': {'block_complexity': budget,
                        'committed': 3, 'decode_calls': 2, 'accepted': 1}} for item in prompts]
        write(assets / 'decode_inputs.json', inputs)
        write(assets / 'decode_observations.json', {'inputs_digest': digest(inputs), 'out': {'rows': rows, 'evidence': evidence}})
        config = {'seed': 123, 'algorithm': {'config': {'spec': spec}}}
        write(assets / 'spec_mtp_sources.json', dict(producer, config_digest=digest(config)))
        write(assets / 'numeric_policy.json', {'dtype': 'torch.bfloat16', 'bf16_reduced_precision_reduction': True,
                                             'deterministic': False})
        write(folder / 'config.yaml', config)
        write(folder / 'result.json', {'status': 'failed', 'spec_mtp': {'passed': False}})
    write(study / 'index.json', {'experiments': entries})
    return study, execution, output, source, write


@pytest.mark.parametrize('mismatch', [False, True])
def test_full_readout_preserves_ar_quality_and_complete_negative_results(saved_full, mismatch):
    study, execution, output, source, write = saved_full
    raw = study / 'runs/cell-10-1/assets/decode_observations.json'
    if mismatch:
        data = json.loads(raw.read_bytes())
        data['out']['evidence'][0]['ar_ids'][-1] = 9
        data['out']['rows'][0]['exact_match'] = 0
        write(raw, data)
    before = {path: path.read_bytes() for path in study.rglob('*') if path.is_file()}
    producer_before = source.read_bytes()
    result = reader.analyze(study, output, execution_root=execution)
    assert result['all_cells_observed'] and result['quality_passed'] is not mismatch
    assert [cell['role'] for cell in result['cells']] == ['auxiliary', 'primary', 'primary', 'control']
    assert result['cells'][0]['exact_count'] == (1 if mismatch else 2)
    assert all(cell['pooled_tau'] == 1.5 and cell['main_calls'] == 6 for cell in result['cells'])
    assert (output / 'protocol_evidence_v1.json').is_file() and (output / 'readout_inputs.json').is_file()
    report = (output / 'FULL_REPORT.md').read_text(encoding='utf-8')
    assert '../study/docs/prompts_manifest.json' in report and 'argmax' in report
    assert all(path.read_bytes() == value for path, value in before.items()) and source.read_bytes() == producer_before


@pytest.mark.parametrize('fault', ['source_changed', 'source_missing', 'source_unknown', 'missing_binding',
                                 'input_digest', 'exact_lie', 'missing_ar', 'model_hash', 'duplicate_cell',
                                 'config_digest', 'captured_seed', 'row_count'])
def test_full_readout_rejects_unbound_sources_or_inconsistent_saved_records_before_output(saved_full, fault):
    study, execution, output, source, write = saved_full
    assets = study / 'runs/cell-10-1/assets'
    if fault == 'source_changed':
        source.write_bytes(b'changed producer')
    elif fault == 'source_missing':
        source.unlink()
    elif fault in ('source_unknown', 'missing_binding'):
        path = assets / 'spec_mtp_sources.json'
        if fault == 'missing_binding':
            path.unlink()
        else:
            body = json.loads(path.read_bytes()); body['digest'] = 'unknown'; write(path, body)
    elif fault == 'duplicate_cell':
        path = study / 'index.json'; body = json.loads(path.read_bytes())
        body['experiments'][-1] = body['experiments'][0]; write(path, body)
    elif fault == 'model_hash':
        path = study / 'docs/modelscope_snapshot.json'; body = json.loads(path.read_bytes())
        body['files'][0]['sha256'] = 'b' * 64; write(path, body)
    elif fault in ('config_digest', 'captured_seed'):
        if fault == 'config_digest':
            path = assets.parent / 'config.yaml'; body = json.loads(path.read_bytes())
            body['data'] = {'changed': True}; write(path, body)
        else:
            path = assets / 'decode_inputs.json'; body = json.loads(path.read_bytes())
            body['settings']['run']['seed'] = 999; write(path, body)
            raw_path = assets / 'decode_observations.json'; raw = json.loads(raw_path.read_bytes())
            raw['inputs_digest'] = digest(body); write(raw_path, raw)
    else:
        path = assets / 'decode_observations.json'; body = json.loads(path.read_bytes())
        if fault == 'input_digest':
            body['inputs_digest'] = 'changed'
        elif fault == 'row_count':
            body['out']['rows'][0]['committed'] = 999
        elif fault == 'exact_lie':
            body['out']['evidence'][0]['ar_ids'][-1] = 9
        else:
            body['out']['evidence'][0]['ar_ids'] = None
        write(path, body)
    with pytest.raises((ValueError, FileNotFoundError)):
        reader.analyze(study, output, execution_root=execution)
    assert not output.exists()


@pytest.mark.parametrize('fault', ['input', 'current_tool'])
def test_full_readout_rechecks_original_inputs_and_current_tool_before_publication(saved_full, monkeypatch, fault):
    study, execution, output, source, _ = saved_full
    original = reader.protocol.render_report
    def render(*args, **kwargs):
        text = original(*args, **kwargs)
        if fault == 'input':
            source.write_bytes(b'changed during report render')
        else:
            monkeypatch.setattr(reader.SavedReadout, '_source_files', lambda self: {})
        return text
    monkeypatch.setattr(reader.protocol, 'render_report', render)
    with pytest.raises(ValueError, match='changed'):
        reader.analyze(study, output, execution_root=execution)
    assert not output.exists()


def test_full_readout_never_overwrites_existing_output(saved_full):
    study, execution, output, _, _ = saved_full
    output.mkdir(); keep = output / 'keep'; keep.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        reader.analyze(study, output, execution_root=execution)
    assert keep.read_bytes() == b'original' and list(output.iterdir()) == [keep]


@pytest.mark.parametrize('failure', [False, True])
def test_full_readout_publishes_ready_evidence_last_and_keeps_failed_publication_partial(saved_full, monkeypatch, failure):
    study, execution, output, _, _ = saved_full
    original = Path.rename
    published = []
    def rename(path, target):
        if Path(target).name == 'protocol_evidence_v1.json':
            assert (output / 'FULL_REPORT.md').is_file() and (output / 'readout_inputs.json').is_file()
            if failure:
                raise OSError('ready publication failed')
        published.append(Path(target).name)
        return original(path, target)
    monkeypatch.setattr(Path, 'rename', rename)
    if failure:
        with pytest.raises(OSError, match='ready publication failed'):
            reader.analyze(study, output, execution_root=execution)
        assert not (output / 'protocol_evidence_v1.json').exists()
        assert (output / 'protocol_evidence_v1.json.partial').is_file()
    else:
        reader.analyze(study, output, execution_root=execution)
        assert published[-1] == 'protocol_evidence_v1.json'


@pytest.mark.parametrize('missing', ['--study-dir', '--execution-root', '--output-root'])
def test_full_cli_requires_all_explicit_input_and_output_roots(monkeypatch, missing):
    monkeypatch.setattr(reader, 'analyze', lambda *a, **kw: pytest.fail('incomplete CLI dispatched'))
    args = {name: 'fixture' for name in ('--study-dir', '--execution-root', '--output-root')}
    with pytest.raises(SystemExit) as caught:
        reader.main([item for name, value in args.items() if name != missing for item in (name, value)])
    assert caught.value.code == 2
