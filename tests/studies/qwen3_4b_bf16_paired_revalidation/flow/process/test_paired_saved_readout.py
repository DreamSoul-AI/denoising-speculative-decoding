"""cpu pure IO/stats: the independently repeated counter report keeps all tau/WTL comparisons."""
import copy
import hashlib
import importlib
import json
from itertools import combinations
from pathlib import Path

import pytest

from studies._shared.structure import geometry_4b as geometry
from studies.qwen3_4b_bf16_five_methods.structure import readout as counts

reader = importlib.import_module('studies.qwen3_4b_bf16_paired_revalidation.flow.process.run')
ROOT = Path(__file__).resolve().parents[5]


def fixture_reports(tmp_path, monkeypatch):
    from spec_mtp.structure.artifact import provenance
    rows = [dict(question_id=i, category='synthetic', accepted=1, committed=2, decode_calls=1,
        new_tokens=3, model_calls=2, argmax_checked_tokens=3, AR_exact=True,
        calls_delta_vs_AR=0, calls_vs_AR='tie') for i in range(480)]
    body = dict(complete=True, argmax_is_gate_AR_is_diagnostic=True, timing_disabled=True,
                cells={}, pairs=[], versus_original=[], analyzer_sha256='1'*64)
    for name in counts.CASES:
        cell = dict(native_status='succeeded', rows=copy.deepcopy(rows), totals=counts.summarize(rows, name == 'AR'),
                    run_id=name.replace('/', '_'), files_sha256={})
        if name != 'AR':
            cell['category_mean_pooled_tau'] = geometry.category_mean(rows)
        body['cells'][name] = cell
    prior = copy.deepcopy(body)
    for name in counts.CASES[1:]:
        body['versus_original'].append(dict(first='original/'+name, second=name,
            **geometry.compare(prior['cells'][name]['rows'], body['cells'][name]['rows'])))
    for bc in (30, 60):
        for a, b in combinations(counts.METHODS, 2):
            first, second = f'{bc}/{a}', f'{bc}/{b}'
            body['pairs'].append(dict(first=first, second=second,
                **geometry.compare(body['cells'][first]['rows'], body['cells'][second]['rows'])))
    folder = tmp_path/'inputs'; folder.mkdir()
    input_path, prior_path = folder/'repeat.json', folder/'prior.json'
    input_path.write_text(json.dumps(body), encoding='utf-8')
    prior_path.write_text(json.dumps(prior), encoding='utf-8')
    blobs = {'studies/qwen3_4b_bf16_paired_revalidation/docs/paired_evidence_v1.json': input_path.read_bytes(),
             'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json': prior_path.read_bytes()}
    def fixed(repository, commit, member, expected):
        assert repository == ROOT and commit == 'a'*40
        assert hashlib.sha256(blobs[member]).hexdigest() == expected
        return blobs[member]
    monkeypatch.setattr(provenance, 'read_historical_bytes', fixed)
    return dict(input_path=input_path, prior=prior_path, repository=ROOT, commit='a'*40,
        input_git_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        prior_git_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(), output_root=tmp_path/'readout')


def test_paired_saved_counts_preserve_tau_and_original_identity(tmp_path, monkeypatch):
    args = fixture_reports(tmp_path, monkeypatch)
    body = reader.published(**args)
    assert len(body['pairs']) == 20 and all(p['tau'] == {'win': 0, 'tie': 480, 'loss': 0} for p in body['pairs'])
    assert body['analyzer_sha256'] == '1'*64
    scope = json.loads((args['output_root']/'counter_scope.json').read_bytes())
    assert scope['native_run_revalidated'] is False and scope['recorded_files_sha256_checked'] == 0
    with pytest.raises(FileExistsError):
        reader.published(**args)


@pytest.mark.parametrize('damage', ['runtime', 'tau', 'optional_missing_original'])
def test_paired_saved_counter_failures_do_not_publish(tmp_path, monkeypatch, damage):
    args = fixture_reports(tmp_path, monkeypatch)
    if damage == 'optional_missing_original':
        root = tmp_path/'missing-original'; root.mkdir()
        args['execution_root'] = root
    else:
        data = json.loads(args['input_path'].read_bytes())
        if damage == 'tau':
            data['pairs'][0]['tau']['tie'] = 479
        else:
            data['analyzer_sha256'] = 'changed'
        args['input_path'].write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        reader.published(**args)
    assert not args['output_root'].exists()
