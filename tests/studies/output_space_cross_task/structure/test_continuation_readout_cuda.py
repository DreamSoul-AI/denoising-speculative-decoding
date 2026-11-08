"""Complete six-unit/three-bank saved-data engineering audit; no model or draws."""

import copy
import hashlib
import json
import pytest
import torch

from studies._shared.structure.extension_diagnostics import choose_window
from studies._shared.structure.sampling_diagnostics import event_summary
from studies.output_space_cross_task.structure import sampling_readout as protocol


@pytest.fixture
def continuation(record_property):
    assert torch.cuda.is_available(), 'real CUDA required; no CPU fallback'
    device = torch.device('cuda:0')
    record_property('cuda_device', torch.cuda.get_device_name(device))
    print(f'continuation engineering audit device={device} name={torch.cuda.get_device_name(device)}', flush=True)
    trace = {'root_position': 10, 'cache_before': 10, 'cache_after': 11,
             'accepted_nodes': [], 'committed_tokens': [1], 'stopped': False}
    stats = {'trace': [trace], 'accepted': 0, 'new_tokens': 3, 'committed': 2,
             'decode_calls': 2, 'hit_context_limit': False,
             'attempt_by_depth': {1: 2}, 'accept_by_depth': {1: 0}}
    units, tensors = [], {}
    for index, (qid, seed) in enumerate(((242, 126), (244, 126), (245, 126), (245, 127), (246, 126), (246, 127))):
        native = [{'case': name, 'output_equal_independent_sampled_AR': True,
                   'output_ids': [1, 2, 3], 'stats': copy.deepcopy(stats)}
                  for name in ('output_order0', 'output_order1')]
        unit = {'qid': qid, 'seed': seed, 'temperature': .8, 'selection': choose_window([trace], [trace]),
                'selection_sha256': 'b' * 64, 'native': native, 'AR': {'output_ids': [1, 2, 3],
                'stats': {'ar_new_tokens': 3, 'ar_calls': 3}}, 'cases': [], 'observed': []}
        if index == 5:
            for row in native:
                row['stats']['trace'] = []
            unit['selection'] = choose_window([], [])
        else:
            for label, accepted in (('self', 0), ('graft', 1)):
                replay_trace = copy.deepcopy(trace)
                if accepted:
                    replay_trace.update(accepted_nodes=[1], cache_after=12, committed_tokens=[1, 2])
                unit['cases'].append({'label': label, 'replay': {
                    'output_equal_independent_AR': True, 'pre_intervention_trace_equal': True,
                    'only_selected_vector_replaced': True, 'replacement_equal_donor': True,
                    'modified_provider_calls': 1, 'trace': [replay_trace], 'observer_forward_calls': 1,
                    'stats': {**copy.deepcopy(stats), 'accepted': accepted}}})
            original = torch.zeros((2, 2, 3), device=device)
            donor = torch.ones_like(original)
            actual = original.clone()
            actual[0, 1] = donor[0, 1]
            prefix = f'q{qid}s{seed}'
            tensors.update({f'{prefix}:baseline:output_order1:10': {'current_masks': original},
                            f'{prefix}:baseline:output_order0:10': {'current_masks': donor},
                            f'{prefix}:graft:10': {'current_masks': actual}})
        units.append(unit)
    logits = torch.zeros((1, 1, 324), device=device)
    logits[0, 0, 11], logits[0, 0, 323] = 1., 2.
    groups = []
    for temperature, count in ((.8, 16384), (1., 16384), (2., 131072)):
        # A fixed stored bank, never a fresh sampler call or model experiment.
        draws = [0] * count
        if temperature == 2.:
            draws[1], draws[-1] = 11, 323
        seeds = list(range(10000, 10000 + count))
        probabilities = torch.softmax(logits[0, 0] / temperature, dim=-1)
        cells = []
        for label, token in (('S0P0', 11), ('S0P1', 323), ('S1P0', 11), ('S1P1', 323)):
            probability = float(probabilities[token])
            summary = event_summary(draws, seeds, token, probability)
            cells.append({'label': label, 'candidate': token, 'exact_candidate_probability': probability,
                          'first_slot_hits': summary['hits'], **summary})
        groups.append({'temperature': temperature, 'verifier_draws': count, 'seeds': seeds,
                       'verifier_token_ids': draws, 'RNG_consumed': True, 'same_verifier_draws_for_all_cells': True,
                       'role': 'changed_temperature_stress' if temperature == 2. else 'original_temperature_control',
                       'cells': cells})
    expansion = {'units': units, 'source_binding': {'plan': 'old-plan'}, 'raw_tensor_sha256': 'fixture',
                 'source_digest': protocol.SOURCE_DIGEST, 'quality_passed': True,
                 'model_digest': 'old-model', 'runtime': {'gpu': 'historical runtime stays unchanged'}}
    rare = {'groups': groups, 'source_binding': {'plan': 'old-plan'}, 'quality_passed': True,
            'source_digest': protocol.SOURCE_DIGEST}
    bindings = {'analysis': 'current', 'plan': 'old-plan', 'expansion_report': 'original-runtime',
                'rare_report': 'original-rare'}
    return expansion, rare, tensors, {'frame:719': {'logits': logits}}, dict.fromkeys(
        [(u['qid'], u['seed']) for u in units], 'b' * 64), bindings, device


def test_saved_continuation_audits_all_six_units_and_three_draw_banks_on_cuda(continuation):
    expansion, rare, data, context, selections, bindings, device = continuation
    before = {name: sample['current_masks'].clone() for name, sample in data.items()}
    result = protocol.analyze_continuation(*continuation)
    assert len(result['rows']) == 6 and len(result['full_semantic_paths']) == 5
    assert result['rows'][-1]['status'] == 'infeasible_not_a_null_effect'
    assert [row['verdict'] for row in result['H8']] == ['preserved', 'falsified']
    assert result['H9_changed_temperature_each_candidate_observed'] is True
    assert result['rare_native_verifier_draws'] == 163840 and result['teacher_forwards'] == 0
    assert [group['verifier_draws'] for group in result['rare_groups']] == [16384, 16384, 131072]
    assert result['rare_groups'][0]['cells'][0]['Clopper_Pearson_95pct'][0] == 0.
    assert result['rare_groups'][2]['cells'][0]['first_hit_seed'] == 10001
    assert result['rare_groups'][2]['cells'][1]['first_hit_seed'] == 141071
    assert result['source_binding'] == {'expansion': expansion['source_binding'], 'rare': rare['source_binding']}
    assert result['runtime'] == expansion['runtime'] and result['plan_sha256'] == bindings['plan']
    for original, group in zip(rare['groups'], result['rare_groups']):
        assert group['draw_bank_sha256'] == hashlib.sha256(json.dumps(original['verifier_token_ids']).encode()).hexdigest()
    assert all(torch.equal(data[name]['current_masks'], tensor) for name, tensor in before.items())
    # A missing observed rare candidate must falsify H9 without inventing zero-data certainty.
    group = rare['groups'][-1]
    group['verifier_token_ids'][-1] = 0
    for cell in group['cells']:
        summary = event_summary(group['verifier_token_ids'], group['seeds'], cell['candidate'], cell['exact_candidate_probability'])
        cell.update(summary, first_slot_hits=summary['hits'])
    assert protocol.analyze_continuation(*continuation)['H9_changed_temperature_each_candidate_observed'] is False


@pytest.mark.parametrize('damage', ['unit_order', 'selection', 'mask', 'scope', 'context',
                                   'bank_length', 'empty_cells', 'candidate', 'role', 'count', 'hits'])
def test_saved_continuation_rejects_invalid_data_context_and_schema_on_cuda(continuation, damage):
    expansion, rare, data, context, selections, bindings, device = continuation
    if damage == 'unit_order':
        expansion['units'].reverse()
    elif damage == 'selection':
        selections[242, 126] = 'different'
    elif damage == 'mask':
        data['q242s126:graft:10']['current_masks'][1, 0, 0] = 7
    elif damage == 'scope':
        expansion['units'][0]['cases'][1]['replay']['modified_provider_calls'] = 2
    elif damage == 'context':
        context['frame:719']['logits'][0, 0, 11] = 9
    elif damage == 'bank_length':
        rare['groups'][0]['verifier_token_ids'].pop()
    elif damage == 'empty_cells':
        rare['groups'][0]['cells'] = []
    elif damage == 'candidate':
        rare['groups'][0]['cells'][0]['candidate'] = 0
    elif damage == 'role':
        rare['groups'][-1]['role'] = 'original_temperature_control'
    elif damage == 'count':
        rare['groups'][-1]['verifier_draws'] -= 1
    else:
        rare['groups'][-1]['cells'][0]['first_slot_hits'] += 1
    with pytest.raises(AssertionError):
        protocol.analyze_continuation(*continuation)


@pytest.mark.parametrize('damage', [None, 'input_manifest', 'dataset', 'tensor', 'context_tensor'])
def test_continuation_cli_binds_saved_inputs_before_tensor_loading_on_cuda(continuation, tmp_path, monkeypatch, damage):
    from spec_mtp.structure.artifact import provenance
    from studies.output_space_cross_task.flow.process.run import analyze

    expansion, rare, tensors, context, selections, bindings, device = continuation
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    repository = inputs / 'repository'
    repository.mkdir()
    blobs = {}
    old_commit, report_commit = 'b' * 40, 'a' * 40

    def entry(name, value, *, member=None, commit=report_commit):
        path = inputs / name
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        member = member or 'saved/' + name
        blobs[commit, member] = data
        return {'path': str(path), 'sha256': sha, 'member': member, 'git_sha256': sha, 'commit': commit}

    files = {}
    for role, data in (('tensor', tensors), ('context_tensor', context)):
        path = inputs / (role + '.pt')
        torch.save(data, path)
        files[role] = {'path': str(path), 'sha256': provenance.file_sha256(path)}
    for role in ('input_manifest', 'dataset'):
        files[role] = entry(role, (role + ' original bytes').encode())
        expansion['source_binding'][role] = files[role]['sha256']
    for unit in expansion['units']:
        role = f"selection:{unit['qid']}:{unit['seed']}"
        files[role] = entry(role.replace(':', '-') + '.json', unit['selection'])
        unit['selection_sha256'] = files[role]['sha256']
    sources = {}
    for role, report, tool in (('expansion', expansion, 'diagnose_sampling_continuation.py'),
                               ('rare', rare, 'diagnose_rare_stress.py')):
        sources[role] = {
            'tool': entry(role + '-tool.py', b'# fixed original producer\n', member='tools/' + tool, commit=old_commit),
            'plan': entry(role + '-plan.md', b'# fixed original plan\n',
                          member='studies/output_space_cross_task/docs/SAMPLING_CONTINUATION_PLAN.md', commit=old_commit)}
        report['source_binding'].update({claim: item['sha256'] for claim, item in sources[role].items()})
    expansion['raw_tensor_sha256'] = files['tensor']['sha256']
    context_report = {'raw_tensor_sha256': files['context_tensor']['sha256']}
    reports = {role: entry(role + '.json', value)
               for role, value in (('expansion', expansion), ('rare', rare), ('context', context_report))}
    historical_calls = []

    def historical(repo, commit, member, sha):
        data = blobs[commit, member]
        assert hashlib.sha256(data).hexdigest() == sha
        historical_calls.append((commit, member))
        return data

    monkeypatch.setattr(provenance, 'read_historical_bytes', historical)
    if damage in ('input_manifest', 'dataset'):
        expansion['source_binding'][damage] = '0' * 64
        reports['expansion'] = entry('expansion.json', expansion)
    elif damage is not None:
        # Caller-supplied file identity remains valid, but original report binding differs.
        files[damage]['sha256'] = '0' * 64
    index = {'schema': 1, 'kind': 'continuation', 'repository': str(repository), 'commit': report_commit,
             'reports': reports, 'files': files, 'producer_sources': sources}
    path = inputs / 'index.json'
    path.write_text(json.dumps(index), encoding='utf-8')
    output = tmp_path / 'output'
    before = {str(path): provenance.file_sha256(path) for path in inputs.iterdir() if path.is_file()}
    if damage is not None:
        monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('invalid saved identity reached tensor loading'))
        with pytest.raises(ValueError):
            analyze('continuation', path, provenance.file_sha256(path), output, device=str(device))
        assert not output.exists()
    else:
        result = analyze('continuation', path, provenance.file_sha256(path), output, device=str(device))
        assert result['readout_model_forwards'] == result['readout_sampling_draws'] == 0
        assert result['readout_device'] == str(device)
        marker = json.loads((output / 'readout_inputs.json').read_text())
        assert marker['output_sources']['sampling_continuation_analysis_v1.json']
        assert any(commit == old_commit and member.startswith('tools/') for commit, member in historical_calls)
    assert before == {str(path): provenance.file_sha256(path) for path in inputs.iterdir() if path.is_file()}
