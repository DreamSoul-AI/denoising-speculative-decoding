"""Read fixed state/path captures on explicit CUDA without models, draws or implicit temporary roots."""
import argparse
import json

from studies.output_space_cross_task.structure import state_readout as protocol

ROOT = 'studies/output_space_cross_task/docs/'
PRODUCERS = {
    'prior': ('diagnose_q241_canonical.py', 'CANONICAL_PLAN.md'),
    'local': ('diagnose_q241_local_factors.py', 'LOCAL_FACTOR_PLAN.md'),
    'extension': ('diagnose_state_extension.py', 'STATE_EXTENSION_PLAN.md'),
    'state': ('diagnose_q241_state_factors.py', 'STATE_FACTOR_PLAN.md'),
    'operator': ('diagnose_q178_rounding_factors.py', 'ROUNDING_FACTOR_PLAN.md'),
    'sampling': ('diagnose_sampling_paths.py', 'SAMPLING_PLAN.md'),
    'owner': ('diagnose_q241_owner1.py', 'OWNER_PLAN.md'),
    'data': ('diagnose_sampling_continuation.py', 'SAMPLING_CONTINUATION_PLAN.md'),
}
MEMBERS = {role: ROOT + name for role, name in {
    'prior': 'canonical_diagnosis_v1.json', 'local': 'local_factor_diagnosis_v1.json',
    'extension': 'state_extension_diagnosis_v1.json', 'state': 'state_factor_diagnosis_v1.json',
    'sampling': 'sampling_diagnosis_v1.json', 'owner': 'owner1_diagnosis_v1.json',
    'native': 'execution_v1.json', 'data': 'sampling_continuation_analysis_v1.json'}.items()}
MEMBERS['operator'] = 'studies/output_space_extension/docs/rounding_factor_diagnosis_v1.json'
ROLES = {'local-extension': {'local', 'extension', 'prior', 'native'},
         'state-factors': {'prior', 'state', 'operator', 'native'},
         'sampling-owner': {'sampling', 'owner', 'local', 'prior'},
         'sampled-depth': {'data', 'native'}}


def producer_members(kind):
    producers: dict[str, dict[str, str | tuple[str, ...]]] = {}
    for role in ROLES[kind] - {'native'}:
        tool, plan = PRODUCERS[role]
        producers[role] = {'tool': 'studies/_shared/structure/' + tool,
                          'plan': ('studies/output_space_extension/docs/' if role == 'operator' else ROOT) + plan}
    if 'data' in producers:
        producers['data']['tool'] = ('studies/_shared/structure/diagnose_sampling_continuation.py',
                                     'tools/diagnose_sampling_continuation.py')
    return producers


def analyze(kind, inputs, inputs_sha256, output_root, *, device=None):
    import torch
    from studies.output_space_cross_task.flow.process.run import bind_inputs
    if kind not in ROLES:
        raise ValueError('unknown fixed state readout')
    producers = producer_members(kind)
    session, index, reports, report_hashes, hashes = bind_inputs(kind, inputs, inputs_sha256, output_root,
        roles=ROLES[kind], producers=producers,
        quality_fields={role: 'study_quality_passed' if role == 'native' else 'quality_passed' for role in ROLES[kind]},
        source_roles=ROLES[kind])
    for role, entry in index['reports'].items():
        if entry.get('member') != MEMBERS[role]:
            raise ValueError('report Git member does not match the fixed state input role')
        field = entry.get('runtime_sha256_field')
        if field != ('expansion_report_sha256' if role == 'data' else None):
            raise ValueError('state report runtime identity is not its declared original binding')
    actual_device = torch.device(device) if device is not None else None
    if (actual_device is None or actual_device.type != 'cuda' or actual_device.index is None
            or not torch.cuda.is_available() or actual_device.index >= torch.cuda.device_count()):
        raise ValueError('state tensor audit requires an explicit available CUDA device; no CPU fallback')
    tensor_roles = ROLES[kind] - {'native', 'operator'}
    if 'operator' in reports:
        if hashes.get('operator_tensor') != reports['operator']['raw_tensor_sha256']:
            raise ValueError('original operator tensor identity differs')
    for role in tensor_roles:
        if hashes.get(role + '_tensor') != reports[role]['raw_tensor_sha256']:
            raise ValueError(f'original {role} tensor identity differs')
    for role, report in reports.items():
        original = report.get('source_binding', {})
        if 'native' in reports:
            matrix_sha = original.get('native', original.get('matrix'))
            if matrix_sha is not None and matrix_sha != report_hashes['native']:
                raise ValueError('native execution is not the original producer runtime bytes')
        for target, claim in (('prior', 'canonical'), ('local', 'local')):
            if target in reports and target != role:
                if claim + '_report' in original and original[claim + '_report'] != report_hashes[target]:
                    raise ValueError('original context report binding differs')
                if claim + '_tensor' in original and original[claim + '_tensor'] != hashes[target + '_tensor']:
                    raise ValueError('original context tensor binding differs')
    eos = []
    if kind != 'sampling-owner':
        if 'decode_inputs' not in hashes:
            raise ValueError('explicit original decode inputs are required for EOS semantics')
        expected = {value for report in reports.values() for key, value in report.get('source_binding', {}).items()
                    if key.endswith('assets/decode_inputs.json')}
        if hashes['decode_inputs'] not in expected:
            raise ValueError('decode input bytes lack an original producer binding')
        eos = session.read_json(index['files']['decode_inputs']['path'])['eos_ids']
    selections = {}
    if kind == 'sampling-owner':
        for unit in reports['sampling']['units']:
            role = f"selection:{unit['seed']}"
            if hashes.get(role) != unit['selection_sha256']:
                raise ValueError('sampling selection is missing or unbound')
            selections[unit['seed']] = hashes[role]
    if kind == 'sampled-depth':
        plan = index['files'].get('analysis_plan', {})
        if plan.get('member') != ROOT + 'SAMPLED_DEPTH_PLAN.md' or 'git_sha256' not in plan:
            raise ValueError('sampled-depth requires the fixed original analysis plan')
    session.verify()
    print(f'saved state audit device={actual_device} name={torch.cuda.get_device_name(actual_device)}; no model/draws', flush=True)
    load_roles = tensor_roles - {'sampling'}
    tensors = {role: torch.load(session._input(index['files'][role + '_tensor']['path']),
                               weights_only=True, map_location=actual_device, mmap=True) for role in load_roles}
    binding = {'analysis': session.sha(protocol.__file__),
               **{role + '_report': sha for role, sha in report_hashes.items()},
               **{role: sha for role, sha in hashes.items()}}
    if kind == 'local-extension':
        result = protocol.analyze_local_extension(reports['local'], reports['extension'], tensors['local'],
            tensors['extension'], tensors['prior'], reports['native'], eos, binding)
    elif kind == 'state-factors':
        result = protocol.analyze_state_factors(reports['prior'], reports['state'], reports['operator'],
            tensors['prior'], tensors['state'], reports['native'], eos, binding)
    elif kind == 'sampling-owner':
        result = protocol.analyze_sampling_owner(reports['sampling'], reports['owner'], tensors['local'],
            tensors['prior'], tensors['owner'], selections, binding)
    else:
        result = protocol.analyze_sampled_depth(reports['data'], tensors['data'], eos, binding)
    result.update(readout_only=True, readout_model_forwards=0, readout_sampling_draws=0,
                  readout_device=str(actual_device), readout_device_name=torch.cuda.get_device_name(actual_device),
                  historical_CPU_margin_revalidated=False,
                  quality_scope='Current state/path/discrete gates; historical CPU margin numbers not revalidated',
                  numerical_scope='Current CUDA FP32/FP64 audit; old CPU references retain their original provenance, not bit-equivalence')
    session.write_text(session.output / (kind.replace('-', '_') + '_analysis_v1.json'),
                       json.dumps(result, indent=2, allow_nan=False) + '\n')
    session.finish()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=tuple(ROLES))
    for name in ('inputs', 'inputs-sha256', 'output-root', 'device'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    analyze(args.kind, args.inputs, args.inputs_sha256, args.output_root, device=args.device)
    return 0
