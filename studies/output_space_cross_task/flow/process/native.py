"""Explicit native exports, H1/H3, canonical and saved cross/Phi audits; no model forwards."""
import json
from pathlib import Path

from studies.output_space_cross_task.flow.process.run import PLAN_ROOT, bind_inputs
from studies.output_space_cross_task.structure import native_readout as protocol

KINDS = ('export', 'hypotheses', 'mre', 'cross-phi', 'canonical')


def _export(session, index, native, hashes):
    import yaml
    from types import SimpleNamespace
    from studies._shared.structure.evidence import digest
    from studies._shared.structure.config import legacy_cfg
    from studies._shared.structure.algorithm import effective_spec

    def read(role):
        if role not in index['files']:
            raise ValueError('original native input is missing; identity remains unknown: ' + role)
        return session.read_json(index['files'][role]['path'])

    original_tool = index['files'].get('native_tool', {})
    if (hashes.get('native_tool') != native['tool_sha256']
            or original_tool.get('member') != 'studies/output_space_cross_task/check_hypotheses.py'):
        raise ValueError('original native exporter bytes and Git identity are required')
    process = read('process')
    assert process['complete'] and process['spec_mtp']['passed']
    rendered = read('manifest')['rendered_prompts']
    assert [p['question_id'] for p in rendered] == protocol.QIDS
    experiments = read('index')['experiments']
    assert len(experiments) == len(native['runs']) == 4
    run_ids = [entry['runs'][0]['id'] for entry in experiments if len(entry['runs']) == 1]
    assert len(run_ids) == len(set(run_ids)) == 4
    rows = {item['run_id']: item for item in native['runs']}
    assert set(rows) == set(run_ids)
    checked, eos = [], None
    for run_id in run_ids:
        original = rows[run_id]
        observation, inputs = read(run_id + ':raw'), read(run_id + ':inputs')
        assert hashes[run_id + ':raw'] == original['raw_sha256']
        assert observation['inputs_digest'] == digest(inputs) and observation['out'] == original['raw']
        cfg = yaml.safe_load(session.read_text(index['files'][run_id + ':config']['path']))
        result, sources = read(run_id + ':result'), read(run_id + ':sources')
        assert result['status'] == 'succeeded' and result['spec_mtp']['passed']
        assert sources['digest'] == result['spec_mtp']['sources_digest'] == native['source_digest']
        assert sources['config_digest'] == digest(cfg)
        saved, settings = inputs['settings'], cfg['algorithm']['config']
        assert isinstance(saved['data'], dict) and isinstance(saved['model'], dict)
        device = saved['eval']['device']
        assert device == cfg['system']['config']['device']
        actual = inputs['device']
        assert actual == device or (device == 'cuda' and actual.startswith('cuda:'))
        algorithm = SimpleNamespace(setting=lambda key, default=None: settings.get(key, default))
        expected = legacy_cfg(SimpleNamespace(meta={'data_cfg': saved['data']}),
            SimpleNamespace(meta={'model_cfg': saved['model']}), SimpleNamespace(device=device), algorithm, cfg['seed'])
        expected.spec = effective_spec(expected.spec, algorithm)
        assert saved == expected and settings['version'] == native['version']
        spec = settings.get('spec', {})
        case = 'AR' if settings['task'] == 'ar_baseline' else (
            'frozen_mean_chain' if spec['method'] == 'esp' else 'output_order' + str(spec['output']['order']))
        assert case == original['case'] and spec == original['spec']
        raw = observation['out']
        assert len(raw['evidence']) == len(inputs['prompts']) == 4
        assert all(a['input_ids'] == b['input_ids'] == c['input_ids'] for a, b, c in zip(rendered, inputs['prompts'], raw['evidence']))
        checked.append({'case': case, 'run_id': run_id, 'spec': spec, 'raw': raw,
                        'raw_sha256': hashes[run_id + ':raw'], 'runtime': inputs['runtime'], 'model_digest': inputs['model_digest']})
        if case == 'AR':
            eos = inputs['eos_ids']
    assert checked == native['runs'] and len({r['model_digest'] for r in checked}) == 1
    cases = {r['case']: r for r in checked}
    assert set(cases) == {'AR', 'frozen_mean_chain', 'output_order0', 'output_order1'}
    assert all(all(r['raw']['evidence'][i]['output_ids'] == cases['AR']['raw']['evidence'][i]['output_ids'] for r in checked) for i in range(4))
    return native, eos


def analyze(kind, inputs, inputs_sha256, output_root, *, device=None):
    if kind == 'mre':
        from spec_mtp.structure.artifact.layout import validate_study_root
        from spec_mtp.structure.artifact.paths import validate_output_root
        from studies.output_space_cross_task.flow.process.run import CrossTaskReadout
        if inputs is not None or inputs_sha256 is not None or device is not None:
            raise ValueError('mre is a pure synthetic JSON fixture without historical inputs or device')
        root = Path(__file__).resolve().parents[4]
        session = CrossTaskReadout(root, validate_study_root(validate_output_root(output_root, root)))
        n, d = protocol.fixture()
        verdict = protocol.verdict(n, d, [])
        assert verdict['H1'] == 'falsified_bounded_occurrence'
        result = {'synthetic_only_no_model_evidence': True, 'verdict': verdict}
        filename = 'synthetic_mre.json'
    else:
        roles = {'export': {'native'}, 'hypotheses': {'native', 'cross'},
                 'cross-phi': {'native', 'cross', 'phi'}, 'canonical': {'canonical'}}
        if kind not in roles:
            raise ValueError('unknown native saved-readout command')
        producers = {}
        if kind in ('hypotheses', 'cross-phi'):
            producers['cross'] = {'tool_sha256': ('studies/_shared/structure/diagnose_study_windows.py', 'tools/diagnose_study_windows.py'),
                                  'plan_sha256': PLAN_ROOT + 'PLAN.md'}
        if kind == 'canonical':
            producers['canonical'] = {'tool': ('studies/_shared/structure/diagnose_q241_canonical.py', 'tools/diagnose_q241_canonical.py'),
                                      'plan': PLAN_ROOT + 'CANONICAL_PLAN.md'}
        if kind == 'cross-phi':
            producers['phi'] = {'tool': ('studies/_shared/structure/diagnose_q178_phi_rms.py', 'tools/diagnose_q178_phi_rms.py'),
                                'plan': 'studies/output_space_extension/docs/PHI_RMS_PLAN.md',
                                'common': 'studies/output_space_extension/kernel_common.py'}
        session, index, reports, report_hashes, hashes = bind_inputs(kind, inputs, inputs_sha256, output_root,
            roles=roles[kind], producers=producers,
            quality_fields={role: 'study_quality_passed' if role == 'native' else 'quality_passed' for role in roles[kind]},
            source_roles=roles[kind] - {'canonical'})
        expected = {'native': PLAN_ROOT + 'execution_v1.json', 'cross': PLAN_ROOT + 'transfer_diagnosis_v1.json',
                    'canonical': PLAN_ROOT + 'canonical_diagnosis_v1.json',
                    'phi': 'studies/output_space_extension/docs/phi_rms_diagnosis_v1.json'}
        if any(index['reports'][role]['member'] != expected[role] for role in roles[kind]):
            raise ValueError('historical report member differs from its native protocol role')
        bindings = {'analysis': session.sha(protocol.__file__),
                    'cross_report': report_hashes.get('cross'), 'phi_report': report_hashes.get('phi'),
                    'report': report_hashes.get('canonical')}
        if kind in ('export', 'hypotheses'):
            if device is not None:
                raise ValueError('native export/H1/H3 use saved JSON only, without a tensor device')
            result, eos = _export(session, index, reports['native'], hashes)
            filename = 'native_export_readout.json'
            from studies._shared.structure import extension_diagnostics
            rows = {row['case']: row for row in result['runs']}
            selection = {'matrix_sha256': report_hashes['native'],
                'readout_selection_module_sha256': session.sha(extension_diagnostics.__file__),
                'prompts': [{'question_id': qid, **extension_diagnostics.choose_window(
                    rows['output_order0']['raw']['evidence'][i]['stats']['trace'],
                    rows['output_order1']['raw']['evidence'][i]['stats']['trace'])}
                    for i, qid in enumerate(protocol.QIDS)]}
            session.write_text(session.output / 'window_selection_readout.json', json.dumps(selection, indent=2) + '\n')
            if kind == 'hypotheses':
                cross = reports['cross']
                assert cross['source_binding']['matrix_sha256'] == report_hashes['native']
                assert hashes['tensor'] == cross['raw_tensor_sha256']
                result = protocol.verdict(result, cross, eos)
                filename = 'hypotheses_readout.json'
        else:
            import torch
            if device is None:
                raise ValueError('saved tensor audits require an explicit available CUDA device')
            device = torch.device(device)
            if device.type != 'cuda' or device.index is None or not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
                raise ValueError('saved tensor audits require an explicit available CUDA device')
            role = 'canonical' if kind == 'canonical' else 'cross'
            assert hashes['tensor'] == reports[role]['raw_tensor_sha256']
            if kind == 'cross-phi':
                assert reports['cross']['source_binding']['matrix_sha256'] == report_hashes['native']
                assert hashes['context_tensor'] == reports['phi']['raw_tensor_sha256']
            session.verify()
            print(f'saved native {kind} device={device} name={torch.cuda.get_device_name(device)}', flush=True)
            data = torch.load(session._input(index['files']['tensor']['path']), weights_only=True, map_location=device, mmap=True)
            if kind == 'canonical':
                result = protocol.analyze_canonical(reports['canonical'], data, bindings)
                filename = 'canonical_readout.json'
            else:
                context = torch.load(session._input(index['files']['context_tensor']['path']), weights_only=True, map_location=device, mmap=True)
                result = {'cross': protocol.analyze_cross(reports['native'], reports['cross'], data, bindings),
                          'phi': protocol.analyze_phi(reports['phi'], context, bindings)}
                filename = 'cross_phi_readout.json'
            result['readout_device'] = str(device)
            result['readout_device_name'] = torch.cuda.get_device_name(device)
        session.write_text(session.output / 'native_scope.json', json.dumps({
            'native_run_revalidated': False, 'readout_model_forwards': 0, 'readout_sampling_draws': 0,
            'original_native_plan_binding': 'not supplied by the original native export; unknown',
            'scope': 'saved evidence audit only; original gate/tool/plan identities remain historical',
        }, indent=2) + '\n')
    session.write_text(session.output / filename, json.dumps(result, indent=2, allow_nan=False) + '\n')
    session.finish()
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=KINDS)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args(argv)
    if args.kind != 'mre' and (args.inputs is None or args.inputs_sha256 is None):
        parser.error('saved readouts require --inputs and --inputs-sha256')
    analyze(args.kind, args.inputs, args.inputs_sha256, args.output_root, device=args.device)
    return 0
