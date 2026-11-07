"""A fixed diagnostic reservation admits real CUDA Flow work before its phases."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('reservation', ['ready', 'missing_ledger'])
def test_fixed_study_budget_admits_cuda_flow_and_refuses_missing_reservation(
    tmp_path, monkeypatch, cuda_decode, record_property, reservation,
):
    import torch
    import yaml
    from spec_mtp.flow import runner as implementation
    from spec_mtp.flow.context import FlowContext
    from spec_mtp.structure.artifact.config.io import load_config
    from spec_mtp.structure.artifact.layout import ArtifactLayout
    from spec_mtp.structure.artifact.result.io import load_result, write_result
    from spec_mtp.structure.make.write import expand_study
    from spec_mtp.structure.system.config import SystemConfig
    from spec_mtp.structure.system.factory import SystemFactory

    device = cuda_decode.device
    assert torch.cuda.is_available() and device.type == 'cuda'
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'study.yaml').write_text(yaml.safe_dump({
        'study': 'fixed-budget-cuda-engineering', 'seeds': [7],
        'diagnostic_budget_bytes': 65536, 'flow': {'study_phases': False},
    }), encoding='utf-8')
    (source / 'experiment_config.yaml').write_text(yaml.safe_dump({
        'experiment': 'fixed-budget-cuda-engineering', 'seed': 7,
        'data': {'name': 'fixture'}, 'model': {'name': 'fixture'},
        'algorithm': {'mode': 'eval', 'config': {'version': 'round59-cuda-engineering'}},
        'system': {'config': {'device': str(device)}},
    }), encoding='utf-8')
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}
    expanded = expand_study(source, run_root=tmp_path / 'runtime')
    config_path, = expanded['configs']
    layout = ArtifactLayout(root=config_path.parent, study_dir=expanded['study_dir'])
    ctx = FlowContext(study_dir=expanded['study_dir'], layout=layout, config=load_config(config_path))
    ledger = expanded['study_dir'] / '.diagnostic-budget.json'
    assert ledger.is_file()
    family_before = {path.name: path.read_bytes() for path in (layout.assets_dir / 'logs').iterdir()}
    if reservation == 'missing_ledger':
        ledger.unlink()  # Damage only this test's newly materialized reservation.
    phases, forwards = [], []
    record_property('budget_forward_devices', forwards)

    def execute(current):
        phases.append('execute')
        system = SystemFactory.build(SystemConfig(config={'device': str(device)}), layout.assets_dir,
                                     logger=current.state['logger'])
        with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
            model = torch.nn.Linear(2, 1, bias=False, device=device)
            model.weight.fill_(2)
        model = system.place_module(model)
        output = model(torch.tensor([[1., 2.]], device=device))
        assert output.device == model.weight.device == device
        forwards.append(str(output.device))
        cuda_decode.observed_devices.append(str(output.device))
        assert torch.equal(output, torch.tensor([[6.]], device=device))
        torch.cuda.synchronize(device)
        current.state['result_draft'] = {
            'status': 'succeeded', 'control': {}, 'metrics': {'cuda_sum': output.item()}, 'paths': {},
        }

    def publish(current):
        phases.append('write')
        write_result(layout.result_path, current.state['result_draft'])

    modules = {'execute': SimpleNamespace(run=execute), 'write': SimpleNamespace(run=publish)}
    monkeypatch.setattr(implementation, 'import_module', lambda name: modules[name.rsplit('.', 1)[-1]])
    monkeypatch.setattr(implementation, 'run_study_phase', lambda *args, **kwargs: None)
    if reservation == 'missing_ledger':
        with pytest.raises(FileNotFoundError):
            implementation.FlowRunner(phases=['execute', 'write']).run(ctx)
        assert phases == forwards == []
        assert load_result(layout.result_path)['status'] == 'failed'
        assert {path.name: path.read_bytes() for path in (layout.assets_dir / 'logs').iterdir()} == family_before
    else:
        assert implementation.FlowRunner(phases=['execute', 'write']).run(ctx) == layout.result_path
        assert phases == ['execute', 'write'] and forwards == [str(device)]
        result = load_result(layout.result_path)
        assert result['status'] == 'succeeded' and result['metrics'] == {'cuda_sum': 6.0}
    assert {path.name: path.read_bytes() for path in source.iterdir()} == source_before
