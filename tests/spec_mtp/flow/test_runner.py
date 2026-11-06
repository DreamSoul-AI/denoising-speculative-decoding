"""Managed diagnostics preserve formal Flow results and original errors."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from spec_mtp.flow.context import FlowContext
from spec_mtp.flow.runner import FlowRunner
from spec_mtp.structure.artifact.layout import ArtifactLayout
from spec_mtp.structure.artifact.result.io import load_result, write_result
from spec_mtp.structure.system.config import SystemConfig
from spec_mtp.structure.system.factory import SystemFactory
from spec_mtp.structure.system.logger import Logger


def _io_context(tmp_path):
    study = tmp_path / 'study'
    layout = ArtifactLayout(root=study / 'runs/r1', study_dir=study)
    layout.ensure()
    ctx = FlowContext(study_dir=study, layout=layout, config={})
    ctx.state['result_draft'] = {'status': 'succeeded', 'control': {},
                                 'metrics': {'saved_count': 1}, 'paths': {'result': str(layout.result_path)}}
    return ctx


@pytest.mark.parametrize('point', ['constructor', 'start', 'execute', 'succeeded'])
def test_flow_diagnostic_failures_preserve_success_and_one_logger(tmp_path, monkeypatch, point):
    from spec_mtp.flow import runner as implementation
    from spec_mtp.structure.system import logger as logger_module
    ctx = _io_context(tmp_path)
    path = ctx.layout.assets_dir / 'logs/events.jsonl'
    primary = OSError('diagnostic persistence failed')
    calls, constructed = [], []
    original_init, original_open, original_text = Logger.__init__, Path.open, logger_module._text
    def initialized(self, *args, **kwargs):
        constructed.append(self)
        assert len(constructed) == 1, 'Flow and System must share a single Logger'
        original_init(self, *args, **kwargs)
    def opened(candidate, *args, **kwargs):
        if point == 'constructor' and candidate == path: raise primary
        return original_open(candidate, *args, **kwargs)
    def text(message):
        if ((point == 'start' and message.startswith('start phases=')) or
                (point == 'execute' and message == 'execute finished') or
                (point == 'succeeded' and message == 'succeeded')): raise primary
        return original_text(message)
    def prepare(current):
        calls.append('prepare')
        current.state['system'] = SystemFactory.build(SystemConfig(), current.layout.assets_dir,
                                                       logger=current.state['logger'])
        assert current.state['system'].logger is current.state['logger'] is constructed[0]
    def execute(current):
        calls.append('execute'); current.state['execute'] = {'finished': True}
        current.state['system'].logger.info('execute finished')
    def persist(current):
        calls.append('write'); write_result(current.layout.result_path, current.state['result_draft'])
    modules = {'prepare': SimpleNamespace(run=prepare), 'execute': SimpleNamespace(run=execute),
               'write': SimpleNamespace(run=persist)}
    monkeypatch.setattr(Logger, '__init__', initialized)
    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(logger_module, '_text', text)
    monkeypatch.setattr(implementation, 'import_module', lambda name: modules[name.rsplit('.', 1)[-1]])
    monkeypatch.setattr(implementation, 'run_study_phase', lambda *a, **k: None)
    expected = dict(ctx.state['result_draft'])
    result = FlowRunner(phases=['prepare', 'execute', 'write']).run(ctx)
    assert result == ctx.layout.result_path and load_result(result) == expected
    assert calls == ['prepare', 'execute', 'write'] and ctx.state['execute'] == {'finished': True}
    assert len(constructed) == 1 and constructed[0].failure_count >= 1


@pytest.mark.parametrize('result_write_fails', [False, True])
def test_flow_keeps_business_exception_when_diagnostics_and_failure_result_fail(tmp_path, monkeypatch, result_write_fails):
    from spec_mtp.flow import runner as implementation
    ctx = _io_context(tmp_path)
    ctx.state['logger'] = Logger(ctx.layout.assets_dir, managed=True)
    primary = ValueError('actual business failure')
    diagnostic = OSError('diagnostic sink unavailable')
    writer_error = PermissionError('result sink unavailable')
    original_open = Path.open
    def opened(path, *args, **kwargs):
        if path == ctx.state['logger'].path: raise diagnostic
        return original_open(path, *args, **kwargs)
    def execute(current): raise primary
    def failed_write(*args, **kwargs): raise writer_error
    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(implementation, 'import_module', lambda name: SimpleNamespace(run=execute))
    monkeypatch.setattr(implementation, 'run_study_phase', lambda *a, **k: None)
    if result_write_fails: monkeypatch.setattr(implementation, 'write_result', failed_write)
    with pytest.raises(ValueError) as caught: FlowRunner(phases=['execute']).run(ctx)
    assert caught.value is primary
    if result_write_fails: assert not ctx.layout.result_path.exists()
    else:
        result = load_result(ctx.layout.result_path)
        assert result['status'] == 'failed' and result['error'] == 'ValueError: actual business failure'
        assert result['metrics'] == {'saved_count': 1}
    assert ctx.state['logger'].failure_count >= 1
