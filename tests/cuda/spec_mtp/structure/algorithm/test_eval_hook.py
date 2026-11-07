"""Diagnostic faults must not interrupt real CUDA evaluation finalization."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spec_mtp.structure.algorithm.eval_hook import eval_test_split
from spec_mtp.structure.algorithm.tracker import AlgorithmTracker
from spec_mtp.structure.system.logger import Logger


@pytest.mark.parametrize('sink', ['file', 'console'])
def test_cuda_evaluation_finalizes_tracker_after_diagnostic_failure(tmp_path, monkeypatch, record_property, sink):
    import torch

    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device('cuda', torch.cuda.current_device())
    print(f'evaluation fault acceptance device={device} name={torch.cuda.get_device_name(device)}')
    record_property('cuda_device', str(device))
    observed = []
    with torch.random.fork_rng(devices=[device.index]):
        module = torch.nn.Linear(2, 2, device=device)
    with torch.no_grad():
        module.weight.copy_(torch.eye(2, device=device))
        module.bias.zero_()
    images = torch.tensor([[2., 0.], [0., 2.]], device=device)
    targets = torch.tensor([0, 1], device=device)

    def check_forward(current, args, output):
        assert current.weight.device == args[0].device == output.device == device
        observed.append(str(output.device))

    hook = module.register_forward_hook(check_forward)
    data = SimpleNamespace(iter_batches=lambda split: iter([(images, targets)]))
    model, system = SimpleNamespace(module=module), SimpleNamespace(device=str(device))
    expected_tracker, tracker = AlgorithmTracker(tmp_path / 'reference'), AlgorithmTracker(tmp_path / 'actual')
    logger = Logger(tmp_path / 'actual', managed=True)
    expected = eval_test_split(expected_tracker, None, data, model, system)
    original_open = Path.open

    def failed_open(path, *args, **kwargs):
        if path == logger.path:
            raise PermissionError('diagnostic sink unavailable')
        return original_open(path, *args, **kwargs)

    class FailedConsole:
        def write(self, text):
            raise BrokenPipeError('diagnostic console unavailable')

        def flush(self):
            pass

    try:
        with monkeypatch.context() as patch:
            if sink == 'file':
                patch.setattr(Path, 'open', failed_open)
            else:
                import sys
                patch.setattr(sys, 'stdout', FailedConsole())
            actual = eval_test_split(tracker, logger, data, model, system)
    finally:
        hook.remove()
    assert actual == expected and actual['Accuracy'] == 100.0
    assert module.training and not tracker.has_samples('test')
    assert tracker.state_dict()['splits'] == expected_tracker.state_dict()['splits']
    assert json.loads(tracker.state_path.read_text())['last_segment']['test'] == expected
    assert tracker.jsonl_path.read_bytes() == expected_tracker.jsonl_path.read_bytes()
    assert logger.failure_count == 1 and observed == [str(device), str(device)]
    record_property('forward_devices', observed)
