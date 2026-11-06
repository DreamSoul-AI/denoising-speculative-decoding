"""Cancellation reaps the directly owned CUDA writer before launch returns."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_cancelled_scheduler_reaps_its_cuda_writer_and_preserves_original_results(
    tmp_path, monkeypatch, cuda_decode, record_property, error_type,
):
    import torch
    from spec_mtp.structure.artifact.result.io import write_result
    from spec_mtp.structure.make import schedule

    device = cuda_decode.device
    assert torch.cuda.is_available() and device.type == 'cuda'
    study = tmp_path / 'runtime'
    study.mkdir()
    (study / 'study.yaml').write_text('study: cuda-cancel-cleanup\n', encoding='utf-8')
    jobs, rows = [], []
    for run_id in ('first', 'later'):
        config = study / 'runs' / run_id / 'config.yaml'
        config.parent.mkdir(parents=True)
        config.write_text(f'id: {run_id}\nseed: 7\nalgorithm:\n  mode: train\n'
                          f'system:\n  config:\n    device: {device}\n', encoding='utf-8')
        jobs.append({'run_id': run_id, 'config': str(config), 'mode': 'train',
                     'device': str(device), 'gpu': str(device.index)})
        rows.append({'id': run_id, 'run_dir': run_id, 'seed': 7,
                     'config': config.relative_to(study).as_posix()})
    (study / 'index.json').write_text(json.dumps({
        'study': 'cuda-cancel-cleanup', 'experiments': [{'factors': {}, 'runs': rows}],
    }), encoding='utf-8')
    result = study / 'runs' / 'first' / 'result.json'
    write_result(result, {'status': 'succeeded', 'control': {}, 'metrics': {'preserved': 7}, 'paths': {}})
    other = study / 'runs' / 'later' / 'untouched-input.bin'
    other.write_bytes(b'other Run input remains unchanged')
    protected = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in study.rglob('*') if path.is_file()}
    original_index_sha = hashlib.sha256((study / 'index.json').read_bytes()).hexdigest()
    original_config_sha = hashlib.sha256(Path(jobs[0]['config']).read_bytes()).hexdigest()
    assets = study / 'runs' / 'first' / 'assets'
    ready, stderr = tmp_path / 'worker-ready.json', tmp_path / 'worker-stderr.log'
    child_code = '''
import json, os, sys, time
from pathlib import Path
import torch
from spec_mtp.structure.system.logger import Logger
device = torch.device(sys.argv[1])
assert torch.cuda.is_available() and device.type == 'cuda'
with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
    model = torch.nn.Linear(2, 1, bias=False, device=device)
    model.weight.fill_(2)
    output = model(torch.tensor([[1., 2.]], device=device))
    assert output.device == model.weight.device == device
    assert torch.equal(output, torch.tensor([[6.]], device=device))
    torch.cuda.synchronize(device)
logger = Logger(Path(sys.argv[2]), managed=True)
logger.info('CUDA child ready after a real Linear forward')
ready = Path(sys.argv[3])
pending = ready.with_suffix('.pending')
pending.write_text(json.dumps({'pid': os.getpid(), 'device': str(output.device),
                              'output': output.item()}), encoding='utf-8')
os.replace(pending, ready)
counter = 0
while True:
    logger.info(f'owned CUDA writer heartbeat={counter}')
    counter += 1
    time.sleep(.02)
'''
    primary = error_type('cancel this owned launch')
    processes, dispatched, events = [], [], []
    real_popen = subprocess.Popen

    class OwnedWorker:
        def __init__(self, process):
            self.process = process
            self.pid = process.pid

        def poll(self):
            return self.process.poll()

        def terminate(self):
            events.append('terminate')
            self.process.terminate()

        def kill(self):
            events.append('kill')
            self.process.kill()

        def wait(self, timeout=None):
            events.append(('wait', timeout))
            if timeout is None:
                deadline = time.monotonic() + 30
                while not ready.is_file():
                    assert self.process.poll() is None, stderr.read_text(encoding='utf-8')
                    assert time.monotonic() < deadline, 'CUDA child did not publish its ready receipt'
                    time.sleep(.02)
                facts = json.loads(ready.read_text(encoding='utf-8'))
                assert facts == {'pid': self.process.pid, 'device': str(device), 'output': 6.0}
                cuda_decode.observed_devices.append(facts['device'])
                record_property('owned_child_cuda', facts)
                anchor = assets / 'logs' / 'events.jsonl'
                initial = anchor.stat().st_size
                while anchor.stat().st_size == initial:
                    assert self.process.poll() is None
                    assert time.monotonic() < deadline, 'owned child did not continue writing before cancellation'
                    time.sleep(.02)
                raise primary
            return self.process.wait(timeout=timeout)

    def controlled_popen(command, **options):
        dispatched.append(command[-1])
        assert dispatched == ['first'], 'cancellation must stop later groups and retries'
        with stderr.open('wb') as errors:
            process = real_popen([sys.executable, '-u', '-c', child_code, str(device), str(assets), str(ready)],
                                 cwd=options['cwd'], env=options['env'], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=errors)
        processes.append(process)
        return OwnedWorker(process)

    monkeypatch.setattr(schedule.subprocess, 'Popen', controlled_popen)
    publish = getattr(schedule, '_write_interruption_receipt', None)
    published = []

    def publish_after_exit(study_dir, body):
        assert events[-1] == ('wait', 5)
        assert processes and all(process.poll() is not None for process in processes)
        assert not hasattr(primary, 'cleanup_receipt_path')
        path = publish(study_dir, body)
        assert path.is_file()
        published.append(path)
        return path

    if publish is not None:
        monkeypatch.setattr(schedule, '_write_interruption_receipt', publish_after_exit)
    try:
        with pytest.raises(error_type) as caught:
            schedule.launch_jobs(study, jobs, round_size=1, retry_failed=True, console='shared')
        assert caught.value is primary
        assert dispatched == ['first'] and len(processes) == 1
        process, = processes
        assert process.poll() is not None, 'scheduler returned while its direct child was still running'
        assert events.count('terminate') == 1 and ('wait', 5) in events
        assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in protected} == protected
        log_dir = assets / 'logs'
        stopped = {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                   for path in log_dir.iterdir()}
        assert 'events.jsonl' in stopped and 'events.policy.json' in stopped
        for _ in range(3):
            time.sleep(.05)
            assert process.poll() is not None
            assert {path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                    for path in log_dir.iterdir()} == stopped
        record_property('owned_child_returncode', process.returncode)
        record_property('stopped_diagnostic_sha256', {name: value[0] for name, value in stopped.items()})
        record_property('post_exit_stability_checks', 3)
        receipt_path = Path(primary.cleanup_receipt_path)
        assert receipt_path.is_file() and receipt_path.resolve().is_relative_to(study.resolve())
        assert published == [receipt_path]
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert receipt['schema'] == 1 and receipt['scope'] == 'owned_current_group_only'
        assert receipt['study'] == str(study.resolve()) and receipt['launcher_pid'] == os.getpid()
        assert receipt['primary_exception_type'] == error_type.__name__
        assert receipt['index_sha256'] == original_index_sha
        assert receipt['jobs'] == [{'run_id': 'first', 'config_sha256': original_config_sha}]
        worker, = receipt['workers']
        assert worker['run_id'] == 'first' and worker['pid'] == process.pid
        assert worker['terminate_requested'] is True and worker['kill_requested'] is False
        assert worker['direct_exit_confirmed'] is True
        assert type(worker['returncode']) is int and worker['returncode'] == process.returncode
        assert worker['cleanup_error_types'] == []
        assert receipt['cleanup_wait_seconds'] == 5
        assert receipt['descendants'] == receipt['writer_stop'] == receipt['unreturned_popen'] == 'unknown'
        record_property('cleanup_receipt_path', str(receipt_path))
        record_property('cleanup_receipt_sha256', hashlib.sha256(receipt_path.read_bytes()).hexdigest())
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
