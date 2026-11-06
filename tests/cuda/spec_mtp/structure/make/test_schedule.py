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


@pytest.mark.parametrize('mode', ['timeout', 'normal'])
def test_frozen_run_deadline_reaps_real_cuda_writer_and_keeps_normal_completion(
    tmp_path, monkeypatch, cuda_decode, record_property, mode,
):
    import torch
    import yaml
    from spec_mtp.structure.artifact.result.io import write_result
    from spec_mtp.structure.make import schedule
    from spec_mtp.structure.make.write import expand_study

    device = cuda_decode.device
    assert torch.cuda.is_available() and device.type == 'cuda'
    budget = .1 if mode == 'timeout' else 30
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'study.yaml').write_text(yaml.safe_dump({
        'study': 'cuda-worker-deadline', 'freeze': True, 'seeds': [7, 11],
        'flow': {'study_phases': False},
    }), encoding='utf-8')
    (source / 'experiment_config.yaml').write_text(yaml.safe_dump({
        'experiment': 'cuda-worker-deadline', 'seed': 7, 'data': {'name': 'fixture'},
        'model': {'name': 'fixture'}, 'algorithm': {'mode': 'train'},
        'system': {'config': {'device': str(device), 'worker_timeout_seconds': budget}},
    }), encoding='utf-8')
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}
    expanded = expand_study(source, run_root=tmp_path / 'runtime')
    study = expanded['study_dir']
    jobs = schedule.plan_jobs(study, expanded['configs'], init_gpu=0, num_gpus=1, include_done=True)
    first, later = [job['run_id'] for job in jobs]
    for index, job in enumerate(jobs):
        write_result(Path(job['config']).with_name('result.json'), {
            'status': 'succeeded', 'control': {}, 'metrics': {'preserved': index}, 'paths': {},
        })
    other = study / 'runs' / later / 'untouched-input.bin'
    other.write_bytes(b'other Run input remains unchanged')
    protected = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in study.rglob('*') if path.is_file()}
    index_sha = hashlib.sha256((study / 'index.json').read_bytes()).hexdigest()
    config_sha = hashlib.sha256(Path(jobs[0]['config']).read_bytes()).hexdigest()
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
logger.info('deadline child ready after a real CUDA Linear forward')
ready = Path(sys.argv[3])
pending = ready.with_suffix('.pending')
pending.write_text(json.dumps({'pid': os.getpid(), 'device': str(output.device),
                              'output': output.item()}), encoding='utf-8')
os.replace(pending, ready)
if sys.argv[4] == 'normal':
    logger.info('normal direct child completed')
else:
    while True:
        logger.info('owned deadline writer heartbeat')
        time.sleep(.02)
'''
    real_popen = subprocess.Popen
    processes, dispatched, waits, actions, publications = [], [], [], [], []
    starts = {}
    original_publish = schedule._write_interruption_receipt

    class OwnedWorker:
        def __init__(self, process, run_id):
            self.process, self.run_id, self.pid = process, run_id, process.pid

        def poll(self):
            return self.process.poll()

        def terminate(self):
            actions.append((self.run_id, 'terminate'))
            self.process.terminate()

        def kill(self):
            actions.append((self.run_id, 'kill'))
            self.process.kill()

        def wait(self, timeout=None):
            if mode == 'timeout' and timeout is None:
                pytest.fail('deadline monitor bypassed: old serial wait reached after real CUDA readiness')
            code = self.process.wait(timeout=timeout)
            waits.append((self.run_id, timeout, code))
            return code

    def controlled_popen(command, **options):
        run_id = command[-1]
        dispatched.append(run_id)
        assert dispatched == ([first] if mode == 'timeout' else [first, later][:len(dispatched)])
        ready, errors = tmp_path / (run_id + '.ready.json'), tmp_path / (run_id + '.stderr.log')
        assets = study / 'runs' / run_id / 'assets'
        starts[run_id] = time.monotonic()
        with errors.open('wb') as stream:
            process = real_popen([sys.executable, '-u', '-c', child_code, str(device), str(assets),
                                  str(ready), mode], cwd=options['cwd'], env=options['env'],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stream)
        processes.append((run_id, process))
        startup_deadline = time.monotonic() + 30
        while not ready.is_file():
            assert process.poll() is None, errors.read_text(encoding='utf-8')
            assert time.monotonic() < startup_deadline, 'real CUDA child did not become ready'
            time.sleep(.02)
        facts = json.loads(ready.read_text(encoding='utf-8'))
        assert facts == {'pid': process.pid, 'device': str(device), 'output': 6.0}
        cuda_decode.observed_devices.append(facts['device'])
        record_property('deadline_child_' + run_id, facts)
        if mode == 'timeout':
            while time.monotonic() - starts[run_id] <= budget + .02:
                time.sleep(.01)
            assert process.poll() is None
        return OwnedWorker(process, run_id)

    def publish_after_exit(study_dir, body):
        assert processes and all(process.poll() is not None for _, process in processes)
        assert all(any(row[0] == run_id for row in waits) for run_id, _ in processes)
        path = original_publish(study_dir, body)
        publications.append(path)
        return path

    monkeypatch.setattr(schedule.subprocess, 'Popen', controlled_popen)
    monkeypatch.setattr(schedule, '_write_interruption_receipt', publish_after_exit)
    try:
        if mode == 'timeout':
            with pytest.raises(subprocess.TimeoutExpired) as caught:
                schedule.launch_jobs(study, jobs, round_size=1, retry_failed=True, console='shared')
            failure = caught.value
            assert failure.run_id == first and failure.worker_timeout_seconds == budget
            assert failure.elapsed_seconds >= budget
            assert dispatched == [first]
            receipt_path = Path(failure.cleanup_receipt_path)
            assert publications == [receipt_path] and receipt_path.is_file()
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            assert receipt['deadline'] == {
                'run_id': first, 'worker_timeout_seconds': budget, 'elapsed_seconds': failure.elapsed_seconds,
                'clock': 'monotonic', 'start': 'before_popen', 'poll_interval_seconds': .05,
            }
            assert receipt['scope'] == 'owned_current_group_only' and receipt['index_sha256'] == index_sha
            assert receipt['jobs'] == [{'run_id': first, 'config_sha256': config_sha}]
            worker, = receipt['workers']
            assert worker['run_id'] == first and worker['pid'] == processes[0][1].pid
            assert worker['terminate_requested'] is True and worker['kill_requested'] is False
            assert worker['direct_exit_confirmed'] is True
            assert worker['returncode'] == processes[0][1].returncode
            assert receipt['descendants'] == receipt['writer_stop'] == receipt['unreturned_popen'] == 'unknown'
            assert actions == [(first, 'terminate')] and any(row[:2] == (first, 5) for row in waits)
            record_property('deadline_receipt', str(receipt_path))
        else:
            assert schedule.launch_jobs(study, jobs, round_size=1, retry_failed=True, console='shared') == [0, 0]
            assert dispatched == [first, later] and not actions and not publications
            assert not (study / 'scripts' / 'launch-attempts').exists()
        assert all(process.poll() is not None for _, process in processes)
        assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in protected} == protected
        assert {path.name: path.read_bytes() for path in source.iterdir()} == source_before
        log_files = [path for run_id, _ in processes
                     for path in (study / 'runs' / run_id / 'assets' / 'logs').iterdir()]
        stopped = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                   for path in log_files}
        for _ in range(3):
            time.sleep(.05)
            assert {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                    for path in log_files} == stopped
        record_property('deadline_mode', mode)
        record_property('deadline_child_returncodes', {run_id: process.returncode for run_id, process in processes})
        record_property('deadline_stopped_sha256', {str(path): value[0] for path, value in stopped.items()})
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
