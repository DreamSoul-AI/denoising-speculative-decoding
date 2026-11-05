"""A real CUDA training checkpoint continues identically in a fresh process."""

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import sysconfig

import pytest


def _worker(mode, output, checkpoint=None):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Dataset, RandomSampler

    from spec_mtp.structure.algorithm.config import AlgorithmConfig
    from spec_mtp.structure.algorithm.resume import capture_rng
    from spec_mtp.structure.algorithm.tracker import AlgorithmTracker
    from spec_mtp.structure.algorithm.train import TrainAlgorithm
    from spec_mtp.structure.data.factory import Data
    from spec_mtp.structure.model.factory import Model
    from spec_mtp.structure.system.config import SystemConfig
    from spec_mtp.structure.system.factory import System
    from spec_mtp.structure.system.logger import Logger
    from spec_mtp.structure.system.runtime import apply_runtime

    assert torch.cuda.is_available(), 'CR5 requires a real CUDA device; no CPU fallback'
    scenario = os.environ.get('SD_TEST_SCENARIO', 'step')
    epoch_mode = scenario in ('epoch', 'early_stop')
    eval_events = []
    writes = {'checkpoint': 0, 'tracker': 0}
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    apply_runtime(912 if mode in ('resumed', 'terminal') else 314, SystemConfig(config={
        'deterministic': True, 'cudnn_benchmark': False,
        'cudnn_allow_tf32': False, 'matmul_allow_tf32': False}))
    output.mkdir(parents=True, exist_ok=False)

    class Samples(Dataset):
        def __init__(self):
            self.seen = []

        def __len__(self):
            return 8 if epoch_mode else 32

        def __getitem__(self, index):
            self.seen.append(index)
            # Stochastic transforms exercise all three host RNG domains.
            noise = (random.random() + float(np.random.random()) + float(torch.rand(()))) / 100
            features = torch.tensor([index / 32, index % 3 / 3, index % 5 / 5, 1.0]) + noise
            return features, torch.tensor(index % 2)

    samples = Samples()
    loader = DataLoader(samples, batch_size=2, num_workers=0,
        sampler=RandomSampler(samples, generator=torch.Generator().manual_seed(53)),
        generator=torch.Generator().manual_seed(97))
    data = Data(name='cr5-local', source='test', loaders={'train': loader},
        meta={'seed': 53, 'train_size': len(samples), 'batch_size': 2,
              'config': {'local_fixture': 'cr5', 'num_workers': 0, 'scenario': scenario}})
    if os.environ['SD_TEST_REBIND'] == '1':
        data._train_set = samples
    module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.4), torch.nn.Linear(8, 2))
    initial = {key: value.detach().clone() for key, value in module.state_dict().items()}
    model = Model(name='cr5-dropout', source='test', module=module, meta={})
    logger = Logger(output / 'assets')
    logger.info(f'CR5 mode={mode} device={device} dtype=torch.float32 gpu={torch.cuda.get_device_name(device)}')

    class InterruptedSystem(System):
        def save_checkpoint(self, payload, name):
            assert mode != 'terminal', 'terminal recovery must not publish another checkpoint'
            writes['checkpoint'] += 1
            path = super().save_checkpoint(payload, name)
            interrupt_at = 2 if scenario == 'early_stop' else 1
            should_interrupt = payload['epoch'] == interrupt_at if epoch_mode else payload['step'] == 2
            if mode == 'interrupted' and should_interrupt:
                torch.cuda.synchronize(device)
                torch.save({'pid': os.getpid(), 'device': str(device),
                    'ids': list(samples.seen), 'step': payload['step'],
                    'stopped': payload['training_state']['stopped'],
                    'budget': payload['training_state']['contract']['num_steps'],
                    'epoch': payload['epoch'], 'stall': payload['training_state']['stall'],
                    'eval_events': eval_events,
                    'data': payload['training_state']['data']}, output / 'interrupted.pt')
                # Exit after the real durable checkpoint, without final-stop hooks.
                os._exit(73)
            return path

    system = InterruptedSystem(device=str(device), logger=logger, assets_dir=output / 'assets', meta={})
    settings = {'num_steps': 6, 'step_period': 2, 'eval_period': 0, 'log_period': 0,
        'checkpoint_period': 2, 'save_best': False, 'lr': .05,
        'optimizer': 'SGD', 'momentum': .9, 'scheduler': 'StepLR',
        'step_size': 1, 'gamma': .7, 'resume': False}
    if epoch_mode:
        settings.pop('num_steps')
        settings.update(progress_unit='epoch', num_epochs=5 if scenario == 'early_stop' else 3,
                        step_period=1, eval_period=1, checkpoint_period=1)
        if scenario == 'early_stop':
            settings['early_stop_patience'] = 2
        from importlib import import_module

        train_module = import_module('spec_mtp.structure.algorithm.train')

        def fixed_score(tracker, logger, data, model, system, extra, **kwargs):
            tracker.append('test', n=1, values={'Accuracy': .5})
            tracker.flush('test', progress=extra)
            tracker.save('test')
            tracker.reset('test')
            tracker.flush_state()
            return {'Accuracy': .5}

        train_module.eval_test_split = fixed_score
    if checkpoint is not None:
        settings['resume_from'] = str(checkpoint)
    class ObservedTracker(AlgorithmTracker):
        def flush_state(self):
            writes['tracker'] += 1
            return super().flush_state()

    class ObservedAlgorithm(TrainAlgorithm):
        def on_eval_period(self, *args, **kwargs):
            assert mode != 'terminal', 'terminal recovery must not evaluate again'
            stop = super().on_eval_period(*args, **kwargs)
            extra = args[-1]
            eval_events.append({'epoch': extra['epoch'], 'step': extra['step'],
                                'stall': self._stall, 'best': self._best_test, 'stop': stop})
            return stop

    tracker = ObservedTracker(output / 'assets')
    tracker_before = tracker.jsonl_path.read_bytes()
    if mode == 'terminal':
        data.iter_batches = lambda *args: (_ for _ in ()).throw(AssertionError('terminal recovery consumed a batch'))
    algorithm = ObservedAlgorithm(AlgorithmConfig(mode='train', config=settings))
    summary = algorithm.run(data, model, system, tracker)
    assert all(parameter.is_cuda for parameter in model.module.parameters())
    torch.cuda.synchronize(device)
    rng = capture_rng()
    next_draw = {'python': random.random(), 'numpy': np.random.random(4),
                 'torch': torch.rand(4), 'cuda': torch.rand(4, device=device)}
    final = system.load_checkpoint(str(checkpoint) if mode == 'terminal' else 'latest')
    torch.save({'pid': os.getpid(), 'device': str(device), 'dtype': str(next(model.module.parameters()).dtype),
        'gpu': torch.cuda.get_device_name(device), 'summary': summary, 'initial': initial,
        'module': model.module.state_dict(), 'checkpoint': final, 'data': data.training_state() or data._resume_data,
        'tracker': tracker.state_dict(),
        'writes': writes, 'eval_events': eval_events, 'stall': algorithm._stall, 'best': algorithm._best_test,
        'tracker_file_unchanged': tracker.jsonl_path.read_bytes() == tracker_before,
        'tracker_state_file_exists': tracker.state_path.exists(),
        'ids': list(samples.seen), 'rng': rng, 'next_draw': next_draw}, output / 'finished.pt')
    print(json.dumps({'event': 'cr5.completed', 'pid': os.getpid(), 'device': str(device),
                      'steps': summary['steps'], 'batches': len(samples.seen) // 2}), flush=True)


def _assert_same(left, right):
    import torch
    import numpy as np

    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for actual, expected in zip(left, right):
            _assert_same(actual, expected)
    else:
        assert left == right


def _checkpoint_bytes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def _invoke_worker(root, output, mode, *, checkpoint=None, scenario='step', rebind=False):
    from tests.conftest import _process_tree

    repository = Path(__file__).resolve().parents[5]
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(repository / 'src'), str(repository),
        sysconfig.get_path('purelib'))), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONUTF8': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'TEMP': str(root), 'TMP': str(root), 'TORCH_HOME': str(root / 'torch-cache'),
        'CUDA_CACHE_PATH': str(root / 'cuda-cache'), 'SD_TEST_REBIND': '1' if rebind else '0',
        'SD_TEST_SCENARIO': scenario}
    command = [sys.executable, '-S', '-B', str(Path(__file__).resolve()), mode, str(output)]
    if checkpoint is not None:
        command.append(str(checkpoint))
    result = _process_tree.sh(command, capture_output=True, text=True, encoding='utf-8',
        env=environment, cwd=root, check=False, timeout=60)
    (root / f'{mode}.log').write_text((result.stdout or '') + (result.stderr or ''), encoding='utf-8')
    assert result.cleanup_confirmed and not result.cleanup_errors
    assert result.returncode == (73 if mode == 'interrupted' else 0), result.stdout + result.stderr
    return result.process_id


@pytest.mark.parametrize('rebind', [False, True])
def test_cuda_train_resume_in_new_process_matches_uninterrupted_six_step_run(tmp_path, record_property, rebind):
    import torch

    assert torch.cuda.is_available(), 'CR5 requires real CUDA; no CPU fallback'
    outputs = {name: tmp_path / name for name in ('continuous', 'interrupted', 'resumed')}

    def invoke(mode, checkpoint=None):
        return _invoke_worker(tmp_path, outputs[mode], mode, checkpoint=checkpoint, rebind=rebind)

    control_pid = invoke('continuous')
    interrupted_pid = invoke('interrupted')
    checkpoint = outputs['interrupted'] / 'assets/checkpoints/latest.pt'
    assert checkpoint.is_file() and not (outputs['interrupted'] / 'finished.pt').exists()
    before = _checkpoint_bytes(checkpoint.parent)
    resumed_pid = invoke('resumed', checkpoint)
    assert len({control_pid, interrupted_pid, resumed_pid, os.getpid()}) == 4
    assert before == _checkpoint_bytes(checkpoint.parent)

    def load(name, member='finished.pt'):
        return torch.load(outputs[name] / member, map_location='cpu', weights_only=False)

    control, interrupted, resumed = load('continuous'), load('interrupted', 'interrupted.pt'), load('resumed')
    assert control['pid'] == control_pid and interrupted['pid'] == interrupted_pid and resumed['pid'] == resumed_pid
    assert interrupted['step'] == 2 and interrupted['budget'] == 6 and interrupted['stopped'] is False
    assert interrupted['data']['consumed'] == 4 and interrupted['data']['exhausted'] is False
    assert interrupted['data']['spec']['num_samples'] == (24 if rebind else 32)
    assert control['summary']['steps'] == resumed['summary']['steps'] == 6
    assert len(control['ids']) == 24 and len(interrupted['ids']) == 8 and len(resumed['ids']) == 16
    assert interrupted['ids'] + resumed['ids'] == control['ids']
    assert len(set(control['ids'])) == 24 and control['ids'] != list(range(24))
    assert any(not torch.equal(control['initial'][key], resumed['initial'][key]) for key in control['initial'])
    _assert_same(control['module'], resumed['module'])
    for key in ('model', 'optimizer', 'scheduler'):
        _assert_same(control['checkpoint'][key], resumed['checkpoint'][key])
    assert control['checkpoint']['optimizer']['state']
    assert control['checkpoint']['scheduler']['last_epoch'] == resumed['checkpoint']['scheduler']['last_epoch'] == 6
    def ledger(value):
        return {key: item for key, item in value.items() if key not in ('jsonl_path', 'jsonl_offset')}

    _assert_same(ledger(control['tracker']), ledger(resumed['tracker']))
    _assert_same(ledger(control['checkpoint']['tracker']), ledger(resumed['checkpoint']['tracker']))
    assert ledger(control['tracker'])['step'] == 12
    assert ledger(control['tracker'])['progress'] == {'step': 6, 'epoch': 1}
    assert ledger(control['tracker'])['splits']['train']['Loss']['history']
    assert control['summary']['resume_sources'] is None
    assert resumed['summary']['resume_sources'] == {
        'path': str(checkpoint.resolve()), 'sha256': before['latest.pt']}
    for result in (control, resumed):
        _assert_same(result['module'], result['checkpoint']['model'])
        assert result['data']['consumed'] == 12 and result['data']['exhausted'] is False
        assert result['data']['spec']['num_samples'] == (24 if rebind else 32)
        assert (result['data']['start']['loader'] is None) is rebind
        assert result['checkpoint']['step'] == 6 and result['checkpoint']['training_state']['stopped'] is True
        assert result['device'] == 'cuda:0' and result['dtype'] == 'torch.float32'
    _assert_same(control['data'], resumed['data'])
    _assert_same(control['rng'], resumed['rng'])
    _assert_same(control['next_draw'], resumed['next_draw'])
    record_property('cuda_device', control['gpu'])
    record_property('optimizer_steps', 6)
    record_property('checkpoint_step', 2)
    record_property('data_rebind_steps', rebind)
    record_property('independent_worker_pids', [control_pid, interrupted_pid, resumed_pid])
    record_property('resume_scope', 'native TrainAlgorithm, zero-worker Data, shuffle, stochastic transforms, CUDA dropout')


@pytest.mark.parametrize('scenario', ['epoch', 'early_stop'])
def test_cuda_epoch_resume_preserves_stall_and_terminal_checkpoints_do_no_work(tmp_path, record_property, scenario):
    import torch

    assert torch.cuda.is_available(), 'CR5 requires real CUDA; no CPU fallback'
    outputs = {name: tmp_path / name for name in ('continuous', 'interrupted', 'resumed', 'terminal')}

    def invoke(mode, checkpoint=None):
        return _invoke_worker(tmp_path, outputs[mode], mode, checkpoint=checkpoint, scenario=scenario)

    pids = [invoke('continuous'), invoke('interrupted')]
    checkpoint = outputs['interrupted'] / 'assets/checkpoints/latest.pt'
    original = _checkpoint_bytes(checkpoint.parent)
    pids.append(invoke('resumed', checkpoint))
    assert original == _checkpoint_bytes(checkpoint.parent)
    terminal_checkpoint = outputs['continuous'] / 'assets/checkpoints/latest.pt'
    terminal_original = _checkpoint_bytes(terminal_checkpoint.parent)
    pids.append(invoke('terminal', terminal_checkpoint))
    assert terminal_original == _checkpoint_bytes(terminal_checkpoint.parent)
    assert len({*pids, os.getpid()}) == 5
    results = {name: torch.load(outputs[name] / ('interrupted.pt' if name == 'interrupted' else 'finished.pt'),
                               map_location='cpu', weights_only=False) for name in outputs}
    continuous, interrupted, resumed, terminal = (results[name] for name in outputs)
    assert interrupted['epoch'] == (2 if scenario == 'early_stop' else 1)
    assert interrupted['stall'] == (1 if scenario == 'early_stop' else 0)
    assert interrupted['stopped'] is False and interrupted['data']['exhausted'] is True
    assert continuous['summary']['steps'] == resumed['summary']['steps'] == terminal['summary']['steps'] == 12
    assert continuous['summary']['epochs'] == resumed['summary']['epochs'] == terminal['summary']['epochs'] == 3
    assert interrupted['ids'] + resumed['ids'] == continuous['ids'] and terminal['ids'] == []
    assert interrupted['eval_events'] + resumed['eval_events'] == continuous['eval_events']
    assert len(continuous['eval_events']) == 3 and terminal['eval_events'] == []
    assert continuous['checkpoint']['training_state']['stall'] == (2 if scenario == 'early_stop' else 0)
    assert continuous['stall'] == resumed['stall'] == terminal['stall'] == (2 if scenario == 'early_stop' else 0)
    assert continuous['best'] == resumed['best'] == terminal['best'] == .5
    assert continuous['checkpoint']['training_state']['stopped'] is True
    assert terminal['writes'] == {'checkpoint': 0, 'tracker': 0}
    assert terminal['tracker_file_unchanged'] is True and terminal['tracker_state_file_exists'] is False
    assert not (outputs['terminal'] / 'assets/checkpoints').exists()
    for key in ('model', 'optimizer', 'scheduler', 'training_state'):
        _assert_same(continuous['checkpoint'][key], resumed['checkpoint'][key])
    for key in ('module', 'data', 'rng', 'next_draw'):
        _assert_same(continuous[key], resumed[key])
        _assert_same(continuous[key], terminal[key])
    for key in ('tracker',):
        ledgers = [{name: value for name, value in result[key].items() if name not in ('jsonl_path', 'jsonl_offset')}
                   for result in (continuous, resumed, terminal)]
        _assert_same(ledgers[0], ledgers[1])
        _assert_same(ledgers[0], ledgers[2])
    assert all(result['device'] == 'cuda:0' for result in (continuous, resumed, terminal))
    record_property('cuda_device', continuous['gpu'])
    record_property('independent_worker_pids', pids)
    record_property('recovery_scenario', scenario)
    record_property('controlled_score', .5)
    record_property('optimizer_steps', 12)


def test_cuda_checkpoint_failure_keeps_primary_error_when_tracker_cleanup_also_fails(tmp_path, record_property):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from types import SimpleNamespace
    from spec_mtp.structure.algorithm.config import AlgorithmConfig
    from spec_mtp.structure.algorithm.tracker import AlgorithmTracker
    from spec_mtp.structure.algorithm.train import TrainAlgorithm
    from spec_mtp.structure.data.factory import Data
    from spec_mtp.structure.system.factory import System
    from spec_mtp.structure.system.logger import Logger

    assert torch.cuda.is_available(), 'checkpoint failure validation requires CUDA; no CPU fallback'
    module = torch.nn.Linear(2, 2).cuda()
    dataset = TensorDataset(torch.eye(2), torch.tensor([0, 1]))
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    data = Data(name='write-failure', source='test', loaders={'train': loader}, meta={})

    class FailingSystem(System):
        def save_checkpoint(self, payload, name):
            assert payload['step'] == 1 and payload['model']['weight'].is_cuda
            raise OSError('primary checkpoint failure')

    class FailingTracker(AlgorithmTracker):
        def flush_state(self):
            raise RuntimeError('secondary tracker cleanup failure')

    system = FailingSystem(device='cuda:0', assets_dir=tmp_path / 'assets', logger=Logger(tmp_path / 'assets'), meta={})
    tracker = FailingTracker(tmp_path / 'assets')
    config = AlgorithmConfig(mode='train', config={'num_steps': 2, 'eval_period': 0, 'log_period': 0,
        'checkpoint_period': 1, 'save_best': False, 'resume': False})
    with pytest.raises(OSError, match='primary checkpoint failure') as caught:
        TrainAlgorithm(config).run(data, SimpleNamespace(module=module), system, tracker)
    assert isinstance(getattr(caught.value, 'tracker_cleanup_error', None), RuntimeError)
    assert str(caught.value.tracker_cleanup_error) == 'secondary tracker cleanup failure'
    torch.cuda.synchronize()
    record_property('cuda_device', torch.cuda.get_device_name(0))
    record_property('failure_boundary', 'checkpoint write primary preserved; tracker cleanup secondary recorded')


def test_cuda_epoch_step_cap_terminal_recovery_does_not_consume_another_batch(tmp_path, monkeypatch, record_property):
    import torch
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, TensorDataset
    from spec_mtp.structure.algorithm.config import AlgorithmConfig
    from spec_mtp.structure.algorithm.tracker import AlgorithmTracker
    from spec_mtp.structure.algorithm.train import TrainAlgorithm
    from spec_mtp.structure.data.factory import Data
    from spec_mtp.structure.system.factory import System
    from spec_mtp.structure.system.logger import Logger

    assert torch.cuda.is_available(), 'terminal checkpoint validation requires CUDA; no CPU fallback'
    module = torch.nn.Linear(2, 2).cuda()
    data = Data(name='legacy-cap', source='test', loaders={'train': DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=2)}, meta={})
    source = System(device='cuda:0', assets_dir=tmp_path/'source', logger=Logger(tmp_path/'source'), meta={})
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    bundle = source.save_checkpoint({'model': module.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'step': 3, 'epoch': 1}, 'latest')
    original = _checkpoint_bytes(bundle.parent)
    system = System(device='cuda:0', assets_dir=tmp_path/'target', logger=Logger(tmp_path/'target'), meta={})
    tracker = AlgorithmTracker(tmp_path/'target')
    algorithm = TrainAlgorithm(AlgorithmConfig(mode='train', config={
        'progress_unit': 'epoch', 'num_epochs': 3, 'resume_from': str(bundle), 'eval_period': 0}))

    def forbidden(*args, **kwargs):
        pytest.fail('the step cap must prevent extra training, evaluation and scientific writes')

    monkeypatch.setattr(data, 'iter_batches', forbidden)
    monkeypatch.setattr(algorithm, 'on_eval_period', forbidden)
    monkeypatch.setattr(system, 'save_checkpoint', forbidden)
    monkeypatch.setattr(tracker, 'begin_run', forbidden)
    monkeypatch.setattr(tracker, 'flush_state', forbidden)
    with pytest.warns(RuntimeWarning, match='partial'):
        summary = algorithm.run(data, SimpleNamespace(module=module), system, tracker)
    assert summary['steps'] == 3 and summary['epochs'] == 1
    assert _checkpoint_bytes(bundle.parent) == original
    assert not (tmp_path/'target/checkpoints').exists()
    assert all(parameter.is_cuda for parameter in module.parameters())
    record_property('cuda_device', torch.cuda.get_device_name(0))
    record_property('terminal_reason', 'epoch budget optimizer-step cap; legacy partial scope')


if __name__ == '__main__':
    _worker(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
