"""A real CUDA training checkpoint continues identically in a fresh process."""

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import sysconfig


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
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    apply_runtime(912 if mode == 'resumed' else 314, SystemConfig(config={
        'deterministic': True, 'cudnn_benchmark': False,
        'cudnn_allow_tf32': False, 'matmul_allow_tf32': False}))
    output.mkdir(parents=True, exist_ok=False)

    class Samples(Dataset):
        def __init__(self):
            self.seen = []

        def __len__(self):
            return 32

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
        meta={'seed': 53, 'train_size': 32, 'batch_size': 2,
              'config': {'local_fixture': 'cr5', 'num_workers': 0}})
    module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.4), torch.nn.Linear(8, 2))
    initial = {key: value.detach().clone() for key, value in module.state_dict().items()}
    model = Model(name='cr5-dropout', source='test', module=module, meta={})
    logger = Logger(output / 'assets')
    logger.info(f'CR5 mode={mode} device={device} dtype=torch.float32 gpu={torch.cuda.get_device_name(device)}')

    class InterruptedSystem(System):
        def save_checkpoint(self, payload, name):
            path = super().save_checkpoint(payload, name)
            if mode == 'interrupted' and payload['step'] == 2:
                torch.cuda.synchronize(device)
                torch.save({'pid': os.getpid(), 'device': str(device),
                    'ids': list(samples.seen), 'step': payload['step'],
                    'stopped': payload['training_state']['stopped'],
                    'budget': payload['training_state']['contract']['num_steps'],
                    'data': payload['training_state']['data']}, output / 'interrupted.pt')
                # Exit after the real durable checkpoint, without final-stop hooks.
                os._exit(73)
            return path

    system = InterruptedSystem(device=str(device), logger=logger, assets_dir=output / 'assets', meta={})
    settings = {'num_steps': 6, 'step_period': 2, 'eval_period': 0, 'log_period': 0,
        'checkpoint_period': 2, 'save_best': False, 'lr': .05,
        'optimizer': 'SGD', 'momentum': .9, 'scheduler': 'StepLR',
        'step_size': 1, 'gamma': .7, 'resume': False}
    if checkpoint is not None:
        settings['resume_from'] = str(checkpoint)
    tracker = AlgorithmTracker(output / 'assets')
    summary = TrainAlgorithm(AlgorithmConfig(mode='train', config=settings)).run(data, model, system, tracker)
    assert all(parameter.is_cuda for parameter in model.module.parameters())
    torch.cuda.synchronize(device)
    rng = capture_rng()
    next_draw = {'python': random.random(), 'numpy': np.random.random(4),
                 'torch': torch.rand(4), 'cuda': torch.rand(4, device=device)}
    final = system.load_checkpoint('latest')
    torch.save({'pid': os.getpid(), 'device': str(device), 'dtype': str(next(model.module.parameters()).dtype),
        'gpu': torch.cuda.get_device_name(device), 'summary': summary, 'initial': initial,
        'module': model.module.state_dict(), 'checkpoint': final, 'data': data.training_state(),
        'tracker': tracker.state_dict(),
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


def test_cuda_train_resume_in_new_process_matches_uninterrupted_six_step_run(tmp_path, record_property):
    import torch
    from conftest import _process_tree

    assert torch.cuda.is_available(), 'CR5 requires real CUDA; no CPU fallback'
    repository = Path(__file__).resolve().parents[5]
    outputs = {name: tmp_path / name for name in ('continuous', 'interrupted', 'resumed')}
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(repository / 'src'), str(repository),
        sysconfig.get_path('purelib'))), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONUTF8': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'TEMP': str(tmp_path), 'TMP': str(tmp_path), 'TORCH_HOME': str(tmp_path / 'torch-cache'),
        'CUDA_CACHE_PATH': str(tmp_path / 'cuda-cache')}

    def invoke(mode, checkpoint=None):
        command = [sys.executable, '-S', '-B', str(Path(__file__).resolve()), mode, str(outputs[mode])]
        if checkpoint is not None:
            command.append(str(checkpoint))
        result = _process_tree.sh(command, capture_output=True, text=True, encoding='utf-8',
            env=environment, cwd=tmp_path, check=False, timeout=60)
        (tmp_path / f'{mode}.log').write_text((result.stdout or '') + (result.stderr or ''), encoding='utf-8')
        assert result.cleanup_confirmed and not result.cleanup_errors
        assert result.returncode == (73 if mode == 'interrupted' else 0), result.stdout + result.stderr
        return result.process_id

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
        assert result['checkpoint']['step'] == 6 and result['checkpoint']['training_state']['stopped'] is True
        assert result['device'] == 'cuda:0' and result['dtype'] == 'torch.float32'
    _assert_same(control['data'], resumed['data'])
    _assert_same(control['rng'], resumed['rng'])
    _assert_same(control['next_draw'], resumed['next_draw'])
    record_property('cuda_device', control['gpu'])
    record_property('optimizer_steps', 6)
    record_property('checkpoint_step', 2)
    record_property('independent_worker_pids', [control_pid, interrupted_pid, resumed_pid])
    record_property('resume_scope', 'native TrainAlgorithm, zero-worker Data, shuffle, stochastic transforms, CUDA dropout')


if __name__ == '__main__':
    _worker(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
