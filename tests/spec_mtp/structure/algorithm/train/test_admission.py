"""Native training rejects invalid snapshots before live scientific mutations."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from spec_mtp.structure.algorithm.config import AlgorithmConfig
from spec_mtp.structure.algorithm.resume import admit_training_resume, capture_rng
from spec_mtp.structure.algorithm.tracker import AlgorithmTracker
from spec_mtp.structure.algorithm.train import TrainAlgorithm, _build_payload, _training_contract
from spec_mtp.structure.algorithm.progress import resolve_budget


@pytest.mark.parametrize('damage', ['contract', 'version', 'python_rng', 'torch_rng', 'data_capability',
    'data_state', 'model_keys', 'model_shape', 'model_dtype', 'optimizer', 'optimizer_groups', 'optimizer_missing_key',
    'tracker', 'missing_tracker', 'missing_step', 'missing_epoch', 'fractional_step',
    'scheduler_missing', 'scheduler_schema', 'scheduler_value', 'missing_stopped', 'missing_stall', 'missing_percent',
    'epoch_exceeds_budget', 'tracker_progress'])
def test_training_admission_precedes_all_live_state_changes(tmp_path, monkeypatch, damage):
    config = AlgorithmConfig.from_mapping({'mode': 'train', 'config': {'num_steps': 4, 'lr': .05,
        'scheduler': 'StepLR', 'step_size': 1, 'gamma': .5}})
    if damage == 'epoch_exceeds_budget':
        config.config.update(progress_unit='epoch', num_epochs=2)
    algo = TrainAlgorithm(config)
    model = SimpleNamespace(module=torch.nn.Linear(2, 2))
    tracker = AlgorithmTracker(tmp_path / 'assets')
    optimizer = torch.optim.SGD(model.module.parameters(), lr=.05)
    scheduler = algo.make_scheduler(optimizer, 4)
    data = SimpleNamespace(meta={}, steps_per_epoch=lambda: 2)

    def admit(state, **kwargs):
        if state != {'valid': True}:
            raise ValueError('invalid fixture data state')

    data.validate_training_state = admit
    payload = _build_payload(model.module, optimizer, scheduler, {'step': 0, 'epoch': 0}, tracker)
    payload['training_state'] = {'version': 1, 'data': {'valid': True}, 'rng': capture_rng(),
        'stall': 0, 'stopped': False, 'already_percent': [],
        'contract': _training_contract(config, resolve_budget(config, steps_per_epoch=2),
                                       torch.device('cpu'), model, tracker)}
    payload = deepcopy(payload)
    checkpoint = tmp_path / 'admitted.pt'
    checkpoint.write_bytes(b'stub loader checkpoint bytes')
    original_sources = {}
    algo.load_resume(SimpleNamespace(checkpoint_path=lambda reference: checkpoint,
                                    load_checkpoint=lambda reference: payload), original_sources)
    admit_training_resume(payload, model=model, tracker=tracker, data=data,
                          contract=payload['training_state']['contract'], num_steps=4, step_period=1,
                          resume_sources=original_sources['resume_sources'])
    if damage == 'contract':
        payload['training_state']['contract']['num_steps'] = 99
    elif damage == 'version':
        payload['training_state']['version'] = 99
    elif damage == 'python_rng':
        payload['training_state']['rng']['python'] = ('invalid',)
    elif damage == 'torch_rng':
        payload['training_state']['rng']['torch'] = torch.tensor([1], dtype=torch.uint8)
    elif damage == 'data_capability':
        data.validate_training_state = None
    elif damage == 'data_state':
        payload['training_state']['data']['valid'] = False
    elif damage == 'model_keys':
        payload['model'].pop('weight')
    elif damage == 'model_shape':
        payload['model']['weight'] = torch.ones(3, 2)
    elif damage == 'model_dtype':
        payload['model']['weight'] = payload['model']['weight'].double()
    elif damage == 'optimizer':
        payload['optimizer']['param_groups'][0]['params'] = []
    elif damage == 'optimizer_groups':
        group = payload['optimizer']['param_groups'][0]
        group['params'] = [0]
        payload['optimizer']['param_groups'].append({**group, 'params': [1]})
    elif damage == 'optimizer_missing_key':
        payload['optimizer']['param_groups'][0].pop('momentum')
    elif damage == 'tracker':
        payload['tracker']['splits'] = 'invalid'
    elif damage in ('missing_tracker', 'missing_step', 'missing_epoch'):
        payload.pop(damage.removeprefix('missing_'))
    elif damage == 'scheduler_missing':
        payload['scheduler'] = None
    elif damage == 'scheduler_schema':
        payload['scheduler'].pop('last_epoch')
    elif damage == 'scheduler_value':
        payload['scheduler']['last_epoch'] = 'invalid'
    elif damage in ('missing_stopped', 'missing_stall', 'missing_percent'):
        payload['training_state'].pop({'missing_stopped': 'stopped', 'missing_stall': 'stall',
                                        'missing_percent': 'already_percent'}[damage])
    elif damage == 'epoch_exceeds_budget':
        payload['epoch'] = 3
    elif damage == 'tracker_progress':
        payload['tracker']['progress'] = {'step': 1, 'epoch': 0}
    else:
        payload['step'] = 1.5
    weights = deepcopy(model.module.state_dict())
    state = deepcopy(tracker.state_dict())
    rng = capture_rng()
    files = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail('live scientific state changed before checkpoint admission')

    logger = SimpleNamespace(info=forbidden, warning=forbidden, load_state_dict=forbidden)
    system = SimpleNamespace(device='cpu', logger=logger, load_checkpoint=lambda reference: payload,
                             checkpoint_path=lambda reference: checkpoint, place_module=forbidden)
    monkeypatch.setattr(model.module, 'load_state_dict', forbidden)
    monkeypatch.setattr(model.module, 'train', forbidden)
    monkeypatch.setattr(tracker, 'load_state_dict', forbidden)
    monkeypatch.setattr(tracker, 'begin_run', forbidden)
    monkeypatch.setattr(algo, 'make_optimizer', forbidden)
    data.iter_batches = data.restore_training_state = data.rebind_train_steps = forbidden
    with pytest.raises(ValueError):
        algo.run(data, model, system, tracker)
    assert all(torch.equal(value, weights[key]) for key, value in model.module.state_dict().items())
    assert tracker.state_dict() == state
    after = capture_rng()
    assert rng['python'] == after['python'] and torch.equal(rng['torch'], after['torch'])
    assert all((a == b).all() if hasattr(a, 'shape') else a == b for a, b in zip(rng['numpy'], after['numpy']))
    assert files == {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}


@pytest.mark.parametrize('explicit', [False, True])
def test_missing_explicit_resume_rejects_while_default_latest_remains_optional(tmp_path, explicit):
    values = {'resume_from': str(tmp_path / 'missing.pt')} if explicit else {}
    algo = TrainAlgorithm(AlgorithmConfig.from_mapping({'mode': 'train', 'config': values}))
    system = SimpleNamespace(load_checkpoint=lambda reference: None)
    if explicit:
        with pytest.raises(FileNotFoundError):
            algo.load_resume(system, {})
    else:
        assert algo.load_resume(system, {}) is None
    assert list(tmp_path.iterdir()) == []


def test_legacy_checkpoint_application_retains_partial_resume_warning(tmp_path):
    algo = TrainAlgorithm(AlgorithmConfig(mode='train'))
    model = SimpleNamespace(module=torch.nn.Linear(2, 2))
    tracker = AlgorithmTracker(tmp_path / 'assets')
    payload = {'model': deepcopy(model.module.state_dict()), 'epoch': 1, 'step': 1}
    system = SimpleNamespace(logger=None)
    with pytest.warns(RuntimeWarning, match='partial'):
        algo.apply_resume(payload, model, system, tracker, {})


def test_checkpoint_read_sha_binds_original_bytes_and_rejects_drift(tmp_path):
    import hashlib

    path = tmp_path / 'checkpoint.pt'
    path.write_bytes(b'original checkpoint bytes')
    algo = TrainAlgorithm(AlgorithmConfig(mode='train'))
    system = SimpleNamespace(checkpoint_path=lambda reference: path, load_checkpoint=lambda reference: {})
    extra = {}
    assert algo.load_resume(system, extra) == {}
    assert extra['resume_sources'] == {'path': str(path.resolve()),
        'sha256': hashlib.sha256(b'original checkpoint bytes').hexdigest()}

    def drift(reference):
        path.write_bytes(b'changed during read')
        return {}

    system.load_checkpoint = drift
    with pytest.raises(ValueError, match='checkpoint bytes changed'):
        algo.load_resume(system, {})


def test_native_recovery_requires_a_committed_sha_bound_bundle(tmp_path):
    from spec_mtp.structure.algorithm.resume import admit_training_resume
    from spec_mtp.structure.system.factory import System
    from spec_mtp.structure.system.logger import Logger

    config = AlgorithmConfig.from_mapping({'mode': 'train', 'config': {'num_steps': 4, 'resume': 'bound'}})
    algo = TrainAlgorithm(config)
    model = SimpleNamespace(module=torch.nn.Linear(2, 2))
    tracker = AlgorithmTracker(tmp_path / 'assets')
    system = System(device='cpu', assets_dir=tmp_path / 'assets', logger=Logger(tmp_path / 'assets'), meta={})
    optimizer = torch.optim.SGD(model.module.parameters(), lr=.1)
    payload = _build_payload(model.module, optimizer, None, {'step': 0, 'epoch': 0}, tracker)
    contract = _training_contract(config, resolve_budget(config, steps_per_epoch=2), torch.device('cpu'), model, tracker)
    payload['training_state'] = {'version': 1, 'data': {}, 'rng': capture_rng(), 'contract': contract,
                                 'stall': 0, 'stopped': False, 'already_percent': []}
    bundle = system.save_checkpoint(payload, 'bound')
    bundle.unlink()
    assert (bundle.with_suffix('') / 'meta.pt').is_file()
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    extra = {}
    restored = algo.load_resume(system, extra)
    assert restored['training_state']['data'] == {} and not extra.get('resume_sources')
    with pytest.raises(ValueError, match='committed checkpoint bundle'):
        admit_training_resume(restored, model=model, tracker=tracker, data=None, contract=contract,
                              num_steps=4, step_period=1, resume_sources=extra.get('resume_sources'))
    assert before == {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
