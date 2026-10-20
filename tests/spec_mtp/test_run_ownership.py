"""Downstream public-entry CPU contract; no model assets or historical Runs."""

import json
import hashlib
import os
import sys
import time

import pytest
import yaml

from conftest import ROOT, cli
from tools._process_tree import ProcessTree


@pytest.fixture
def study(tmp_path):
    (tmp_path / 'study.yaml').write_text(yaml.safe_dump({
        'study': 'ownership-cpu', 'seeds': [1, 2],
        'flow': {'prepare_shared': False, 'study_phases': True}}), encoding='utf-8')
    (tmp_path / 'experiment_config.yaml').write_text(yaml.safe_dump({
        'version': 'ownership-cpu-v1', 'data': {'name': 'Toy', 'source': 'stub'},
        'model': {'name': 'linear'}, 'algorithm': {'mode': 'train', 'num_steps': 1},
        'system': {'device': 'cpu'}}), encoding='utf-8')
    hook = tmp_path / 'flow/execute'
    hook.mkdir(parents=True)
    (hook / '__init__.py').write_text(
        'import time\nfrom pathlib import Path\n'
        'def run(ctx):\n'
        '    root = ctx.layout.root\n'
        '    with (root / "attempts.txt").open("a") as out: out.write("entered\\n")\n'
        '    (root / "entered").touch()\n'
        '    if (ctx.study_dir / "fail").exists(): raise RuntimeError("controlled failure")\n'
        '    if (ctx.study_dir / "interrupt").exists(): raise KeyboardInterrupt("controlled interruption")\n'
        '    deadline = time.monotonic() + 20\n'
        '    while not (root / "release").exists():\n'
        '        if time.monotonic() > deadline: raise TimeoutError("fixture deadline")\n'
        '        time.sleep(.02)\n', encoding='utf-8')
    cli('make', tmp_path, '--num-gpus', '0', '--round', '1', check=True, timeout=30)
    runs = sorted((tmp_path / 'runs').iterdir())
    return tmp_path, runs


@pytest.fixture
def start_owned():
    active = []
    def start(study, run):
        tree = ProcessTree()
        stdout = (run / f'owner-{time.time_ns()}.log').open('wb')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='-1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        active.append((tree, stdout))
        tree.start([sys.executable, '-m', 'spec_mtp', 'run-one', str(study), run.name],
                   cwd=ROOT, env=env, stdout=stdout, stderr=stdout)
        deadline = time.monotonic() + 15
        while not (run / 'entered').exists():
            assert tree.proc.poll() is None, stdout.name
            assert time.monotonic() < deadline, 'owner did not reach the bounded CPU hook'
            time.sleep(.02)
        return tree
    yield start
    for tree, stdout in active:
        try:
            tree.terminate(timeout=5)
            if tree.proc is not None:
                tree.proc.wait(timeout=5)
        finally:
            tree.close()
            stdout.close()


def snapshot(run):
    return {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob('*') if p.is_file()
            and not p.name.startswith('owner-') and p.name != '.rpipe.lock'}


@pytest.mark.parametrize('module', ['rpipe', 'spec_mtp'])
@pytest.mark.parametrize('cached_config', ['foreign', 'missing', 'omitted', 'actual'])
def test_launch_cached_mode_matches_actual_run(study, module, cached_config):
    root, runs = study
    jobs_path = root / 'scripts/jobs.json'
    payload = json.loads(jobs_path.read_text(encoding='utf-8'))
    job = payload['jobs'][0]
    actual_config = root / 'runs' / job['run_id'] / 'config.yaml'
    assert yaml.safe_load(actual_config.read_text(encoding='utf-8'))['algorithm']['mode'] == 'train'
    foreign = root.parent / (root.name + '-foreign.yaml')
    foreign.write_text('algorithm:\n  mode: eval\nsystem:\n  device: cpu\n', encoding='utf-8')
    job['mode'] = 'train' if cached_config == 'actual' else 'eval'
    if cached_config == 'foreign':
        job['config'] = str(foreign)
    elif cached_config == 'missing':
        job['config'] = str(root / 'missing.yaml')
    elif cached_config == 'omitted':
        del job['config']
    jobs_path.write_text(json.dumps(payload), encoding='utf-8')
    before = {str(run): (run / 'config.yaml').read_bytes() for run in runs}
    for run in runs:
        (run / 'release').touch()
    completed = cli('launch', root, '--mode', 'eval', '--include-done', '--num-gpus', '0',
                    '--round', '1', '--console', 'shared', entry=module, timeout=20)
    entered = [run.name for run in runs if (run / 'entered').exists()]
    observation = dict(module=module, cached_config=cached_config, actual_mode='train',
        cached_mode=job['mode'], exit_code=completed.returncode, entered=entered,
        config_bytes_unchanged=all((run / 'config.yaml').read_bytes() == before[str(run)] for run in runs),
        stdout=completed.stdout, stderr=completed.stderr)
    (root / 'cached-mode-observation.json').write_text(json.dumps(observation, indent=2)+'\n', encoding='utf-8')
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not entered, 'eval selection dispatched an actual train Run'
    assert observation['config_bytes_unchanged']


@pytest.mark.parametrize('cached_config', ['foreign', 'missing', 'omitted', 'actual'])
def test_launch_plan_cached_device_matches_actual_run(study, cached_config):
    from rpipe.structure.make.schedule import load_launch_plan

    root, runs = study
    jobs_path = root / 'scripts/jobs.json'
    payload = json.loads(jobs_path.read_text(encoding='utf-8'))
    job = payload['jobs'][0]
    foreign = root.parent / (root.name + '-foreign-device.yaml')
    foreign.write_text('algorithm:\n  mode: train\nsystem:\n  device: cuda\n', encoding='utf-8')
    job['device'] = 'cpu' if cached_config == 'actual' else 'cuda'
    if cached_config == 'foreign':
        job['config'] = str(foreign)
    elif cached_config == 'missing':
        job['config'] = str(root / 'missing.yaml')
    elif cached_config == 'omitted':
        del job['config']
    jobs_path.write_text(json.dumps(payload), encoding='utf-8')
    original = jobs_path.read_bytes()
    plan = load_launch_plan(root, init_gpu=0, num_gpus=0, round_size=1, include_done=True)
    observation = dict(cached_config=cached_config, plan_reused=plan is not None, actual_device='cpu',
                       cached_device=job['device'], jobs_unchanged=jobs_path.read_bytes() == original)
    (root / 'cached-device-observation.json').write_text(json.dumps(observation, indent=2)+'\n', encoding='utf-8')
    assert (plan is not None) == (cached_config == 'actual')
    assert observation['jobs_unchanged']


@pytest.mark.parametrize('omitted', ['mode', 'device', 'both'])
def test_cached_legacy_fields_remain_optional(study, omitted):
    from rpipe.structure.make.schedule import load_launch_plan

    root, runs = study
    path = root / 'scripts/jobs.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    for job in payload['jobs']:
        job.pop('config', None)
        for field in ('mode', 'device') if omitted == 'both' else (omitted,):
            job.pop(field, None)
    path.write_text(json.dumps(payload), encoding='utf-8')
    original = path.read_bytes()
    plan = load_launch_plan(root, init_gpu=0, num_gpus=0, round_size=1, include_done=True)
    assert plan is not None and {job['run_id'] for job in plan['job_list']} == {run.name for run in runs}
    assert path.read_bytes() == original


@pytest.mark.parametrize('field,value', [('mode', None), ('mode', ''), ('device', None), ('device', '')])
def test_cached_legacy_empty_fields_remain_unspecified(study, field, value):
    from rpipe.structure.make.schedule import load_launch_plan

    root, runs = study
    path = root / 'scripts/jobs.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    for job in payload['jobs']:
        job.pop('config', None)
        job[field] = value
        if field == 'mode':
            config_path = root / 'runs' / job['run_id'] / 'config.yaml'
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            config['algorithm']['mode'] = 'eval'
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    path.write_text(json.dumps(payload), encoding='utf-8')
    original = path.read_bytes()
    plan = load_launch_plan(root, init_gpu=0, num_gpus=0, round_size=1, include_done=True)
    (root / 'empty-field-observation.json').write_text(json.dumps(dict(
        field=field,value=value,plan_reused=plan is not None,jobs_unchanged=path.read_bytes()==original),
        indent=2)+'\n',encoding='utf-8')
    assert plan is not None and len(plan['job_list']) == len(runs)
    assert path.read_bytes() == original


@pytest.mark.parametrize('module', ['rpipe', 'spec_mtp'])
@pytest.mark.parametrize('command', ['status', 'process', 'launch'])
@pytest.mark.parametrize('previous', ['[]', '{"metrics":' + '[' * 10000 + '0' + ']' * 10000 + '}'], ids=['array', 'deep'])
def test_public_entries_tolerate_nonobject_result(study, module, command, previous):
    root, runs = study
    for run in runs:
        (run / 'result.json').write_text(previous, encoding='utf-8')
        (run / 'release').touch()
    args = ['--num-gpus', '0', '--round', '1', '--console', 'shared'] if command == 'launch' else []
    completed = cli(command, root, *args, entry=module, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if command == 'launch':
        assert all(json.loads((run / 'result.json').read_text())['status'] == 'succeeded' for run in runs)
    else:
        assert all((run / 'result.json').read_text() == previous for run in runs)
        if command == 'status':
            assert 'pending' in completed.stdout
        else:
            assert json.loads((root / 'process.json').read_text())['complete'] is False


@pytest.mark.parametrize('raw', ['[]', 'null', 'true', '1', '{broken'])
def test_scheduler_corrupt_result_is_not_reusable(tmp_path, raw):
    from rpipe.structure.make.schedule import run_succeeded

    run = tmp_path / 'runs/one'
    run.mkdir(parents=True)
    result = run / 'result.json'
    result.write_text(raw, encoding='utf-8')
    assert run_succeeded(tmp_path, 'one') is False
    assert result.read_text() == raw


@pytest.mark.parametrize('module', ['rpipe', 'spec_mtp'])
@pytest.mark.parametrize('previous', ['[]', '{broken', '{"metrics":' + '[' * 10000 + '0' + ']' * 10000 + '}'], ids=['array', 'broken', 'deep'])
def test_public_run_process_tolerates_corrupt_result_without_rewriting_it(study, module, previous):
    root, runs = study
    run = runs[0]
    result = run / 'result.json'
    result.write_text(previous, encoding='utf-8')
    completed = cli('run-one', root, run.name, '--phases', 'process', entry=module, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert result.read_text() == previous
    body = json.loads((run / 'process.json').read_text())
    assert body['scope'] == 'run' and body['metrics'] == {}


def test_deep_result_strict_decoder_keeps_its_error():
    from rpipe.structure.artifact.result.format import decode_result

    with pytest.raises(RecursionError):
        decode_result('{"metrics":' + '[' * 10000 + '0' + ']' * 10000 + '}')


@pytest.mark.parametrize('module', ['rpipe', 'spec_mtp'])
def test_public_status_tolerates_deep_activity_without_rewriting(study, module):
    root, _ = study
    activity = root / 'activity.json'
    previous = ('[' * 10000 + '0' + ']' * 10000).encode()
    activity.write_bytes(previous)
    completed = cli('status', root, entry=module, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'pending' in completed.stdout
    assert activity.read_bytes() == previous


@pytest.mark.parametrize('module', ['rpipe', 'spec_mtp'])
def test_public_launch_rebuilds_deep_saved_plan(study, module):
    root, runs = study
    (root / 'scripts/jobs.json').write_text('[' * 10000 + '0' + ']' * 10000, encoding='utf-8')
    for run in runs:
        (run / 'release').touch()
    completed = cli('launch', root, '--num-gpus', '0', '--round', '1', '--console', 'shared',
                    entry=module, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((root / 'scripts/jobs.json').read_text())['jobs']
    assert all(json.loads((run / 'result.json').read_text())['status'] == 'succeeded' for run in runs)


@pytest.mark.parametrize('module', ['rpipe', 'spec_mtp'])
@pytest.mark.parametrize('damaged', ['state', 'jsonl'])
def test_public_run_process_skips_deep_history_and_keeps_valid_rows(study, module, damaged):
    root, runs = study
    run = runs[0]
    assets = run / 'assets/tracker'
    assets.mkdir(parents=True)
    deep = '[' * 10000 + '0' + ']' * 10000
    state = assets / 'tracker_state.json'
    log = assets / 'scalars.jsonl'
    row = {'split': 'train', 'name': 'loss', 'mean': 3.0, 'optimizer_step': 1}
    state.write_text(deep if damaged == 'state' else '{}', encoding='utf-8')
    log.write_text(json.dumps(row) + '\n' + (deep + '\n' if damaged == 'jsonl' else '') +
                   json.dumps(dict(row, mean=2.0, optimizer_step=2)) + '\n', encoding='utf-8')
    before = {path: path.read_bytes() for path in (state, log)}
    completed = cli('run-one', root, run.name, '--phases', 'process', entry=module, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    body = json.loads((run / 'process.json').read_text())
    assert body['history'] == {'train': {'loss': [3.0, 2.0]}}
    assert body['history_coordinates']['train']['loss'] == {'unit': 'step', 'x': [1, 2]}
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('case', ['nonobject_job', 'duplicate_done', 'foreign_batch', 'missing_batch',
                                  'invalid_round', 'negative_round', 'infinite_gpu'])
def test_saved_plan_validates_before_filtering_completed_jobs(tmp_path, monkeypatch, case):
    from rpipe.structure.make import schedule

    jobs = [{'run_id': 'one', 'device': 'cpu'}, {'run_id': 'two', 'device': 'cpu'}]
    payload = {'jobs': jobs, 'round': 1, 'num_gpus': 0}
    if case == 'nonobject_job':
        jobs[1] = None
    elif case == 'duplicate_done':
        jobs[1] = dict(jobs[0])
    elif case == 'foreign_batch':
        payload['batches'] = [[jobs[0], {'run_id': 'foreign'}]]
    elif case == 'missing_batch':
        payload['batches'] = [[jobs[0]]]
    elif case == 'invalid_round':
        payload['round'] = 'broken'
    elif case == 'negative_round':
        payload['round'] = -1
    else:
        payload['num_gpus'] = float('inf')
    cache = tmp_path / 'scripts/jobs.json'
    cache.parent.mkdir()
    cache.write_text(json.dumps(payload), encoding='utf-8')
    before = cache.read_bytes()
    monkeypatch.setattr(schedule, 'run_succeeded', lambda *args: True)
    assert schedule.load_launch_plan(tmp_path, init_gpu=0, num_gpus=0, round_size=0) is None
    assert cache.read_bytes() == before
    assert not (tmp_path / 'runs').exists()


def test_saved_plan_keeps_valid_order_and_filters_both_job_lists(tmp_path, monkeypatch):
    from rpipe.structure.make import schedule

    jobs = [{'run_id': name, 'device': 'cpu'} for name in ['pending', 'done', 'last']]
    cache = schedule.write_jobs_json(tmp_path, jobs, round_size=2, init_gpu=0, num_gpus=0,
                                     batches=[[jobs[2], jobs[1]], [jobs[0]]])
    before = cache.read_bytes()
    monkeypatch.setattr(schedule, 'run_succeeded', lambda study, run: run == 'done')
    plan = schedule.load_launch_plan(tmp_path, init_gpu=0, num_gpus=0, round_size=0)
    assert plan['job_list'] == [jobs[0], jobs[2]]
    assert plan['batches'] == [[jobs[2]], [jobs[0]]]
    assert plan['round'] == 2
    assert cache.read_bytes() == before


@pytest.mark.parametrize('entry', ['run-one', 'launch'])
@pytest.mark.parametrize('old_status', ['succeeded', 'failed'])
def test_busy_public_entry_preserves_owner_and_does_not_retry(study, start_owned, entry, old_status):
    root, runs = study
    run = runs[0]
    (run / 'result.json').write_text(json.dumps({
        'status': old_status, 'error': 'controlled previous failure' if old_status == 'failed' else None,
        'control': {}, 'metrics': {}, 'paths': {}}), encoding='utf-8')
    owner = start_owned(root, run)
    before = snapshot(run)
    if entry == 'run-one':
        refused = cli(entry, root, run.name, timeout=20)
        assert refused.returncode == 75
    else:
        # Keep the second declared Run out of this dispatch, with the original plan intact.
        plan = root / 'scripts/jobs.json'
        body = json.loads(plan.read_text())
        body['jobs'] = [j for j in body['jobs'] if j['run_id'] == run.name]
        body.pop('batches', None)
        plan.write_text(json.dumps(body), encoding='utf-8')
        refused = cli('launch', root, '--include-done', '--num-gpus', '0', '--round', '1',
                      '--console', 'shared', timeout=20)
        assert refused.returncode == 1
        assert 'retry ' not in refused.stdout
        assert not (root / 'process.json').exists()
    after = snapshot(run)
    (root / 'ownership-evidence.json').write_text(json.dumps({
        'run_id': run.name, 'owner_pid': owner.proc.pid, 'entry': entry, 'old_status': old_status,
        'refused_exit': refused.returncode, 'refused_stdout': refused.stdout,
        'before_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in before.items()},
        'after_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in after.items()},
    }, indent=2), encoding='utf-8')
    assert after == before
    (run / 'release').touch()
    assert owner.wait(15) == 0
    assert json.loads((run / 'result.json').read_text())['status'] == 'succeeded'


def test_separate_runs_can_execute_while_one_is_locked(study, start_owned):
    root, runs = study
    first = start_owned(root, runs[0])
    second = start_owned(root, runs[1])
    assert first.proc.poll() is None and second.proc.poll() is None
    for run in runs:
        (run / 'release').touch()
    assert first.wait(15) == second.wait(15) == 0


def test_dead_owner_can_be_recovered_without_removing_lock_or_evidence(study, start_owned):
    root, runs = study
    run = runs[0]
    first = start_owned(root, run)
    first.terminate(timeout=5)
    first.proc.wait(timeout=5)
    old = (run / 'execution.json').read_bytes()
    (run / 'release').touch()
    recovered = cli('run-one', root, run.name, check=True, timeout=20)
    assert recovered.returncode == 0
    assert (run / 'execution.json').read_bytes() != old
    assert (run / 'attempts.txt').read_text().splitlines() == ['entered', 'entered']


@pytest.mark.parametrize('failure', ['fail', 'interrupt'])
def test_exception_and_interrupt_release_execution_for_new_attempt(study, failure):
    root, runs = study
    run = runs[0]
    marker = root / failure
    marker.touch()
    assert cli('run-one', root, run.name, timeout=20).returncode != 0
    before = (run / 'execution.json').read_bytes()
    marker.unlink()
    (run / 'release').touch()
    cli('run-one', root, run.name, check=True, timeout=20)
    assert (run / 'execution.json').read_bytes() != before


def test_run_lock_is_not_reentrant_and_releases_after_exception(tmp_path):
    from rpipe.structure.artifact.execution import RunLockedError, run_lock

    with pytest.raises(RuntimeError):
        with run_lock(tmp_path):
            with pytest.raises(RunLockedError):
                with run_lock(tmp_path):
                    pytest.fail('a second executor entered the same Run')
            raise RuntimeError('release this owner')
    with run_lock(tmp_path):
        pass


def test_scheduler_token_is_consumed_once_per_runner_invocation(tmp_path, monkeypatch):
    from rpipe.flow.context import FlowContext
    from rpipe.flow.runner import FlowRunner
    from rpipe.structure.artifact import artifact_layout
    from rpipe.structure.artifact.execution import RUN_ATTEMPT_ENV
    import rpipe.flow.collect as collect

    monkeypatch.setattr(collect, 'run', lambda ctx: None)
    monkeypatch.setenv(RUN_ATTEMPT_ENV, 'single-inherited-attempt')
    contexts = [FlowContext(tmp_path, artifact_layout(tmp_path, name), {}) for name in ['first', 'second']]
    FlowRunner(phases=['collect']).run(contexts[0])
    FlowRunner(phases=['collect']).run(contexts[1])
    tokens = [json.loads((ctx.layout.root / 'execution.json').read_text())['attempt_token'] for ctx in contexts]
    assert tokens[0] == 'single-inherited-attempt' and tokens[1] != tokens[0]
    assert RUN_ATTEMPT_ENV not in os.environ


@pytest.mark.parametrize('code', [0, 7, 75])
def test_launch_never_retries_or_invalidates_unowned_result(tmp_path, monkeypatch, code):
    from rpipe.structure.make import schedule

    run = tmp_path / 'runs/one'
    run.mkdir(parents=True)
    result = run / 'result.json'
    before = b'{"status":"succeeded","control":{},"metrics":{},"paths":{}}'
    result.write_bytes(before)
    calls = []
    class Unowned:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
        def wait(self):
            return code
    monkeypatch.setattr(schedule.subprocess, 'Popen', Unowned)
    refused = []
    assert schedule.launch_jobs(tmp_path, [{'run_id': 'one', 'device': 'cpu'}],
                                console='shared', refused_runs=refused) == [code]
    assert refused == ['one'] and len(calls) == 1
    assert result.read_bytes() == before


@pytest.mark.parametrize('first_code', [0, 7])
def test_only_owned_failure_retries_with_a_new_token(tmp_path, monkeypatch, first_code):
    from rpipe.structure.artifact.execution import RUN_ATTEMPT_ENV, run_lock, start_execution
    from rpipe.structure.artifact.result import write_result
    from rpipe.structure.make import schedule

    run = tmp_path / 'runs/one'
    calls = []
    class Owned:
        def __init__(self, *args, **kwargs):
            token = kwargs['env'][RUN_ATTEMPT_ENV]
            calls.append(token)
            self.code = first_code if len(calls) == 1 else 0
            with run_lock(run):
                start_execution(run, token)
                write_result(run / 'result.json', {
                    'status': 'failed' if len(calls) == 1 and first_code == 0 else 'succeeded',
                    'error': 'controlled first-attempt failure' if len(calls) == 1 else None,
                    'control': {}, 'metrics': {}, 'paths': {}})
        def wait(self):
            return self.code
    monkeypatch.setattr(schedule.subprocess, 'Popen', Owned)
    refused = []
    assert schedule.launch_jobs(tmp_path, [{'run_id': 'one', 'device': 'cpu'}],
                                console='shared', refused_runs=refused) == [first_code, 0]
    assert not refused and len(calls) == len(set(calls)) == 2
    assert json.loads((run / 'result.json').read_text())['status'] == 'succeeded'


@pytest.mark.parametrize('jobs,batches', [
    ([{'run_id': '../escape'}], None),
    ([{'run_id': 'one'}, {'run_id': 'ONE'}], None),
    ([{'run_id': 'one'}], [[{'run_id': 'other'}]]),
])
def test_invalid_launch_plan_refuses_before_creating_run_locks(tmp_path, jobs, batches):
    from rpipe.structure.make.schedule import launch_jobs

    if os.name != 'nt' and len(jobs) > 1:
        jobs[1]['run_id'] = 'one'  # Path aliases follow the host filesystem.
    with pytest.raises(ValueError):
        launch_jobs(tmp_path, jobs, batches=batches, console='shared')
    assert not (tmp_path / 'runs').exists()


def test_generated_bash_payload_refuses_before_process(tmp_path, monkeypatch):
    from rpipe.structure.make import schedule
    import rpipe.flow.process as process

    def refuse(*args, **kwargs):
        kwargs['refused_runs'].append('one')
    monkeypatch.setattr(schedule, 'launch_jobs', refuse)
    monkeypatch.setattr(process, 'run_study', lambda *args: pytest.fail('refusal published a Study result'))
    script, = schedule.render_bash(tmp_path, [{'run_id': 'one', 'device': 'cpu'}],
                                  python_exe=sys.executable, round_size=1, split_round=1)
    payload = script.split("<<'RPIPE_LAUNCH'\n", 1)[1].split('\nRPIPE_LAUNCH', 1)[0]
    with pytest.raises(SystemExit) as caught:
        exec(payload, {})
    assert caught.value.code == 1


def freeze_study(root, study_phases):
    declaration = yaml.safe_load((root / 'study.yaml').read_text())
    declaration['freeze'] = True
    declaration['flow']['study_phases'] = study_phases
    (root / 'study.yaml').write_text(yaml.safe_dump(declaration), encoding='utf-8')
    cli('make', root, '--num-gpus', '0', '--round', '1', check=True, timeout=30)


@pytest.mark.parametrize('study_phases', [False, True])
def test_scheduler_token_freeze_refusal_preserves_previous_run(study, monkeypatch, study_phases):
    from rpipe.flow.cli import main
    from rpipe.structure.artifact.execution import RUN_ATTEMPT_ENV, start_execution
    from rpipe.structure.artifact.provenance import ProvenanceChangedError

    root, runs = study
    run = runs[0]
    freeze_study(root, study_phases)
    (run / 'result.json').write_text(json.dumps({
        'status': 'succeeded', 'control': {}, 'metrics': {}, 'paths': {}}), encoding='utf-8')
    start_execution(run, 'previous-admitted-owner')
    (run / 'assets/logs').mkdir(parents=True, exist_ok=True)
    (run / 'assets/logs/run.log').write_bytes(b'previous run log\n')
    before = snapshot(run)
    hook = root / 'flow/execute/__init__.py'
    hook.write_text(hook.read_text() + '# deliberate frozen source drift\n', encoding='utf-8')
    monkeypatch.setenv(RUN_ATTEMPT_ENV, 'new-refused-attempt')
    with pytest.raises(ProvenanceChangedError):
        main(['run-one', str(root), run.name])
    assert snapshot(run) == before


def test_admitted_scheduler_attempt_invalidates_success_before_business(study, monkeypatch):
    from rpipe.flow.context import FlowContext
    from rpipe.flow.runner import FlowRunner
    from rpipe.structure.artifact import artifact_layout
    from rpipe.structure.artifact.execution import RUN_ATTEMPT_ENV
    import rpipe.flow.collect as collect

    root, runs = study
    run = runs[0]
    (run / 'result.json').write_text(json.dumps({
        'status': 'succeeded', 'control': {}, 'metrics': {}, 'paths': {}}), encoding='utf-8')
    observed = []
    monkeypatch.setattr(collect, 'run', lambda ctx: observed.append(json.loads(ctx.layout.result_path.read_text())))
    monkeypatch.setenv(RUN_ATTEMPT_ENV, 'new-admitted-attempt')
    FlowRunner(phases=['collect']).run(FlowContext(root, artifact_layout(root, run.name), {}))
    assert len(observed) == 1 and observed[0]['status'] == 'failed'
    assert 'rerun started' in observed[0]['error']


def test_public_launch_freeze_race_is_unowned_and_never_retried(study, monkeypatch, capsys):
    from types import SimpleNamespace
    from rpipe.flow.cli import main
    from rpipe.structure.artifact.execution import start_execution
    from rpipe.structure.make import schedule

    root, runs = study
    run = runs[0]
    freeze_study(root, True)
    (run / 'result.json').write_text(json.dumps({
        'status': 'succeeded', 'control': {}, 'metrics': {}, 'paths': {}}), encoding='utf-8')
    start_execution(run, 'previous-admitted-owner')
    plan = root / 'scripts/jobs.json'
    body = json.loads(plan.read_text())
    body['jobs'] = [job for job in body['jobs'] if job['run_id'] == run.name]
    body.pop('batches', None)
    plan.write_text(json.dumps(body), encoding='utf-8')
    before = snapshot(run)
    children = []
    class BoundedChild:
        def __init__(self, command, **kwargs):
            hook = root / 'flow/execute/__init__.py'
            hook.write_text(hook.read_text() + '# source changed after parent admission\n', encoding='utf-8')
            self.tree = ProcessTree()
            children.append(self)
            self.deadline = time.monotonic() + 20
            self.tree.start(command, **kwargs)
        def wait(self):
            return self.tree.wait(max(0, self.deadline - time.monotonic()))
    monkeypatch.setattr(schedule, 'subprocess', SimpleNamespace(Popen=BoundedChild))
    try:
        assert main(['launch', str(root), '--include-done', '--num-gpus', '0', '--round', '1',
                     '--console', 'shared']) == 1
        assert len(children) == 1
        assert 'retry ' not in capsys.readouterr().out
        assert snapshot(run) == before
        assert not (root / 'process.json').exists()
    finally:
        for child in children:
            try:
                child.tree.terminate(timeout=5)
                child.tree.proc.wait(timeout=5)
            finally:
                child.tree.close()


@pytest.mark.parametrize('entry', ['run-one', 'runner'])
@pytest.mark.parametrize('previous', [b'[]\n', b'{broken JSON\n'])
def test_token_admission_with_corrupt_previous_result_reaches_owned_phase(study, monkeypatch, entry, previous):
    from rpipe.flow.cli import main
    from rpipe.flow.context import FlowContext
    from rpipe.flow.runner import FlowRunner
    from rpipe.structure.artifact import artifact_layout
    from rpipe.structure.artifact.execution import RUN_ATTEMPT_ENV
    import rpipe.flow.collect as collect

    root, runs = study
    run = runs[0]
    (run / 'result.json').write_bytes(previous)
    entered = []
    monkeypatch.setattr(collect, 'run', lambda ctx: entered.append(ctx.layout.root))
    monkeypatch.setenv(RUN_ATTEMPT_ENV, 'new-attempt-after-corrupt-result')
    if entry == 'run-one':
        assert main(['run-one', str(root), run.name, '--phases', 'collect']) == 0
    else:
        FlowRunner(phases=['collect']).run(FlowContext(root, artifact_layout(root, run.name), {}))
    assert entered == [run]
    assert (run / 'result.json').read_bytes() == previous


@pytest.mark.parametrize('entry', ['run-one', 'runner'])
def test_token_admission_preserves_result_read_permission_error(study, monkeypatch, entry):
    from rpipe.flow.cli import main
    from rpipe.flow.context import FlowContext
    from rpipe.structure.artifact import artifact_layout
    from rpipe.structure.artifact.execution import RUN_ATTEMPT_ENV
    import rpipe.flow.runner as runner
    import rpipe.flow.collect as collect

    root, runs = study
    run = runs[0]
    result = run / 'result.json'
    before = b'{"status":"succeeded","control":{},"metrics":{},"paths":{}}'
    result.write_bytes(before)
    primary = PermissionError('controlled previous-result read refusal')
    entered = []
    def refused(path):
        raise primary
    monkeypatch.setattr(runner, 'load_result', refused)
    monkeypatch.setattr(collect, 'run', lambda ctx: entered.append(ctx))
    monkeypatch.setenv(RUN_ATTEMPT_ENV, 'attempt-with-refused-result-read')
    with pytest.raises(PermissionError) as caught:
        if entry == 'run-one':
            main(['run-one', str(root), run.name, '--phases', 'collect'])
        else:
            runner.FlowRunner(phases=['collect']).run(FlowContext(root, artifact_layout(root, run.name), {}))
    assert caught.value is primary
    assert not entered and result.read_bytes() == before


@pytest.mark.parametrize('module', ['rpipe', 'spec_mtp'])
@pytest.mark.parametrize('busy', [True, False], ids=['busy-eval', 'idle-eval'])
@pytest.mark.parametrize('parent_state', ['failed', 'newer'])
def test_launch_refuses_busy_unselected_stale_eval(tmp_path, monkeypatch, module, busy, parent_state):
    from contextlib import nullcontext
    from rpipe.structure.artifact import load_result, write_result
    from rpipe.structure.artifact.execution import run_lock, start_execution
    from rpipe.structure.make.schedule import sibling_train_dependency

    for key, value in {'PYTHONPATH': str(ROOT / 'src'), 'CUDA_VISIBLE_DEVICES': '-1',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'safe.directory',
                       'GIT_CONFIG_VALUE_0': str(ROOT)}.items():
        monkeypatch.setenv(key, value)
    (tmp_path / 'recipe.py').write_text(
        'from rpipe.structure.algorithm.base import Algorithm\n'
        'from rpipe.structure.algorithm.factory import AlgorithmRegistry\n'
        'from rpipe.structure.model.factory import Model, ModelRegistry\n'
        'class Once(Algorithm):\n'
        '    def run(self, data, model, system, tracker):\n'
        '        with (system.assets_dir / "attempts.txt").open("a") as out: out.write("attempt\\n")\n'
        '        (system.assets_dir / "exclusive").mkdir()\n'
        '        return {}\n'
        'def register(ctx):\n'
        '    AlgorithmRegistry.register("inference", "ownership_fixture", Once)\n'
        '    ModelRegistry.register("none", "ownership_fixture", lambda config, root, **kwargs:\n'
        '        Model(name="none", source="ownership_fixture", module=None, meta={}))\n', encoding='utf-8')
    (tmp_path / 'study.yaml').write_text(yaml.safe_dump({
        'study': 'stale-selection-cpu', 'seeds': [0], 'freeze': True, 'recipe': 'recipe.py',
        'flow': {'prepare_shared': False},
        'axes': {'algorithm.mode': ['train', 'eval', 'inference']}}), encoding='utf-8')
    (tmp_path / 'experiment_config.yaml').write_text(yaml.safe_dump({
        'version': 'stale-selection-cpu-v1', 'data': {'name': 'Toy', 'source': 'stub'},
        'model': {'name': 'none', 'source': 'ownership_fixture'},
        'algorithm': {'mode': 'inference', 'source': 'ownership_fixture'},
        'system': {'device': 'cpu'}}), encoding='utf-8')
    cli('make', tmp_path, '--num-gpus', '0', '--round', '1', entry=module, check=True, timeout=30)
    index = json.loads((tmp_path / 'index.json').read_text())
    configs = [tmp_path / run['config'] for exp in index['experiments'] for run in exp['runs']]
    by_mode = {yaml.safe_load(path.read_text())['algorithm']['mode']: path.parent for path in configs}
    train, evaluation, selected = (by_mode[mode] for mode in ('train', 'eval', 'inference'))
    assert len(configs) == 3 and sibling_train_dependency(tmp_path, evaluation.name) == train.name
    provenance = json.loads((tmp_path / 'provenance.json').read_text())['digest']
    for run in (train, evaluation):
        with run_lock(run):
            start_execution(run, 'prior-' + run.name)
            write_result(run / 'result.json', {'status': 'succeeded', 'control': {}, 'paths': {},
                         'metrics': {'accuracy': 42}, 'provenance': provenance})
    if parent_state == 'failed':
        with run_lock(train):
            write_result(train / 'result.json', {'status': 'failed', 'error': 'prior train failed'})
    os.utime(evaluation / 'result.json', ns=(1_000_000_000, 1_000_000_000))
    os.utime(train / 'result.json', ns=(2_000_000_000, 2_000_000_000))
    protected = snapshot(evaluation)
    process = tmp_path / 'process.json'
    prior = b'{"retained": "prior Study process"}\n'
    process.write_bytes(prior)
    inputs = {path: path.read_bytes() for path in [tmp_path / name for name in
              ('study.yaml', 'experiment_config.yaml', 'recipe.py', 'index.json', 'provenance.json')] + configs}
    with run_lock(evaluation) if busy else nullcontext():
        completed = cli('launch', tmp_path, '--mode', 'inference', '--num-gpus', '0', '--round', '1',
                        '--console', 'shared', entry=module, timeout=30)
    (tmp_path / 'launch.stdout').write_text(completed.stdout, encoding='utf-8')
    (tmp_path / 'launch.stderr').write_text(completed.stderr, encoding='utf-8')
    assert load_result(selected / 'result.json')['status'] == 'succeeded', completed.stdout + completed.stderr
    assert (selected / 'assets/attempts.txt').read_text().splitlines() == ['attempt']
    assert {path: path.read_bytes() for path in inputs} == inputs
    assert (evaluation / 'execution.json').read_bytes() == protected['execution.json']
    assert not (evaluation / 'assets/attempts.txt').exists()
    if busy:
        assert completed.returncode == 1, completed.stdout + completed.stderr
        assert process.read_bytes() == prior
        assert snapshot(evaluation) == protected
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert process.read_bytes() != prior
        assert load_result(evaluation / 'result.json')['status'] == 'failed'
        assert load_result(evaluation / 'result.json')['metrics'] == {'accuracy': 42}
