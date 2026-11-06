"""Shared fixtures for this repository's FlowRunner and scientific contracts."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests import _process_tree

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'studies' / 'smoke'

# Child interpreters must verify this checkout, not a different globally editable SD.
os.environ['PYTHONPATH'] = os.pathsep.join((
    str(ROOT / 'src'), str(ROOT), str(ROOT / 'studies' / '_shared' / 'structure')))
if str(ROOT / 'studies' / '_shared' / 'structure') not in sys.path:
    sys.path.insert(0, str(ROOT / 'studies' / '_shared' / 'structure'))


def fixed_tiny_model():
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM

    torch.manual_seed(7)
    return HFCausalLM(tiny_config={'vocab_size': 199, 'hidden_size': 64,
                                   'intermediate_size': 128, 'num_layers': 2,
                                   'num_heads': 4, 'num_kv_heads': 2,
                                   'max_position_embeddings': 1024})


def copy_study(tmp_path, patch=None, name='smoke_copy'):
    """Copy declarations, recipe and flow (no runs) and apply a study.yaml patch."""
    dest = tmp_path / name
    shutil.copytree(SMOKE, dest, ignore=shutil.ignore_patterns(
        'runs', 'shared', 'scripts', 'index.json', 'process.json', 'provenance.json',
        'spec_mtp_*.json', 'activity.json', '__pycache__', 'docs'))
    study = yaml.safe_load((dest / 'study.yaml').read_text(encoding='utf-8'))
    study['study'] = name
    study['fixed']['algorithm']['config']['version'] = name
    if patch:
        patch(study)
    (dest / 'study.yaml').write_text(yaml.safe_dump(study, sort_keys=False), encoding='utf-8')
    return dest


def cli(*args, entry='studies._shared.flow.cli', check=False, timeout=90):
    env = {**os.environ, 'PYTHONUTF8': '1', 'MPLBACKEND': 'Agg'}
    proc = _process_tree.sh([sys.executable, '-m', entry, *map(str, args)],
                          capture_output=True, text=True, encoding='utf8', errors='backslashreplace',
                          check=False, env=env, cwd=ROOT, timeout=timeout)
    if check and proc.returncode != 0:
        raise AssertionError(proc.stdout[-4000:] + proc.stderr[-4000:])
    return proc


@pytest.fixture
def study_copy(tmp_path):
    return lambda patch=None, name='smoke_copy': copy_study(tmp_path, patch, name)


@pytest.fixture(scope='session')
def historical_file(tmp_path_factory):
    """Read immutable historical inputs, never import or execute archived source."""
    import hashlib

    commit = 'b8b20b5ada7fd860894114cbf224ce749af430cf'
    root = tmp_path_factory.mktemp('historical-inputs')

    def materialize(relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('historical member must stay inside the archive')
        command = ['git', '-c', 'safe.directory=' + str(ROOT), '-C', str(ROOT)]
        oid = subprocess.check_output(command + ['rev-parse', commit + ':' + relative.as_posix()],
                                      text=True, timeout=20).strip()
        data = subprocess.check_output(command + ['cat-file', 'blob', oid], timeout=20)
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == oid
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)
        assert path.read_bytes() == data, 'historical input was changed: ' + str(relative)
        return path

    return materialize


@pytest.fixture
def saved_decode_config():
    def _config(task='esp_mtp', run_ar=True):
        return {'algorithm': {'config': {'task': task, 'eval': {'run_ar_baseline': run_ar}}}}
    return _config


@pytest.fixture
def saved_decode_assets(saved_decode_config):
    def _assets(tmp_path, rate='1.0', with_sources=True, overall=True, name='a'):
        import csv
        import json
        from studies._shared.structure import sources
        from studies._shared.structure.evidence import digest

        assets = tmp_path / name
        assets.mkdir(parents=True)
        rows = [{'category': 'synthetic', 'exact_match_rate': rate, 'accept_d1': '5'}]
        if overall:
            rows.append({'category': 'overall', 'exact_match_rate': rate, 'accept_d1': '5'})
        with (assets / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        cfg = saved_decode_config()
        saved = sources.manifest()
        saved['config_digest'] = digest(cfg)
        study_dir = assets.parents[2]
        if (study_dir / 'study.yaml').exists():
            from spec_mtp.structure.artifact.provenance import source_files
            from spec_mtp.structure.make.expand import load_study_yaml
            saved['study_files'] = source_files(study_dir, load_study_yaml(study_dir))
        if with_sources:
            (assets / 'spec_mtp_sources.json').write_text(json.dumps(saved), encoding='utf-8')
        inputs = {'settings': {'run': {'seed': 0}, 'eval': {'run_ar_baseline': True, 'device': 'cpu'}, 'spec': {}},
                  'files': {}, 'device': 'cpu', 'prompts': [{'input_ids': [1, 2], 'category': 'synthetic'}]}
        stats = {'new_tokens': 6, 'committed': 5, 'accepted': 5, 'prefill_calls': 1, 'decode_calls': 1,
                 'trace': [{'root_position': 2, 'cache_before': 2, 'accepted_nodes': [1, 2, 3, 4, 5],
                            'stopped': True, 'cache_after': None, 'committed_tokens': [4, 5, 6, 7, 8]}]}
        row = {'category': 'synthetic', **{k: stats[k] for k in ('new_tokens', 'committed', 'accepted', 'decode_calls')},
               'spec_time': 1, 'ar_time': 1, 'exact_match': float(rate) if rate else 1,
               'accept_d1': 5, 'attempt_d1': 5}
        item = {'input_ids': [1, 2], 'output_ids': [1, 2, 3, 4, 5, 6, 7, 8],
                'ar_ids': [1, 2, 3, 4, 5, 6, 7, 8], 'stats': stats}
        import hashlib
        token_hash = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        stats['verification_argmax_audit'] = {'passed': True, 'checked_committed_tokens': 6,
            'input_sha256': token_hash(item['input_ids']), 'output_sha256': token_hash(item['output_ids']),
            'committed': [{'forward': 0, 'row': 1, 'chosen': 3, 'argmax': 3}] +
                [{'forward': 1, 'row': i, 'chosen': token, 'argmax': token} for i, token in enumerate([4,5,6,7,8])]}
        if rate not in ('1.0',''):
            item['ar_ids'][-1] = 99
            row['exact_match'] = 0
        raw = {'task': 'esp_mtp', 'inputs_digest': digest(inputs), 'out': {'rows': [row], 'evidence': [item]}}
        (assets / 'decode_inputs.json').write_text(json.dumps(inputs))
        (assets / 'decode_observations.json').write_text(json.dumps(raw))
        if overall and rate == '1.0':
            from spec_mtp.structure.algorithm.counts import _aggregate
            flat = [{'category': k, **v} for k, v in _aggregate([row]).items()]
            with (assets / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                writer.writeheader()
                writer.writerows(flat)
        return assets
    return _assets


@pytest.fixture
def saved_decode_study(saved_decode_config, saved_decode_assets):
    def _study(tmp_path, runs, axes=None, seeds=(123,)):
        import json
        from types import SimpleNamespace
        from studies._shared.structure import sources
        import yaml

        study = tmp_path / 'study'
        (study / 'runs').mkdir(parents=True)
        (study / 'experiment_config.yaml').write_text(yaml.safe_dump(saved_decode_config()))
        (study / 'study.yaml').write_text(yaml.safe_dump(
            {'study': 'x', 'axes': axes or {'algorithm.config.variant': ['a', 'b']}, 'seeds': list(seeds)}))
        experiments = []
        for variant, status, passed, source_digest in runs:
            run_id = f'run_{variant}'
            experiments.append({'factors': {'algorithm.config.variant': variant},
                                'runs': [{'id': run_id, 'run_dir': run_id, 'seed': 123}]})
            if status is None:
                continue
            (study / 'runs' / run_id).mkdir()
            saved_decode_assets(study / 'runs' / run_id, name='assets')
            (study / 'runs' / run_id / 'config.yaml').write_text(__import__('yaml').safe_dump(saved_decode_config()))
            digest_value = sources.manifest()['digest'] if source_digest == 'd' else source_digest
            (study / 'runs' / run_id / 'result.json').write_text(json.dumps(
                {'status': status, 'spec_mtp': {'passed': passed, 'sources_digest': digest_value}}))
        (study / 'index.json').write_text(json.dumps({'study': 'x', 'experiments': experiments}))
        return SimpleNamespace(scope='study', study_dir=study, state={})
    return _study


@pytest.fixture
def corrupt_saved_argmax():
    def _corrupt_audit(assets, key, value):
        import json

        path = assets / 'decode_observations.json'
        raw = json.loads(path.read_text())
        raw['out']['evidence'][0]['stats']['verification_argmax_audit'][key] = value
        path.write_text(json.dumps(raw))
    return _corrupt_audit


def decode_request(**kwargs):
    """Construct a typed request without importing tensor/model machinery."""
    from spec_mtp.structure.api.decode_api import DecodeRequest, Prompt

    return DecodeRequest(Prompt('p0', (1, 2, 3)),
                             {'num_masks': 1, 'method': 'esp', 'impl': 'efficient',
                              'tree': {'mode': 'static', 'branches': [3], 'pruning': True}},
                             max_new_tokens=7, sources={'science': 'frozen'}, **kwargs)


def decode_fixture(device):
    """Build explicitly placed sessions lazily for injected decode tests."""
    from types import SimpleNamespace

    import torch

    from spec_mtp.structure.algorithm.decode.loop import generate
    from spec_mtp.structure.algorithm.mask import ESPMaskProvider

    actual_device = torch.device(device)
    observed_devices = []
    class FakeSession:
        """Enumerable +1 token model; cache storage is opaque to the decoder."""
        def __init__(self):
            self.embedding_table = torch.eye(8, device=actual_device)
            self.device, self.dtype = actual_device, torch.float32
            self.vocab_size, self.max_positions, self.tied_embeddings = 8, 128, True
            assert self.embedding_table.device == actual_device
            self.calls = 0
            self.kept = []

        def embed(self, ids):
            assert ids.device == actual_device
            return self.embedding_table[ids]

        def min_mask_value(self):
            return torch.finfo(self.dtype).min

        def new_cache(self):
            return SimpleNamespace(length=0)

        def cache_len(self, cache):
            return cache.length

        def crop_cache(self, cache, length):
            cache.length = length

        def keep_cache(self, cache, prefix, indices):
            self.kept.append((prefix, list(indices)))
            cache.length = prefix + len(indices)

        def __call__(self, embeds, positions, *, past_key_values, **_):
            assert embeds.device == positions.device == actual_device
            self.calls += 1
            tokens = (embeds.argmax(-1) + 1) % 8
            logits = torch.nn.functional.one_hot(tokens, 8).float() * 20
            assert logits.device == actual_device
            observed_devices.append(str(logits.device))
            past_key_values.length += embeds.shape[1]
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)


    class MemorySink:
        def __init__(self):
            self.rows = []

        def write(self, row):
            self.rows.append(dict(row))


    class ConstantSession(FakeSession):
        def __call__(self, *args, **kwargs):
            out = super().__call__(*args, **kwargs)
            out.logits.zero_()
            out.logits[..., 4] = 20
            return out


    def run(req, model=None, sink=None, cancel=None):
        model, sink = model or FakeSession(), sink or MemorySink()
        result = generate(req, model, ESPMaskProvider(1, model.embedding_table), sink, cancel)
        return result, model, sink



    return SimpleNamespace(FakeSession=FakeSession, MemorySink=MemorySink,
                           ConstantSession=ConstantSession, run=run,
                           device=actual_device, observed_devices=observed_devices)


@pytest.fixture
def cuda_decode(record_property):
    import torch

    assert torch.cuda.is_available(), 'CUDA correctness must not fall back to CPU'
    device = torch.device('cuda:0')
    bundle = decode_fixture(device)
    record_property('cuda_device', str(device))
    record_property('cuda_name', torch.cuda.get_device_name(device))
    record_property('decode_forward_devices', bundle.observed_devices)
    return bundle


@pytest.fixture
def cuda_tiny_model(cuda_decode, record_property):
    """Create owned tiny HF sessions on the explicit device and close all of them."""
    import torch
    from spec_mtp.structure.model.hf_causal import HFCausalLM

    owned = []
    model_devices = []
    record_property('tiny_model_devices', model_devices)

    def build(config=None, *, seed=0):
        actual = {'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64,
                  'num_layers': 1, 'num_heads': 2, 'num_kv_heads': 1,
                  'max_position_embeddings': 128}
        actual.update(config or {})
        device = cuda_decode.device
        with torch.random.fork_rng(devices=[device.index]), torch.device(device):
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.default_generators[device.index].manual_seed(seed)
            model = HFCausalLM(tiny_config=actual)
        owned.append(model)
        assert model.embedding_table.device == device
        model_devices.append(str(model.embedding_table.device))

        def check_forward(module, args, output):
            assert module.embedding_table.device == args[0].device == output.logits.device == device
            assert args[1].device == device
            cuda_decode.observed_devices.append(str(output.logits.device))

        model.register_forward_hook(check_forward)
        return model

    try:
        yield build
    finally:
        for model in reversed(owned):
            model.close()


def cli_model_args(tmp_path):
    from dataclasses import asdict
    import json
    import hashlib
    from types import SimpleNamespace
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}', encoding='utf-8')
    manifest = tmp_path / 'model-sources.json'
    manifest.write_text(json.dumps({'config.json': hashlib.sha256(b'{}').hexdigest()}), encoding='utf-8')
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(asdict(decode_request())), encoding='utf-8')
    return SimpleNamespace(request=path, model=model, model_sources=manifest, output=tmp_path / 'raw.jsonl')


def cursor_data(kind='random', *, workers=0, seed=7, explicit_loader=True, device=None, size=13):
    from spec_mtp.structure.data.factory import Data
    import torch
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    reads = []

    class Samples:
        def __len__(self):
            return size

        def __getitem__(self, index):
            reads.append(index)
            if device is None:
                return index
            return torch.stack((torch.tensor(index, device=device), torch.rand((), device=device)))

    dataset = Samples()
    sampler_generator = torch.Generator().manual_seed(seed)
    sampler = (SequentialSampler(dataset) if kind == 'sequential' else
               RandomSampler(dataset, generator=sampler_generator))
    loader_generator = (sampler_generator if kind == 'shared' else
                        torch.Generator().manual_seed(seed + 1) if explicit_loader else None)
    options = {} if device is not None else {'collate_fn': lambda values: values}
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, generator=loader_generator,
                        num_workers=workers, **options)
    data = Data(name='cursor-fixture', source='local', loaders={'train': loader},
                meta={'seed': seed, 'batch_size': 3, 'config': {'fixture': 'fixed-id-order'}})
    return data, dataset, reads


def cursor_values_equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(cursor_values_equal(left[key], right[key]) for key in left)
    return left == right


def norm_control_module(monkeypatch):
    from types import SimpleNamespace
    from studies._shared.structure.geometry_4b import observe_initial_bias
    return SimpleNamespace(STUDY=ROOT/'studies/qwen3_4b_bf16_bias_norm_scan',
                           observe_initial_bias=observe_initial_bias)


def saved_diagnostic(message='saved'):
    import json

    return json.dumps({
        'timestamp': '2026-01-01T00:00:00Z', 'level': 'INFO', 'event': 'task.progress',
        'message': message, 'logger': 'spec_mtp', 'app_name': 'spec_mtp',
        'app_version': 'test', 'runtime_id': 'runtime', 'process_name': 'test', 'pid': 1,
    }, ensure_ascii=False) + '\n'
