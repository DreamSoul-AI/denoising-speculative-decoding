"""Local diagnostic binding/load checks; no scientific/default registration."""

import json
import os
from pathlib import Path
from types import SimpleNamespace


def load(out, plan, tool):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import yaml
    from diagnose_output_windows import file_hash
    from rpipe.structure.system.config import SystemConfig
    from rpipe.structure.system.runtime import apply_runtime
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.sources import manifest
    from studies._shared.structure.config import _convert
    from spec_mtp.model import HFCausalLM
    root = Path(__file__).resolve().parents[2]
    study = root / 'studies/output_space_extension'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    paths = {'plan': plan, 'tool': tool, 'common': Path(__file__), 'native': study / 'docs/execution_v1.json',
        'previous_report': root / '.tmp/q178-numerical-20261014-v1/diagnosis.json',
        'previous_tensor': root / '.tmp/q178-numerical-20261014-v1/selected_tensors.pt',
        'snapshot': study / 'docs/modelscope_snapshot.json'}
    for name in ('diagnose_output_windows.py', 'diagnose_proxy_transfer.py', 'diagnose_q178_numerics.py'):
        paths[name] = root / 'tools' / name
    native, previous = read(paths['native']), read(paths['previous_report'])
    rows = {r['case']: r for r in native['runs']}
    folder = study / 'runs' / rows['output_order1']['run_id']
    for name in ('config.yaml', 'assets/decode_inputs.json', 'assets/decode_observations.json'):
        paths[name] = folder / name
    binding = {k: file_hash(p) for k, p in paths.items()}
    assert previous['quality_passed'] and binding['previous_tensor'] == previous['raw_tensor_sha256']
    assert binding['previous_tensor'] == '2f71c78210b1609a889b0d3cfec223fd818dd74c625d6eed32540d91008d86a0'
    assert binding['assets/decode_observations.json'] == rows['output_order1']['raw_sha256']
    assert manifest()['digest'] == native['source_digest'] == '670c744e0535dcad'
    old = torch.load(paths['previous_tensor'], map_location='cpu', weights_only=True)
    snapshot = read(paths['snapshot'])
    cache = study / 'shared/model/modelscope/Qwen/Qwen3-0.6B'
    assert all(file_hash(cache / n) == m['sha256'] for n, m in snapshot['files'].items())
    torch.set_num_threads(1)
    apply_runtime(126, SystemConfig.from_mapping({'device': 'cuda', 'deterministic': True,
        'cudnn_deterministic': True, 'cudnn_benchmark': False, 'matmul_allow_tf32': False, 'cudnn_allow_tf32': False}))
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=False)
    model = HFCausalLM(name_or_path=str(cache), local_files_only=True,
                      torch_dtype='float32', attn_implementation='eager').to('cuda')
    assert model_digest(model) == native['model_digest']
    assert runtime() == previous['runtime'] and not torch.is_deterministic_algorithms_warn_only_enabled()
    ar = rows['AR']['raw']['evidence'][3]
    reference = rows['output_order1']['raw']['evidence'][3]
    assert reference['output_ids'] == ar['output_ids'] and len(ar['output_ids']) == 112
    inputs = read(paths['assets/decode_inputs.json'])
    assert inputs['prompts'][3]['input_ids'] == ar['input_ids']
    cfg = yaml.safe_load(paths['config.yaml'].read_text(encoding='utf-8'))
    targets = sorted({int(k.rsplit(':', 1)[1]) for k in old
                      if k.startswith(('baseline:output_order1:', 'canonical_one_from_zero:')) and k.rsplit(':', 1)[1].isdigit()})
    call = next(i + 1 for i, t in enumerate(reference['stats']['trace']) if t['root_position'] == 85)
    c = SimpleNamespace(root=root, study=study, out=out, paths=paths, binding=binding, native=native,
        previous=previous, old=old, model=model, snapshot=snapshot, cache=cache, targets=targets, call=call,
        ar=ar, reference=reference, states=old['canonical_states']['output_order1'])
    c.prompt = torch.tensor([ar['input_ids']], device=model.device)
    c.keywords = dict(prompt=c.prompt, spec=_convert(rows['output_order1']['spec']),
        eval_cfg=_convert(cfg['algorithm']['config']['eval']), eos=inputs['eos_ids'],
        reference=reference, original_masks=old['baseline:output_order1:85']['current_masks'],
        root=85, targets=targets, placebo=False, seed=126)
    return c


def verify(c):
    from diagnose_output_windows import file_hash
    from studies._shared.structure.evidence import model_digest, runtime
    from studies._shared.structure.sources import manifest
    assert all(file_hash(p) == c.binding[k] for k, p in c.paths.items())
    assert all(file_hash(c.cache / n) == m['sha256'] for n, m in c.snapshot['files'].items())
    assert model_digest(c.model) == c.native['model_digest'] and runtime() == c.previous['runtime']
    assert manifest()['digest'] == c.native['source_digest']


def compare_control(c, samples, replay, graft):
    from diagnose_q178_numerics import tensor_equal
    label = 'canonical_one_from_zero' if graft else 'canonical_one_self'
    prior = next(r['replay'] for r in c.previous['cases'] if r['label'] == label)
    assert replay['trace'] == prior['trace'] and replay['stats'] == prior['stats']
    assert all(tensor_equal(v, c.old[f'{label}:{p}']) for p, v in samples.items())


def profile(call, target, operators):
    import torch
    from diagnose_output_windows import file_hash
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
            call()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(target))
        events = json.loads(target.read_text(encoding='utf-8'))['traceEvents']
        cpu = [e for e in events if e.get('cat') == 'cpu_op' and e.get('name') in operators]
        ids = {e.get('args', {}).get('External id') for e in cpu}
        kernels = [e for e in events if e.get('cat') == 'kernel']
        return {'available': bool(kernels), 'trace_sha256': file_hash(target),
            'operator_events': [{'name': e['name'], 'external_id': e.get('args', {}).get('External id'),
                                 'input_dims': e.get('args', {}).get('Input Dims')} for e in cpu],
            'matched_kernel_names': sorted({e['name'] for e in kernels if e.get('args', {}).get('External id') in ids}),
            'all_kernel_names': sorted({e['name'] for e in kernels})}
    except (RuntimeError, OSError) as error:
        return {'available': False, 'error': f'{type(error).__name__}: {error}'}
