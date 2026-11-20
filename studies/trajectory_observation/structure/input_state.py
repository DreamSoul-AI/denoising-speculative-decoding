"""E1 input-state dev selection and frozen paired evaluation; no import-time work."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from spec_mtp.audit import JsonlSink, audit_run, verify_model_files
from spec_mtp.model import HFCausalLM
from spec_mtp.structure.algorithm.decode.loop import generate
from spec_mtp.structure.algorithm.mask import build_mask_provider
from spec_mtp.structure.api.decode_api import DecodeRequest, Prompt


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def choose_inputs(data, banned, seed):
    """Reserve evaluation first, then dev; original text is the exclusion key."""
    groups, seen = {}, set(banned)
    for index, row in enumerate(data):
        text = row['instruction'] + ('\n\n' + row['context'] if row.get('context') else '')
        if text in seen:
            continue
        seen.add(text)
        groups.setdefault(row['category'], []).append({'qid': str(index), 'category': row['category'], 'text': text})
    if len(groups) != 8:
        raise ValueError('E1 requires all eight Dolly categories')
    result = {'dev': [], 'eval': []}
    generator = random.Random(seed)
    for category in sorted(groups):
        rows = groups[category]
        if len(rows) < 5:
            raise ValueError('not enough unseen inputs in ' + category)
        generator.shuffle(rows)
        result['eval'].extend(rows[:4])
        result['dev'].append(rows[4])
    return result


def candidate_specs():
    common = {'method': 'input_trajectory', 'num_masks': 1, 'impl': 'efficient',
              'tree': {'mode': 'static', 'branches': [14], 'pruning': True}}
    arms = {'order0': {**common, 'trajectory': {'kind': 'order0', 'alpha': .1}},
            'esp': {**common, 'method': 'esp', 'mask': {'init': 'mean_prompt', 'lam': .1, 'update': True}}}
    for beta in (.9, .99, .9999):
        arms[f'diff_{beta}'] = {**common, 'trajectory': {'kind': 'difference', 'alpha': .1, 'beta': beta}}
    for alpha in (.1, .5, 1.):
        for beta in (.9, .99, .9999):
            arms[f'holt_{alpha}_{beta}'] = {**common, 'trajectory': {'kind': 'holt', 'alpha': alpha, 'beta': beta}}
    for q_level in (.0001, .01, 1.):
        for q_velocity in (.0001, .01, 1.):
            arms[f'kalman_{q_level}_{q_velocity}'] = {**common, 'trajectory': {
                'kind': 'kalman', 'q_level': q_level, 'q_velocity': q_velocity, 'r': 1., 'p0': 1.}}
    return arms


def encode_inputs(tokenizer, raw_inputs):
    """Render then encode explicitly across tokenizer return-type versions."""
    encoded = {}
    for split, rows in raw_inputs.items():
        encoded[split] = []
        for row in rows:
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': row['text']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(text, add_special_tokens=False)[-512:]
            encoded[split].append({**row, 'input_ids': ids})
    return encoded


def pooled(rows):
    counts = {key: sum(row[key] for row in rows) for key in ('A', 'C', 'S', 'N')}
    return {**counts, 'prompts': len(rows),
            'accepted_per_call': counts['A'] / counts['C'] if counts['C'] else None,
            'tau': counts['S'] / counts['C'] if counts['C'] else None}


def select_arm(rows, candidates):
    # max retains the first candidate on ties; no eval rows may enter selection.
    if any(row['split'] != 'dev' for row in rows):
        raise ValueError('selection requires dev rows only')
    scores = {}
    expected = None
    for arm in candidates:
        selected = [row for row in rows if row['arm'] == arm]
        qids = [row['qid'] for row in selected]
        if not selected or len(set(qids)) != len(qids):
            raise ValueError('missing or duplicate dev observations')
        if expected is None:
            expected = set(qids)
        if set(qids) != expected:
            raise ValueError('unpaired dev candidates')
        score = pooled(selected)['tau']
        if score is None:
            raise ValueError('zero-call dev comparison')
        scores[arm] = score
    return max(candidates, key=scores.__getitem__)


class ProvenanceSink(JsonlSink):
    """Per-token flushing, with model bytes checked at stage boundaries.

    Avoid rehashing all weights for every prompt. The manifest is attached to
    each raw header; a final drift check must succeed before a summary exists.
    """

    def __init__(self, path, model_sources, model_info):
        super().__init__(path)
        self.declared_model, self.actual_model = model_sources, model_info

    def write(self, observation):
        if observation.get('event') == 'header':
            observation = {**observation, 'model_sources': self.declared_model, 'model': self.actual_model}
        super().write(observation)


def run_experiment(config):
    from transformers import AutoTokenizer

    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config.get('timeout', 3600)
    write_json(root / 'config.json', config)
    model = None
    try:
        source_root = Path(config['source_root'])
        source_members = json.loads(Path(config['source_manifest']).read_text(encoding='utf-8'))
        def check_source():
            if any(file_hash(source_root / p) != digest for p, digest in source_members.items()):
                raise ValueError('frozen experiment source changed')
        check_source()
        model_root = Path(config['model'])
        model_members = json.loads(Path(config['model_manifest']).read_text(encoding='utf-8'))
        verify_model_files(model_root, model_members)
        raw_inputs = json.loads(Path(config['inputs']).read_text(encoding='utf-8'))
        sources = {'code': file_hash(config['source_manifest']), 'inputs': file_hash(config['inputs']),
                   'model': file_hash(config['model_manifest']), 'config': file_hash(root / 'config.json')}
        tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
        encoded = encode_inputs(tokenizer, raw_inputs)
        write_json(root / 'inputs.json', encoded)
        torch.set_num_threads(2)
        model = HFCausalLM(model_root, torch_dtype='bfloat16', attn_implementation='eager').to('cuda')
        model_info = model.describe()
        write_json(root / 'model.json', model_info)
        eos = model.hf.generation_config.eos_token_id
        eos_ids = tuple(eos if isinstance(eos, list) else [eos]) if eos is not None else ()
        arms = candidate_specs()
        if config.get('pilot'):
            names = ('order0', 'diff_0.9', 'holt_0.1_0.9', 'kalman_0.01_0.01')
            arms = {name: arms[name] for name in names}
            base = arms['kalman_0.01_0.01']
            arms['fixed_gain'] = {**base, 'trajectory': {**base['trajectory'], 'kind': 'fixed_gain'}}
        write_json(root / 'arms.json', arms)
        rows = []
        def execute(split, items, names):
            for i, item in enumerate(items):
                order = names[i % len(names):] + names[:i % len(names)]
                for name in order:
                    if time.monotonic() > deadline:
                        raise TimeoutError('E1 experiment budget exhausted')
                    spec = arms[name]
                    path = root / f'{split}-{item["qid"]}-{name}.jsonl'
                    request = DecodeRequest(Prompt(item['qid'], tuple(item['input_ids']), item['category']), spec,
                        max_new_tokens=config.get('new_tokens', 64), eos_ids=eos_ids,
                        seed=config.get('seed', 20261128), run_id=root.name, sources=sources)
                    provider = build_mask_provider(spec, model.embedding_table)
                    sink = ProvenanceSink(path, model_members, model_info)
                    try:
                        result = generate(request, model, provider, sink)
                    finally:
                        sink.close()
                    verdict = audit_run(path, sources, expected_model_sources=model_members)
                    if not (verdict.source_integrity and verdict.argmax_coverage and verdict.argmax_correctness):
                        raise ValueError('T0/provenance failed: ' + repr(verdict))
                    if getattr(provider, 'covariance', None) is not None:
                        p = provider.covariance
                        if not torch.isfinite(p).all() or torch.linalg.eigvalsh(p).min() < -1e-10:
                            raise ValueError('invalid state covariance')
                    row = {'split': split, 'qid': item['qid'], 'category': item['category'], 'arm': name,
                           **{k: result.stats['acceptance_metrics'][k] for k in ('A', 'C', 'S')},
                           'N': len(result.committed), 'tokens': list(result.committed),
                           'audit': asdict(verdict), 'raw': path.name}
                    rows.append(row)
                    with (root / 'rows.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    print(json.dumps({k: row[k] for k in ('split', 'qid', 'arm', 'A', 'C', 'S')}) , flush=True)
        if config.get('pilot'):
            execute('dev', encoded['dev'][:2], list(arms))
            selected = list(arms)
        else:
            execute('dev', encoded['dev'], list(arms))
            difference = select_arm(rows, [a for a in arms if a.startswith('diff_')])
            holt = select_arm(rows, [a for a in arms if a.startswith('holt_')])
            kalman = select_arm(rows, [a for a in arms if a.startswith('kalman_')])
            arms['fixed_gain'] = {**arms[kalman], 'trajectory': {**arms[kalman]['trajectory'], 'kind': 'fixed_gain'}}
            selected = ['order0', 'esp', difference, holt, kalman, 'fixed_gain']
            write_json(root / 'selection.json', {'selected': selected, 'specs': {a: arms[a] for a in selected}})
            execute('dev', encoded['dev'], ['fixed_gain'])
            verify_model_files(model_root, model_members)
            check_source()
            execute('eval', encoded['eval'], selected)
        verify_model_files(model_root, model_members)
        check_source()
        summary = {split: {a: pooled([r for r in rows if r['split'] == split and r['arm'] == a])
                           for a in selected} for split in sorted({r['split'] for r in rows})}
        write_json(root / 'summary.json', {'complete': True, 'summary': summary,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'selected': selected})
    except BaseException as error:
        write_json(root / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        if model is not None:
            model.close()
