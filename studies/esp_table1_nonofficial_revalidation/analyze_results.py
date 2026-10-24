"""CPU readout of the three native full480 nonofficial independent-repeat cells."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BASE = ROOT / 'studies/esp_table1_bf16_full_argmax'
sys.path.insert(0, str(BASE))
_spec = importlib.util.spec_from_file_location('original_table1_aggregation', BASE / 'analyze_results.py')
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
summarize = _original.summarize
TARGETS = {10: None, 30: 1.63, 60: 1.74}
METRICS = ('pooled_tau', 'prompt_mean_tau', 'category_mean_pooled_tau')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_difference(a, b):
    if a == b:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def audited_rows(raw, audits, prompts, budget):
    from argmax_audit import token_digest
    assert len(raw['rows']) == len(raw['evidence']) == len(audits) == len(prompts)
    rows, tokens = [], []
    for prompt, row, item, audit in zip(prompts, raw['rows'], raw['evidence'], audits):
        ids = prompt['input_ids']
        assert row['prompt'] == len(rows)
        assert item['input_ids'] == ids and item['output_ids'][:len(ids)] == item['ar_ids'][:len(ids)] == ids
        stats = item['stats']
        assert row['category'] == prompt['category'] and row['spec_time'] == row['ar_time'] == 0
        assert stats['prefill_calls'] == 1 and stats['block_complexity'] == budget
        for name in ('accepted', 'committed', 'decode_calls', 'new_tokens'):
            assert row[name] == stats[name]
        output, ar = item['output_ids'][len(ids):], item['ar_ids'][len(ids):]
        assert stats['new_tokens'] == len(output) and 0 < len(output) <= 256
        assert stats['committed'] == len(output) - 1
        assert audit == stats['verification_argmax_audit'] and audit['passed'] is True
        assert audit['checked_committed_tokens'] == len(audit['committed']) == len(output)
        assert audit['input_sha256'] == token_digest(ids) and audit['output_sha256'] == token_digest(item['output_ids'])
        assert all(a['chosen'] == a['argmax'] == token for a, token in zip(audit['committed'], output))
        exact = item['output_ids'] == item['ar_ids']
        assert bool(row['exact_match']) == exact
        first = first_difference(output, ar)
        rows.append(dict(row, question_id=prompt['question_id'], exact=exact, AR_new_tokens=len(ar),
            first_new_difference_index=first, argmax_passed=True, argmax_checked_tokens=len(output)))
        tokens.append({'question_id': prompt['question_id'], 'output_sha256': token_digest(item['output_ids']),
            'AR_sha256': token_digest(item['ar_ids']), 'first_new_difference_index': first})
    return rows, tokens


def compare(old, new):
    assert len(old['rows']) == len(new['rows']) == len(old['token_audits']) == len(new['token_audits'])
    rows = []
    for a, b, at, bt in zip(old['rows'], new['rows'], old['token_audits'], new['token_audits']):
        assert a['question_id'] == b['question_id'] == at['question_id'] == bt['question_id']
        assert a['category'] == b['category']
        change = {k: b[k] - a[k] for k in ('accepted', 'decode_calls', 'committed', 'new_tokens')}
        tau_cross = b['committed'] * max(a['decode_calls'], 1) - a['committed'] * max(b['decode_calls'], 1)
        label = lambda x: 'win' if x > 0 else 'loss' if x < 0 else 'tie'
        rows.append({'question_id': a['question_id'], 'category': a['category'],
            'delta_new_minus_old': change, 'accepted': label(change['accepted']),
            'calls': label(-change['decode_calls']), 'tau': label(tau_cross),
            'output_hash_equal': at['output_sha256'] == bt['output_sha256'],
            'AR_hash_equal': at['AR_sha256'] == bt['AR_sha256']})
    return {'n': len(rows), **{metric: {s: sum(r[metric] == s for r in rows) for s in ('win', 'tie', 'loss')}
            for metric in ('accepted', 'calls', 'tau')},
            'output_hash_equal_count': sum(r['output_hash_equal'] for r in rows),
            'AR_hash_equal_count': sum(r['AR_hash_equal'] for r in rows), 'rows': rows}



def verify_contract(inputs, previous, cfg):
    expected = dict(cfg['algorithm']['config']['eval'], device=cfg['system']['config']['device'])
    assert inputs['settings']['eval'] == previous['settings']['eval'] == expected
    assert inputs['settings']['spec'] == previous['settings']['spec'] == cfg['algorithm']['config']['spec']
    assert inputs['eos_ids'] == previous['eos_ids'] and inputs['eos_ids']
    assert inputs['model_digest'] == previous['model_digest']

def main(baseline_runs=None, study_dir=None, output_dir=None):
    from revalidation_contract import check_refs
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    from rpipe.structure.make.expand import load_study_yaml
    from studies._shared.structure.sources import check_frozen, manifest
    from studies._shared.structure.validation import verify_evidence
    from studies._shared.structure.evidence import digest
    study = STUDY if study_dir is None else Path(study_dir).resolve()
    output = study / 'docs' if output_dir is None else Path(output_dir).resolve()
    assert Path.cwd().resolve() == study.parents[1], 'Run from the frozen execution checkout root for declared relative inputs'
    baseline_runs = BASE / 'runs' if baseline_runs is None else Path(baseline_runs)
    check_refs()
    decl = load_study_yaml(study)
    check_frozen(study, decl, manifest())
    pcheck(study, decl)
    original = read(BASE / 'docs/protocol_evidence_v1.json')
    prompt_manifest = read(BASE / 'docs/prompts_manifest.json')
    prompts = prompt_manifest['rendered_prompts']
    snapshot = read(BASE / 'docs/modelscope_snapshot.json')
    assert len(prompts) == 480
    baseline = {c['BC']: c for c in original['cells'] if c['role'] == 'primary'}
    entries = read(study / 'index.json')['experiments']
    assert len(entries) == 3
    cells = {}
    for entry in entries:
        spec = entry['factors']['algorithm.config']['spec']
        budget = spec['block_complexity']
        assert budget in TARGETS and budget not in cells and spec['num_masks'] == 1 and len(entry['runs']) == 1
        assert spec['method'] == 'esp' and spec['tree'] == {'mode': 'dynamic', 'pruning': True}
        assert spec['mask'] == {'init': 'mean_prompt', 'lam': .1, 'update': True}
        folder = study / 'runs' / entry['runs'][0]['id']
        result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
        cells[budget] = {'BC': budget, 'num_masks': 1, 'run_id': folder.name,
            'run_path': folder.relative_to(study.parents[1]).as_posix(), 'native_status': result.get('status', 'pending'), 'target': TARGETS[budget], 'complete': False}
        if result.get('status') != 'succeeded':
            continue
        assert result['spec_mtp']['passed'] and result['spec_mtp']['validation_policy'] == 'verifier-argmax-v1'
        assets = folder / 'assets'
        cfg = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
        assert cfg['seed'] == 123 and cfg['algorithm']['config']['spec'] == spec
        verify_evidence(cfg, assets)
        inputs, raw = read(assets / 'decode_inputs.json'), read(assets / 'decode_observations.json')
        assert raw['inputs_digest'] == digest(inputs)
        assert inputs['dtype'] == 'torch.bfloat16' and inputs['backend'] == 'eager'
        assert inputs['model_digest'] == baseline[budget]['model_digest']
        old_inputs_path = baseline_runs / baseline[budget]['run_id'] / 'assets/decode_inputs.json'
        assert sha(old_inputs_path) == baseline[budget]['inputs_sha256']
        verify_contract(inputs, read(old_inputs_path), cfg)
        settings = inputs['settings']['eval']
        assert settings['temperature'] == 0 and settings['max_new_tokens'] == 256
        assert settings['stop_on_eos'] and settings['run_ar_baseline'] and settings['warmup'] is False
        assert inputs['settings']['spec'] == spec
        assert len(inputs['prompts']) == 480 and all(a['input_ids'] == b['input_ids'] and a['category'] == b['category']
            for a, b in zip(inputs['prompts'], prompts))
        landed = {Path(name).name: value for name, value in inputs['files'].items()}
        assert all(landed[f['path']] == f['sha256'] for f in snapshot['files'])
        policy = read(assets / 'numeric_policy.json')
        assert policy == {'dtype': 'torch.bfloat16', 'bf16_reduced_precision_reduction': True, 'deterministic': False}
        audits = [json.loads(line) for line in (assets / 'argmax_audit.jsonl').read_text(encoding='utf-8').splitlines()]
        rows, tokens = audited_rows(raw['out'], audits, prompts, budget)
        cell = cells[budget]
        cell.update(summarize(rows), complete=True, rows=rows, token_audits=tokens,
            model_digest=inputs['model_digest'], runtime=inputs['runtime'], numeric_policy=policy,
            native_validation_policy=result['spec_mtp']['validation_policy'],
            validator_sha256=result['spec_mtp']['validator_sha256'],
            files_sha256={f.relative_to(folder).as_posix(): sha(f) for f in
                (folder / 'config.yaml', folder / 'result.json', assets / 'decode_observations.json',
                 assets / 'decode_inputs.json', assets / 'argmax_audit.jsonl', assets / 'spec_mtp_sources.json')})
        cell['original_run_id'] = baseline[budget]['run_id']
        cell['original_tau'] = {metric: baseline[budget][metric] for metric in METRICS}
        cell['vs_original'] = compare(baseline[budget], cell)
        cell['metric_deltas'] = {metric: {'new_minus_original': cell[metric] - baseline[budget][metric],
            'new_minus_paper': None if TARGETS[budget] is None else cell[metric] - TARGETS[budget]} for metric in METRICS}
    assert set(cells) == set(TARGETS)
    ready = all(c['complete'] for c in cells.values())
    if ready:
        process, native = read(study / 'process.json'), read(study / 'spec_mtp_study_check.json')
        assert process['complete'] and process['spec_mtp']['passed'] and native['complete'] and native['passed']
    body = {'complete': ready, 'author_official': False, 'no_timing': True, 'fixed_prompts': 480,
        'correctness_gate': 'all committed tokens equal actual verifier-forward argmax; independent AR diagnostic only',
        'original_evidence_sha256': sha(BASE / 'docs/protocol_evidence_v1.json'),
        'source_refs_sha256': sha(study / 'docs/source_refs.json'), 'analyzer_sha256': sha(Path(__file__)),
        'reused_aggregation_sha256': sha(BASE / 'analyze_results.py'),
        'execution_root': str(study.parents[1]), 'cells': [cells[bc] for bc in TARGETS]}
    output.mkdir(parents=True, exist_ok=True)
    path = output / ('repeat_evidence_v1.json' if ready else 'repeat_pending.json')
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if ready and path.exists():
        assert path.read_text(encoding='utf-8') == text
    else:
        path.write_text(text, encoding='utf-8')
    lines = ['# 非官方8B Table1独立复验', '', '固定原480、Qwen3-8B BF16/T0、单mask、λ=.1 updating mean、动态树剪枝、256/EOS；无测速。', '',
        '| BC | native | argmax checked/pass | pooled τ | prompt mean τ | category mean τ | pooled τ−论文 | pooled τ−旧结果 | AR exact/480 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in body['cells']:
        if not c['complete']:
            lines.append(f"| {c['BC']} | {c['native_status']} | pending | pending | pending | pending | pending | pending | pending |")
            continue
        delta = c['metric_deltas']['pooled_tau']
        value = 'n/a' if delta['new_minus_paper'] is None else f"{delta['new_minus_paper']:.9f}"
        lines.append(f"| {c['BC']} | succeeded | {c['argmax_checked_tokens']}/{c['argmax_all_passed']} | {c['pooled_tau']:.9f} | {c['prompt_mean_tau']:.9f} | {c['category_mean_pooled_tau']:.9f} | {value} | {delta['new_minus_original']:.9f} | {c['exact_count']}/480 |")
    lines += ['', '| BC | 聚合 | 原结果τ | 新τ−论文 | 新τ−原结果 |', '| --- | --- | --- | --- | --- |']
    for c in body['cells']:
        if not c['complete']:
            continue
        for metric in METRICS:
            d = c['metric_deltas'][metric]
            paper = 'n/a' if d['new_minus_paper'] is None else f"{d['new_minus_paper']:.9f}"
            lines.append(f"| {c['BC']} | {metric} | {c['original_tau'][metric]:.9f} | {paper} | {d['new_minus_original']:.9f} |")
    lines += ['', '主指标pooled τ=Σcommitted/Σdecode_calls，不含prefill；逐prompt等权及逐类别pooled后等权另列，均不是已确认的作者权重。AR一致率只诊断，不作失败门。', '',
        '逐prompt accepted/calls/τ赢平输、输出长度、旧/新输出与AR摘要是否相同、AR首分叉及所有新旧目标差值见JSON；不同长度/轨迹的配对计数不能解释为速度或同前缀因果。', '',
        '三格完整且原生process通过才生成正式repeat_evidence_v1.json；各已完成格可以单独列有效τ，但不称全矩阵完成。数学自检通过不等于论文目标数值已复现。', '',
        f"来源：`{'repeat_evidence_v1.json' if ready else 'repeat_pending.json'}`，每格files_sha256给出Run原始结果/输入/审计路径与哈希；旧结果`studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json`，新Run路径为每格run_path。原始tokens、logits、完整Run仅本机保留。"]
    (output / 'STUDY_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'complete': ready, 'cells_complete': sum(c['complete'] for c in cells.values())}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-runs', type=Path, default=BASE / 'runs',
                        help='Old immutable Run directory, when stored in another checkout')
    parser.add_argument('--study-dir', type=Path, help='Frozen native Study in its execution checkout')
    parser.add_argument('--output-dir', type=Path, help='CPU report directory; avoid writing into a live execution tree')
    args = parser.parse_args()
    main(args.baseline_runs, args.study_dir, args.output_dir)
