"""Read saved evidence into a fresh external report; never execute a model."""
from studies._shared.flow.process import run as run

import argparse
import json
from pathlib import Path
import yaml
from studies.esp_table1_nonofficial_revalidation.structure.readout import summarize, TARGETS, METRICS, audited_rows, compare, verify_contract
from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout


def check_refs(study, execution_root, session):
    """Read frozen producer references without importing its model execution protocol."""
    from spec_mtp.structure.artifact.provenance import read_historical_bytes

    refs = session.read_json(study / 'docs/source_refs.json')
    for name, expected in refs['files'].items():
        assert session.sha(execution_root / name) == expected, name
    plan = refs['project_plan']
    read_historical_bytes(execution_root, plan['commit'], plan['path'], plan['sha256'])
    session.historical.append(dict(plan))
    return refs

def analyze(study_dir, output_root, *, baseline_dir, baseline_runs, execution_root):
    from spec_mtp.structure.artifact.provenance import check_frozen as pcheck
    from spec_mtp.structure.make.expand import load_study_yaml
    from studies._shared.structure.sources import check_frozen, manifest
    from studies._shared.structure.validation import verify_evidence
    from studies._shared.structure.evidence import digest
    study = Path(study_dir).resolve()
    output = Path(output_root).resolve()
    base = Path(baseline_dir).resolve()
    execution_root = Path(execution_root).resolve()
    assert execution_root == study.parents[1], 'frozen execution checkout root must match the saved Study hierarchy'
    baseline_runs = Path(baseline_runs).resolve()
    session = SavedReadout(study, output, base, baseline_runs, execution_root)
    read, sha = (session.read_json, session.sha)
    check_refs(study, execution_root, session)
    decl = load_study_yaml(study)
    check_frozen(study, decl, manifest())
    pcheck(study, decl)
    original = read(base / 'docs/protocol_evidence_v1.json')
    prompt_manifest = read(base / 'docs/prompts_manifest.json')
    prompts = prompt_manifest['rendered_prompts']
    snapshot = read(base / 'docs/modelscope_snapshot.json')
    assert len(prompts) == 480
    baseline = {c['BC']: c for c in original['cells'] if c['role'] == 'primary'}
    entries = read(study / 'index.json')['experiments']
    assert len(entries) == 3
    cells = {}
    for entry in entries:
        spec = entry['factors']['algorithm.config']['spec']
        budget = spec['block_complexity']
        assert budget in TARGETS and budget not in cells and (spec['num_masks'] == 1) and (len(entry['runs']) == 1)
        assert spec['method'] == 'esp' and spec['tree'] == {'mode': 'dynamic', 'pruning': True}
        assert spec['mask'] == {'init': 'mean_prompt', 'lam': 0.1, 'update': True}
        folder = study / 'runs' / entry['runs'][0]['id']
        result = read(folder / 'result.json') if session.exists(folder / 'result.json') else {}
        cells[budget] = {'BC': budget, 'num_masks': 1, 'run_id': folder.name, 'run_path': folder.relative_to(study.parents[1]).as_posix(), 'native_status': result.get('status', 'pending'), 'target': TARGETS[budget], 'complete': False}
        if result.get('status') != 'succeeded':
            continue
        assert result['spec_mtp']['passed'] and result['spec_mtp']['validation_policy'] == 'verifier-argmax-v1'
        assets = folder / 'assets'
        cfg = yaml.safe_load(session.read_text(folder / 'config.yaml', encoding='utf-8'))
        assert cfg['seed'] == 123 and cfg['algorithm']['config']['spec'] == spec
        verify_evidence(cfg, assets)
        inputs, raw = (read(assets / 'decode_inputs.json'), read(assets / 'decode_observations.json'))
        assert raw['inputs_digest'] == digest(inputs)
        assert inputs['dtype'] == 'torch.bfloat16' and inputs['backend'] == 'eager'
        assert inputs['model_digest'] == baseline[budget]['model_digest']
        old_inputs_path = baseline_runs / baseline[budget]['run_id'] / 'assets/decode_inputs.json'
        assert sha(old_inputs_path) == baseline[budget]['inputs_sha256']
        verify_contract(inputs, read(old_inputs_path), cfg)
        settings = inputs['settings']['eval']
        assert settings['temperature'] == 0 and settings['max_new_tokens'] == 256
        assert settings['stop_on_eos'] and settings['run_ar_baseline'] and (settings['warmup'] is False)
        assert inputs['settings']['spec'] == spec
        assert len(inputs['prompts']) == 480 and all((a['input_ids'] == b['input_ids'] and a['category'] == b['category'] for a, b in zip(inputs['prompts'], prompts)))
        landed = {Path(name).name: value for name, value in inputs['files'].items()}
        assert all((landed[f['path']] == f['sha256'] for f in snapshot['files']))
        policy = read(assets / 'numeric_policy.json')
        assert policy == {'dtype': 'torch.bfloat16', 'bf16_reduced_precision_reduction': True, 'deterministic': False}
        audits = [json.loads(line) for line in session.read_text(assets / 'argmax_audit.jsonl', encoding='utf-8').splitlines()]
        rows, tokens = audited_rows(raw['out'], audits, prompts, budget)
        cell = cells[budget]
        cell.update(summarize(rows), complete=True, rows=rows, token_audits=tokens, model_digest=inputs['model_digest'], runtime=inputs['runtime'], numeric_policy=policy, native_validation_policy=result['spec_mtp']['validation_policy'], validator_sha256=result['spec_mtp']['validator_sha256'], files_sha256={f.relative_to(folder).as_posix(): sha(f) for f in (folder / 'config.yaml', folder / 'result.json', assets / 'decode_observations.json', assets / 'decode_inputs.json', assets / 'argmax_audit.jsonl', assets / 'spec_mtp_sources.json')})
        cell['original_run_id'] = baseline[budget]['run_id']
        cell['original_tau'] = {metric: baseline[budget][metric] for metric in METRICS}
        cell['vs_original'] = compare(baseline[budget], cell)
        cell['metric_deltas'] = {metric: {'new_minus_original': cell[metric] - baseline[budget][metric], 'new_minus_paper': None if TARGETS[budget] is None else cell[metric] - TARGETS[budget]} for metric in METRICS}
    assert set(cells) == set(TARGETS)
    ready = all((c['complete'] for c in cells.values()))
    if ready:
        process, native = (read(study / 'process.json'), read(study / 'spec_mtp_study_check.json'))
        assert process['complete'] and process['spec_mtp']['passed'] and native['complete'] and native['passed']
    body = {'complete': ready, 'author_official': False, 'no_timing': True, 'fixed_prompts': 480, 'correctness_gate': 'all committed tokens equal actual verifier-forward argmax; independent AR diagnostic only', 'original_evidence_sha256': sha(base / 'docs/protocol_evidence_v1.json'), 'source_refs_sha256': sha(study / 'docs/source_refs.json'), 'analyzer_sha256': sha(Path(__file__)), 'reused_aggregation_sha256': sha(base / 'analyze_results.py'), 'execution_root': str(study.parents[1]), 'cells': [cells[bc] for bc in TARGETS]}
    path = output / ('repeat_evidence_v1.json' if ready else 'repeat_pending.json')
    text = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if ready and session.exists(path):
        assert session.read_text(path, encoding='utf-8') == text
    else:
        session.write_text(path, text, encoding='utf-8')
    lines = ['# 非官方8B Table1独立复验', '', '固定原480、Qwen3-8B BF16/T0、单mask、λ=.1 updating mean、动态树剪枝、256/EOS；无测速。', '', '| BC | native | argmax checked/pass | pooled τ | prompt mean τ | category mean τ | pooled τ−论文 | pooled τ−旧结果 | AR exact/480 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
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
    lines += ['', '主指标pooled τ=Σcommitted/Σdecode_calls，不含prefill；逐prompt等权及逐类别pooled后等权另列，均不是已确认的作者权重。AR一致率只诊断，不作失败门。', '', '逐prompt accepted/calls/τ赢平输、输出长度、旧/新输出与AR摘要是否相同、AR首分叉及所有新旧目标差值见JSON；不同长度/轨迹的配对计数不能解释为速度或同前缀因果。', '', '三格完整且原生process通过才生成正式repeat_evidence_v1.json；各已完成格可以单独列有效τ，但不称全矩阵完成。数学自检通过不等于论文目标数值已复现。', '', f"来源：`{('repeat_evidence_v1.json' if ready else 'repeat_pending.json')}`，每格files_sha256给出Run原始结果/输入/审计路径与哈希；旧结果`studies/esp_table1_bf16_full_argmax/docs/protocol_evidence_v1.json`，新Run路径为每格run_path。原始tokens、logits、完整Run仅本机保留。"]
    session.write_text(output / 'STUDY_REPORT.md', '\n'.join(lines) + '\n', encoding='utf-8')
    session.finish()
    return body

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study-dir','output-root','execution-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--baseline-runs',type=Path,required=True)
    args=parser.parse_args(argv)
    analyze(args.study_dir,args.output_root,baseline_dir=args.baseline_dir,baseline_runs=args.baseline_runs,execution_root=args.execution_root)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
