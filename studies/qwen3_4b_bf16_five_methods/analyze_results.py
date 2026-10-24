"""Native evidence readout: paired counts, AR diagnostics, no generation or timing."""
import hashlib
import json
from itertools import combinations
from pathlib import Path

METHODS = ('frozen_mean', 'input_zero', 'input_first', 'output_zero', 'output_first')
CASES = ('AR',) + tuple(f'{bc}/{method}' for bc in (30, 60) for method in METHODS)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def case_name(settings):
    if settings['task'] == 'ar_baseline':
        return 'AR'
    assert settings['task'] == 'esp_mtp'
    spec = settings['spec']
    if spec['method'] == 'esp':
        method = 'input_zero' if spec['mask']['update'] else 'frozen_mean'
    elif spec['method'] == 'ema_velocity':
        method = 'input_first'
    else:
        assert spec['method'] == 'output_hidden' and spec['output']['order'] in (0, 1)
        method = ('output_zero', 'output_first')[spec['output']['order']]
    name = f"{spec['block_complexity']}/{method}"
    assert name in CASES
    return name


def first_difference(output, reference):
    if output == reference:
        return None
    return next((i for i, (a, b) in enumerate(zip(output, reference)) if a != b),
                min(len(output), len(reference)))


def compare_rows(first, second):
    assert first and len(first) == len(second)
    rows = []
    assert len({r['question_id'] for r in first}) == len(first)
    for left, right in zip(first, second):
        assert (left['question_id'], left['category']) == (right['question_id'], right['category'])
        assert all(r[k] >= 0 for r in (left, right) for k in ('accepted', 'model_calls'))
        delta = {k: right[k] - left[k] for k in ('accepted', 'model_calls', 'new_tokens')}
        rows.append({'question_id': left['question_id'], 'category': left['category'],
                     'delta_second_minus_first': delta,
                     'accepted': 'win' if delta['accepted'] > 0 else 'loss' if delta['accepted'] < 0 else 'tie',
                     'calls': 'win' if delta['model_calls'] < 0 else 'loss' if delta['model_calls'] > 0 else 'tie'})
    return {'n': len(rows), 'accepted': {s: sum(r['accepted'] == s for r in rows) for s in ('win', 'tie', 'loss')},
            'calls': {s: sum(r['calls'] == s for r in rows) for s in ('win', 'tie', 'loss')},
            'total_delta_second_minus_first': {k: sum(r['delta_second_minus_first'][k] for r in rows)
                                               for k in ('accepted', 'model_calls', 'new_tokens')}, 'rows': rows}


def summarize(rows, is_ar):
    totals = {k: sum(r[k] for r in rows) for k in ('new_tokens', 'model_calls')}
    if not is_ar:
        totals.update({k: sum(r[k] for r in rows) for k in ('accepted', 'committed', 'decode_calls', 'argmax_checked_tokens')})
        totals.update(pooled_tau=totals['committed']/totals['decode_calls'],
                      prompt_mean_tau=sum(r['committed']/max(r['decode_calls'], 1) for r in rows)/len(rows),
                      AR_exact_count=sum(r['AR_exact'] for r in rows),
                      AR_exact_rate=sum(r['AR_exact'] for r in rows)/len(rows),
                      calls_delta_vs_AR=sum(r['calls_delta_vs_AR'] for r in rows),
                      calls_vs_AR={s: sum(r['calls_vs_AR'] == s for r in rows) for s in ('win', 'tie', 'loss')})
    return totals


def load_cell(study, folder, cfg, prompts, manifest, snapshot, reference):
    from studies._shared.structure.evidence import file_hash
    from studies._shared.structure.validation import verify_evidence
    from untimed_pairing import verify_inputs
    assets = folder/'assets'
    result = read(folder/'result.json')
    gate = result['spec_mtp']
    assert result['status'] == 'succeeded' and gate['passed']
    assert gate['validation_policy'] == 'verifier-argmax-v1'
    verify_evidence(cfg, assets)
    inputs = read(assets/'decode_inputs.json')
    verify_inputs(inputs, manifest, snapshot)
    contract = read(assets/'pairing_contract.json')
    assert contract['dtype'] == 'torch.bfloat16' and contract['device'].startswith('cuda')
    assert contract['tied_storage'] and contract['manifest_checked']
    assert contract['embedding_shape'] == [snapshot['config']['vocab_size'], snapshot['config']['hidden_size']]
    assert contract['bf16_reduced_precision_reduction'] and not contract['deterministic']
    settings = cfg['algorithm']['config']
    is_ar = settings['task'] == 'ar_baseline'
    raw = read(assets/'decode_observations.json')['out']
    assert len(raw['rows']) == len(raw['evidence']) == len(prompts) == 480
    assert is_ar or reference is not None
    rows, outputs = [], []
    for i, (p, row, item) in enumerate(zip(prompts, raw['rows'], raw['evidence'])):
        assert item['input_ids'] == p['input_ids']
        output = item['output_ids'][len(p['input_ids']):]
        stats = item['stats']
        common = {'question_id': p['question_id'], 'category': p['category'], 'generator_seed': 123+i,
                  'new_tokens': len(output), 'output_sha256': hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                  'hit_context_limit': stats.get('hit_context_limit', False)}
        if is_ar:
            assert stats['ar_time'] == row['ar_time'] == 0
            common['model_calls'] = stats['ar_calls']
            outputs.append({'question_id': p['question_id'], 'output_ids': output, 'model_calls': stats['ar_calls']})
        else:
            assert stats['prefill_time'] == stats['decode_time'] == row['spec_time'] == 0
            assert item.get('ar_ids') is None and 'ar_time' not in row
            assert stats['verification_argmax_audit']['passed'] is True
            assert stats['prefill_calls'] == 1
            ar = reference[i]
            assert ar['question_id'] == p['question_id']
            delta = stats['decode_calls']+stats['prefill_calls']-ar['model_calls']
            common.update({k: stats[k] for k in ('accepted', 'committed', 'decode_calls')})
            common.update(model_calls=stats['decode_calls']+stats['prefill_calls'],
                          argmax_checked_tokens=stats['verification_argmax_audit']['checked_committed_tokens'],
                          AR_exact=output == ar['output_ids'], AR_new_tokens=len(ar['output_ids']),
                          AR_model_calls=ar['model_calls'], calls_delta_vs_AR=delta,
                          calls_vs_AR='win' if delta < 0 else 'loss' if delta > 0 else 'tie',
                          first_new_difference_index=first_difference(output, ar['output_ids']))
        rows.append(common)
    files = {str(p.relative_to(folder)).replace('\\', '/'): file_hash(p) for p in
             (folder/'config.yaml', folder/'result.json', assets/'decode_observations.json',
              assets/'decode_inputs.json', assets/'pairing_contract.json', assets/'spec_mtp_sources.json')}
    return {'run_id': folder.name, 'native_status': result['status'], 'validator_sha256': gate['validator_sha256'],
            'source_digest': gate['sources_digest'], 'model_digest': inputs['model_digest'],
            'runtime': inputs['runtime'], 'actual_settings': settings, 'contract': contract,
            'files_sha256': files, 'totals': summarize(rows, is_ar), 'rows': rows}, outputs


def main():
    import torch  # noqa: F401 - Windows torch-first
    import yaml
    from rpipe.structure.artifact.provenance import check_frozen as pcheck
    from rpipe.structure.make.expand import load_study_yaml
    from studies._shared.structure.sources import manifest as sources, check_frozen
    from studies._shared.structure.evidence import file_hash
    study = Path(__file__).resolve().parent
    cells = {name: {'native_status': 'pending'} for name in CASES}
    index_path = study/'index.json'
    if index_path.exists():
        decl = load_study_yaml(study)
        check_frozen(study, decl, sources())
        pcheck(study, decl)
        index = read(index_path)
        assert len(index['experiments']) == 11
        entries = {}
        for exp in index['experiments']:
            name = case_name(exp['factors']['algorithm.config'])
            assert name not in entries and len(exp['runs']) == 1
            entries[name] = exp['runs'][0]['id']
        assert set(entries) == set(CASES)
        manifest = read(study/'docs/prompts_manifest.json')
        snapshot = read(study/'docs/modelscope_snapshot.json')
        reference = None
        for name in CASES:
            folder = study/'runs'/entries[name]
            path = folder/'result.json'
            result = read(path) if path.exists() else {}
            cells[name] = {'run_id': folder.name, 'native_status': result.get('status', 'pending')}
            if result.get('status') != 'succeeded':
                continue
            cfg = yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
            assert cfg['seed'] == 123 and case_name(cfg['algorithm']['config']) == name
            assert cfg['model']['config'] == {'name_or_path': 'Qwen/Qwen3-4B', 'hub': 'modelscope',
                'local_files_only': True, 'torch_dtype': 'bfloat16', 'attn_implementation': 'eager'}
            cells[name], outputs = load_cell(study, folder, cfg, manifest['rendered_prompts'], manifest, snapshot, reference)
            if name == 'AR':
                reference = outputs
            else:
                assert cells[name]['model_digest'] == cells['AR']['model_digest']
                assert cells[name]['source_digest'] == cells['AR']['source_digest']
                assert cells[name]['runtime'] == cells['AR']['runtime']
    pairs = []
    for bc in (30, 60):
        for a, b in combinations(METHODS, 2):
            first, second = f'{bc}/{a}', f'{bc}/{b}'
            if 'rows' in cells[first] and 'rows' in cells[second]:
                pairs.append(dict(first=first, second=second, **compare_rows(cells[first]['rows'], cells[second]['rows'])))
    ready = all('rows' in c for c in cells.values())
    if ready:
        quality, process = read(study/'spec_mtp_study_check.json'), read(study/'process.json')
        assert quality['passed'] and quality['complete'] and process['complete'] and process['spec_mtp']['passed']
    body = {'complete': ready, 'scope': '480 prompts; BF16/T0; two-mask dynamic pruned BC30/60; one seed',
            'argmax_is_gate_AR_is_diagnostic': True, 'timing_disabled': True,
            'analyzer_sha256': file_hash(Path(__file__)), 'cells': cells, 'pairs': pairs}
    if ready:
        body['plotter_sha256'] = file_hash(study/'plot_results.py')
    name = 'paired_evidence_v1.json' if ready else 'paired_pending.json'
    target = study/'docs'/name
    text = json.dumps(body, indent=2, allow_nan=False)+'\n'
    if ready and target.exists():
        assert target.read_text(encoding='utf-8') == text
    else:
        assert ready or not (study/'docs/paired_evidence_v1.json').exists()
    if ready:
        from plot_results import write_plot
        write_plot(body, study/'docs/figures/paired_outcomes.png')
    if not ready or not target.exists():
        target.write_text(text, encoding='utf-8')
    lines = ['# Qwen3-4B BF16 paired report', '', f"完整验收：{ready}；已完成 {sum('rows' in c for c in cells.values())}/11。",
             '', '| Cell | native | accepted | calls (prefill+decode) | new tokens | pooled τ | prompt mean τ | AR exact/n (rate) | argmax checked |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for case, cell in cells.items():
        t = cell.get('totals', {})
        values = [str(t.get(k, 'pending' if not t else 'n/a')) for k in
                  ('accepted', 'model_calls', 'new_tokens', 'pooled_tau', 'prompt_mean_tau', 'AR_exact_count', 'argmax_checked_tokens')]
        if 'AR_exact_count' in t:
            values[5] = f"{t['AR_exact_count']}/{len(cell['rows'])} ({t['AR_exact_rate']:.2%})"
        lines.append(f"| {case} | {cell['native_status']} | "+' | '.join(values)+' |')
    lines += ['', '相对独立AR的逐prompt调用数比较；calls越低越好。AR没有draft接受数，该指标不适用。输出长度及AR首分叉单列，不能将不同长度的调用数直接解释为等长加速。', '',
              '| Cell | calls vs AR W/T/L | Δcalls vs AR |', '| --- | --- | --- |']
    for case, cell in cells.items():
        totals = cell.get('totals', {})
        if 'calls_vs_AR' in totals:
            wtl = '/'.join(str(totals['calls_vs_AR'][s]) for s in ('win', 'tie', 'loss'))
            lines.append(f"| {case} | {wtl} | {totals['calls_delta_vs_AR']} |")
    lines += ['', '以下胜负指第二方法相对第一方法；accepted越高越好，calls越低越好，两个指标分别判定。', '',
              '| first → second | accepted W/T/L | calls W/T/L | Δaccepted | Δcalls | Δnew tokens |', '| --- | --- | --- | --- | --- | --- |']
    for pair in pairs:
        wtl = lambda k: '/'.join(str(pair[k][s]) for s in ('win', 'tie', 'loss'))
        d = pair['total_delta_second_minus_first']
        lines.append(f"| {pair['first']} → {pair['second']} | {wtl('accepted')} | {wtl('calls')} | {d['accepted']} | {d['model_calls']} | {d['new_tokens']} |")
    lines += ['', 'accepted为EOS/预算裁剪前候选接受数；committed及τ包含bonus、按裁剪后实际解码提交数计算，不含prefill首token。argmax审计含首token且必须全过；独立AR首分叉和输出长度全部保留，不作为失败门。AR参考不冒充Spec验证argmax审计。', '',
              '模型、forward及最终mask为BF16；输出EMA/范数沿用原生FP32内部状态。不同方法β等沿用预先定义配方，不声称差异仅由输入/输出空间造成。单seed、固定480条，不推断总体优越或测速收益；零计时字段为占位。', '',
              f'逐prompt计数、两两赢平输及AR首分叉见[{name}]({name})。' if ready else '尚无完整实测结论；本机pending文件不作为公开证据。', '', '[冻结计划](<../../../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-qwen3-4b-bf16-five-methods-docs-plan>)；原始tokens、完整Run和权重仅本机保留。']
    if ready:
        lines += ['', '![Full480 paired wins ties losses](figures/paired_outcomes.png)', '',
                  '图：行方法相对列方法，格内为W/T/L，颜色为净赢数W−L，固定范围±480；接受数越高越好，调用数越低越好。两个预算的全部方法对照和输局均保留，对角不作比较。独立AR调用数、一致率、首分叉与输出长度另列于表格及正式证据；本图不表示测速或总体显著性。']
    (study/'docs/STUDY_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'complete': ready, 'observed': sum('rows' in c for c in cells.values()), 'pairs': len(pairs)}))


if __name__ == '__main__':
    main()
