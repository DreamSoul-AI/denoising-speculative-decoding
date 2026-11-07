"""Full ESP independent-AR evidence checks and report rendering; no model execution."""
from studies._shared.structure.esp_controls import token_digest

CELLS = ((10, 1, 'auxiliary', None), (30, 1, 'primary', 1.63),
         (60, 1, 'primary', 1.74), (60, 2, 'control', 1.74))


def native_rows(out, prompts, spec):
    if not prompts or len(out['rows']) != len(prompts) or len(out['evidence']) != len(prompts):
        raise ValueError('full AR readout requires every fixed prompt row and token record')
    rows, audits = [], []
    for prompt, row, evidence in zip(prompts, out['rows'], out['evidence']):
        for field in ('committed', 'decode_calls', 'accepted'):
            value, recorded = row[field], evidence['stats'][field]
            if type(value) is not int or type(recorded) is not int or value < 0 or recorded < 0 or value != recorded:
                raise ValueError('full AR row counts differ from the saved native statistics')
        if (prompt['input_ids'] != evidence['input_ids'] or prompt['category'] != row['category']
                or row['spec_time'] != 0 or row['ar_time'] != 0
                or evidence['stats']['block_complexity'] != spec['block_complexity']):
            raise ValueError('full AR prompt, category, timing or block budget differs')
        output, ar = evidence['output_ids'], evidence['ar_ids']
        if not isinstance(output, list) or not isinstance(ar, list) or not output or not ar:
            raise ValueError('full AR requires saved independent AR and speculative token sequences')
        if output[:len(prompt['input_ids'])] != prompt['input_ids'] or ar[:len(prompt['input_ids'])] != prompt['input_ids']:
            raise ValueError('saved outputs do not share the original prompt prefix')
        exact = output == ar
        if bool(row['exact_match']) != exact:
            raise ValueError('recorded full AR exact flag differs from saved output tokens')
        rows.append(dict(row, question_id=prompt['question_id'], exact=exact))
        first = (next((index for index, (a, b) in enumerate(zip(output, ar)) if a != b), min(len(output), len(ar)))
                 - len(prompt['input_ids'])) if not exact else None
        audits.append({'question_id': prompt['question_id'], 'output_sha256': token_digest(output),
                       'AR_sha256': token_digest(ar), 'first_new_difference_index': first})
    return rows, audits


def render_report(body, filename, input_links):
    cells = body['cells']
    n = body['fixed_prompts']
    lines = ['# ESP Table 1 BF16 完整独立 AR 协议读件', '',
        f"{n} prompts; observed={body['all_cells_observed']}, token quality passed={body['quality_passed']}.",
        'BC10 为 auxiliary，BC30/1 与 BC60/1 为预声明 primary，BC60/2 为 control；不按结果选择 mask。', '',
        '| BC/masks | role | exact/n | pooled τ | τ − target | prompt mean τ | mean − target | category mean τ | category − target |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for cell in cells:
        metrics = ('pooled_tau', 'pooled_tau_target_delta', 'prompt_mean_tau', 'prompt_mean_tau_target_delta',
                   'category_mean_pooled_tau', 'category_mean_pooled_tau_target_delta')
        values = ['n/a' if key.endswith('_target_delta') and cell['target'] is None else 'pending'
                  if cell.get(key) is None else f'{cell[key]:.6f}' for key in metrics]
        exact = 'pending' if cell['pending'] else f"{cell['exact_count']}/{cell['n']}"
        lines.append(f"| {cell['BC']}/{cell['num_masks']} | {cell['role']} | {exact} | " + ' | '.join(values) + ' |')
    lines += ['', '目标 BC30=1.63，BC60 双方均相对 1.74 列差值。独立 AR 失配格的 τ 仅作诊断，不能算原 full 协议复现成功；argmax 审计不能替代本协议的 AR exact 门。', '',
        'pooled τ=Σcommitted/Σdecode_calls；prompt mean 为逐条 committed/max(decode_calls,1) 等权；category mean 先逐原 category pooled，再对类别等权。main_calls 含每条一次 prefill，τ 不含 prefill。', '',
        '动态树剪枝、BF16/T0/256/seed123、无测速与预热；Eq5 mean mask λ=.1 更新，与后续 4B frozen mean 不混同。', '',
        '完整负结果可以完成读出，quality_passed=false 不改为通过；缺格仍为 pending。旧 Run、来源、研究计划与验收身份不重签。', '',
        '原冻结计划本次未提供独立固定 Git 身份；来源保留未知，不据此恢复执行。', '',
        f'逐 prompt 计数和 token 摘要见 [{filename}]({filename})。', '',
        '实际输入：' + '；'.join(f'[{label}](<{path}>)' for label, path in input_links.items()) + '。']
    return '\n'.join(lines) + '\n'
