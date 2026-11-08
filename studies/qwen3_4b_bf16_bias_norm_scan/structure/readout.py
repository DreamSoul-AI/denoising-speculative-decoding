"""Pure norm-scan saved counts and first-block pairing; no native admission or output."""
import hashlib
import json
from typing import Any

if not __debug__:
    raise RuntimeError('saved norm readout requires Python assertions enabled')

def depth_rows(raw, prompts):
    assert len(raw) == len(prompts)
    rows = []
    for item, prompt in zip(raw, prompts):
        assert item['input_ids'] == prompt['input_ids']
        stats = item['stats']
        assert stats['prefill_calls'] == 1 and len(stats['trace']) == stats['decode_calls']
        prefix = item['output_ids'][:len(prompt['input_ids']) + 1]
        assert len(prefix) == len(prompt['input_ids']) + 1
        rows.append({'question_id': prompt['question_id'],
            'initial_prefix_sha256': hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
            'attempt_by_depth': stats['attempt_by_depth'], 'accept_by_depth': stats['accept_by_depth'],
            'first_block_accepted': len(stats['trace'][0]['accepted_nodes']) if stats['trace'] else None})
    return rows


def first_block_compare(first, second):
    assert len(first) == len(second)
    result: dict[str, Any] = {'total': len(first), 'same_prefix_with_block': 0, 'different_prefix': 0,
              'missing_block': 0, 'accepted': {'win': 0, 'tie': 0, 'loss': 0}}
    for a, b in zip(first, second):
        assert a['question_id'] == b['question_id']
        if a['first_block_accepted'] is None or b['first_block_accepted'] is None:
            result['missing_block'] += 1
        elif a['initial_prefix_sha256'] != b['initial_prefix_sha256']:
            result['different_prefix'] += 1
        else:
            result['same_prefix_with_block'] += 1
            delta = b['first_block_accepted'] - a['first_block_accepted']
            result['accepted']['win' if delta > 0 else 'loss' if delta < 0 else 'tie'] += 1
    assert sum(result['accepted'].values()) == result['same_prefix_with_block']
    return result


def summarize_cells(cells):
    from studies._shared.structure import geometry_4b as directions
    ready = all('rows' in cell for cell in cells.values())
    pairs = []
    for name, cell in cells.items():
        if 'rows' not in cell:
            continue
        cell['totals']['category_mean_pooled_tau'] = directions.category_mean(cell['rows'])
        if name == 'zero/0':
            continue
        mode = name.split('/')[0]
        for base in ('zero/0', mode+'/1'):
            pairs.append(dict(first=base, second=name,
                first_block_same_prefix=first_block_compare(cells[base]['depth_rows'], cell['depth_rows']),
                **directions.compare(cells[base]['rows'], cell['rows'])))
    means = {}
    for scale in (.5, 1., 2.):
        rows = [cells[f'random{i}/{scale:g}'] for i in range(3)]
        if all('rows' in cell for cell in rows):
            means[f'{scale:g}'] = sum(cell['totals']['pooled_tau'] for cell in rows)/3
    return ready, pairs, means


def verify_reused_published(proof, original):
    from studies._shared.structure import geometry_4b as directions
    assert not proof['complete'] and proof['reused_only'] and not proof['new_cells_inspected']
    assert proof['fixed_prompts'] == 480 and len(proof['expected_new_cases']) == 8
    assert set(proof['cells']) == {'zero/0', 'template/1', 'random0/1', 'random1/1', 'random2/1'}
    source = {}
    for name, cell in proof['cells'].items():
        actual = original['cells'][name.split('/')[0]]
        assert cell['totals'] == actual['totals'] and cell['files_sha256'] == actual['files_sha256']
        assert cell['run_id'] == actual['run_id']
        assert cell['raw_argmax_checked_tokens'] == actual['totals']['argmax_checked_tokens']
        assert cell['raw_files_rechecked'] == len(cell['files_sha256']) == 6
        source[name] = actual
    expected = []
    for base in ('zero/0', 'template/1'):
        for name in source:
            if name != base:
                pair = directions.compare(source[base]['rows'], source[name]['rows'])
                pair.pop('rows')
                expected.append(dict(first=base, second=name, **pair))
    assert len(expected) == len(proof['paired_comparisons'])
    for pair, saved in zip(expected, proof['paired_comparisons']):
        assert pair == {key: value for key, value in saved.items() if key != 'first_block_same_prefix'}
        first = saved['first_block_same_prefix']
        assert first['total'] == first['same_prefix_with_block'] + first['different_prefix'] + first['missing_block'] == 480
        assert sum(first['accepted'].values()) == first['same_prefix_with_block']
    assert sum(cell['raw_argmax_checked_tokens'] for cell in proof['cells'].values()) == proof['raw_argmax_checked_tokens']
    return {'cells_checked': len(source), 'pairs_checked': len(expected), 'raw_files_revalidated': False}


def norm_figure(cells, means):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [0., .5, 1., 2.]
    zero = cells['zero/0']['totals']['pooled_tau']
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    try:
        for mode in ('template', 'random0', 'random1', 'random2'):
            y = [zero] + [cells[f'{mode}/{scale:g}']['totals']['pooled_tau'] for scale in x[1:]]
            ax.plot(x, y, marker='o', label=mode)
        ax.plot(x, [zero] + [means[f'{scale:g}'] for scale in x[1:]], 'k--', label='three-random mean')
        ax.set(xlabel='Native step_scale / original direction norm multiplier',
               ylabel='Pooled committed / decode calls', xticks=x, title='Fixed480, Qwen3-4B BF16/T0, BC30')
        ax.legend()
        fig.supxlabel('0 and1 reuse frozen evidence; only0.5 and2 are new. Output lengths may differ; no speed claim.')
        return fig
    except BaseException:
        plt.close(fig)
        raise
