"""Render the fixed saved acceptance and window metrics without reading or writing inputs."""

import math

METHODS = ('frozen_mean_chain', 'output_order0', 'output_order1')
COLORS = ('#3868a8', '#c57736', '#8f5a9a')


def _figure(size, columns=1):
    import torch  # noqa: F401 - Windows torch-first native DLL order
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=size)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(1, columns)


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('saved plot values must be finite numbers')
    return value


def acceptance_figure(data):
    """Keep the original three-prompt counts, grouping, colors, title and fixed axes."""
    if (data.get('study_quality_passed') is not True or data.get('question_ids') != [172, 173, 174]
            or [row.get('question_id') for row in data['per_prompt']] != [172, 173, 174]):
        raise ValueError('acceptance plot requires the passing fixed three-prompt report')
    values = [[_finite(row['cells'][case]['accepted']) for row in data['per_prompt']] for case in METHODS]
    figure, ax = _figure((7, 4.5))
    for j, (case, color, counts) in enumerate(zip(METHODS, COLORS, values)):
        x = [i + (j - 1) * .25 for i in range(3)]
        bars = ax.bar(x, counts, width=.23, label=case, color=color)
        ax.bar_label(bars, padding=3)
    ax.set_xticks(range(3), ['q172', 'q173', 'q174'])
    ax.set_ylim(0, 7)
    ax.set_ylabel('Accepted draft count (all depths)')
    ax.set_title('Qwen3-0.6B / FP32 / T0 / three fixed prompts\nMatched two-slot chain; all outputs exactly match AR')
    ax.legend(fontsize=8)
    figure.tight_layout()
    return figure


def windows_figure(windows, scale):
    """Keep the recorded margins and original fixed 2 x 2 scale comparison."""
    if (windows.get('quality_passed') is not True or scale.get('quality_passed') is not True
            or windows.get('question_id') != 173 or scale.get('question_id') != 173
            or windows.get('windows') != [83, 94, 97] or scale.get('root_position') != 83
            or [case.get('case') for case in windows['cases']] != list(METHODS)
            or any([row.get('root_position') for row in case['windows']] != [83, 94, 97]
                   for case in windows['cases'])):
        raise ValueError('window plot requires the passing fixed q173 windows and root83 scale reports')
    margins = [[_finite(row['first_slot']['target_margin_vs_best_other']) for row in case['windows']]
               for case in windows['cases']]
    cells = {(cell['direction'], cell['norm_source']): cell for cell in scale['cells']}
    if len(scale['cells']) != 4 or set(cells) != {(direction, norm) for direction in ('input', 'output')
                                               for norm in ('input', 'output')}:
        raise ValueError('scale plot requires exactly the four recorded direction/norm cells')
    scale_values = [[_finite(cells[direction, norm]['first_slot']['target_margin_vs_best_other'])
                     for direction in ('input', 'output')] for norm in ('input', 'output')]
    figure, axes = _figure((12, 4.5), 2)
    for i, (case, color, values) in enumerate(zip(windows['cases'], COLORS, margins)):
        bars = axes[0].bar([x + (i - 1) * .25 for x in range(3)], values,
                           width=.23, label=case['case'], color=color)
        axes[0].bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
    axes[0].set_xticks(range(3), ['root83', 'root94', 'root97'])
    axes[0].set_title('Actual preceding seed-mask logits')
    axes[0].legend(fontsize=8, loc='upper left')
    for i, (norm, color, values) in enumerate(zip(('input', 'output'), ('#5f8c88', '#d2a656'), scale_values)):
        bars = axes[1].bar([x + (i - .5) * .28 for x in range(2)], values, width=.26,
                           label='norm .191' if norm == 'input' else 'norm .925', color=color)
        axes[1].bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
    axes[1].set_xticks(range(2), ['input direction', 'output direction'])
    axes[1].set_title('Root83: fixed 2 x 2 isolated-prefix queries')
    axes[1].legend(fontsize=8, loc='upper left')
    for ax in axes:
        ax.axhline(0, color='#333333', linewidth=.8)
        ax.set_ylim(-6.4, 2.7)
        ax.set_ylabel('Target logit margin vs best other token')
    figure.suptitle('Qwen3-0.6B / q173 / FP32 / T0 / fixed selected windows', fontsize=11)
    figure.tight_layout()
    return figure
