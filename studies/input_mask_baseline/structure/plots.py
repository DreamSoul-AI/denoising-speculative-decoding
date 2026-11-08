"""Render the fixed q171 input-mask counts and saved cosine sums/counts."""

import math

MASKS = ('frozen_mean', 'gamma_last')
COLORS = ('#3868a8', '#c57736')


def baseline_figure(data):
    """Preserve the original bars, scalar mean formula, zero-count omission and labels."""
    if data.get('study_quality_passed') is not True or data.get('question_id') != 171:
        raise ValueError('input-mask figure requires the passing fixed q171 report')
    esp = [row for row in data['rows'] if row['task'] == 'esp_mtp']
    probes = [row for row in data['rows'] if row['task'] == 'alignment_probe']
    if ([row['mask'] for row in esp] != list(MASKS) or len(probes) != 1
            or any(len(row['raw']['rows']) != 1 for row in esp)):
        raise ValueError('input-mask figure requires its two one-prompt baselines and single alignment probe')
    accepted = [row['raw']['rows'][0]['accepted'] for row in esp]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in accepted):
        raise ValueError('saved acceptance counts must be finite nonnegative numbers')
    counts, sums = probes[0]['raw']['counts'], probes[0]['raw']['sums']
    if any(type(counts[key]) is not int or counts[key] < 0 for key in ('accept', 'reject')):
        raise ValueError('saved observation counts must be nonnegative integers')
    layers = list(range(1, len(sums['accept']) + 1))
    means = {}
    for key in ('accept', 'reject'):
        n = counts[key]
        if n:
            if (len(sums[key]) != len(layers)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in sums[key])):
                raise ValueError('observed cosine sums must be finite and aligned by layer')
            means[key] = [value / n for value in sums[key]]
    import torch  # noqa: F401 - Windows torch-first native import order
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4.2))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    bars = axes[0].bar([row['mask'] for row in esp], accepted, color=COLORS, width=.55)
    axes[0].bar_label(bars, padding=4)
    axes[0].set_ylabel('Accepted draft count (all depths)')
    axes[0].set_ylim(0, max(accepted) * 1.25)
    axes[0].set_title('Two input-mask baselines; exact-match = 1')
    for key, color in zip(('accept', 'reject'), COLORS):
        if key in means:
            axes[1].plot(layers, means[key], label=f'{key}, observations={counts[key]}', color=color)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean cosine')
    axes[1].set_title('Four hindsight observations; not Figure 2 reproduction')
    axes[1].legend(loc='best')
    figure.suptitle('Qwen3-0.6B / SpecBench q171 / FP32 eager / seed 123 / one prompt', fontsize=11)
    figure.tight_layout()
    return figure
