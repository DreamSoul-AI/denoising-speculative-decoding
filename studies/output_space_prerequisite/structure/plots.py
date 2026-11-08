"""The fixed q171 acceptance figure, rendered from saved scalar evidence only."""

import math

METHODS = ('frozen_mean_chain', 'output_order0', 'output_order1')
COLORS = ('#3868a8', '#c57736', '#8f5a9a')


def acceptance_figure(data):
    """Preserve the original values, color order, title and max-count axis rule."""
    if data.get('study_quality_passed') is not True or data.get('question_id') != 171:
        raise ValueError('acceptance plot requires the passing fixed q171 report')
    rows = [row for row in data['rows'] if row['task'] == 'esp_mtp']
    if ([row['case'] for row in rows] != list(METHODS)
            or any(len(row['raw']['rows']) != 1 for row in rows)):
        raise ValueError('acceptance plot requires the three fixed one-prompt cases')
    accepted = [row['raw']['rows'][0]['accepted'] for row in rows]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in accepted):
        raise ValueError('saved acceptance counts must be finite nonnegative numbers')
    import torch  # noqa: F401 - Windows torch-first native DLL order
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 4.5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    bars = axis.bar([row['case'] for row in rows], accepted, color=COLORS, width=.55)
    axis.bar_label(bars, padding=4)
    axis.set_ylim(0, max(accepted) * 1.25)
    axis.set_ylabel('Accepted draft count (all depths)')
    axis.set_title('Qwen3-0.6B / q171 / FP32 / seed123 / one prompt\nMatched two-slot chain; all outputs exactly match AR')
    figure.tight_layout()
    return figure
