"""The input-mask figure preserves saved scalar means and zero-count omission."""

from copy import deepcopy

import pytest

from studies.input_mask_baseline.structure import plots


def _report():
    return {'study_quality_passed': True, 'question_id': 171, 'rows': [
        {'task': 'esp_mtp', 'mask': mask, 'raw': {'rows': [{'accepted': 19}]}} for mask in plots.MASKS] + [
        {'task': 'alignment_probe', 'raw': {'counts': {'accept': 1, 'reject': 3},
         'sums': {'accept': [.2, .4, .6], 'reject': [1.5, 1.8, 2.1]}}}]}


def test_input_mask_figure_keeps_counts_saved_means_colors_labels_and_axes():
    report = _report()
    before = deepcopy(report)
    figure = plots.baseline_figure(report)
    try:
        from matplotlib.colors import to_rgba
        left, right = figure.axes
        assert [bar.get_height() for bar in left.patches] == [19, 19]
        assert [bar.get_facecolor() for bar in left.patches] == [to_rgba(color) for color in plots.COLORS]
        assert left.get_ylim() == (0, 23.75)
        assert [label.get_text() for label in left.get_xticklabels()] == list(plots.MASKS)
        assert right.lines[0].get_xdata().tolist() == [1, 2, 3]
        assert right.lines[0].get_ydata().tolist() == pytest.approx([.2, .4, .6])
        assert right.lines[1].get_ydata().tolist() == pytest.approx([.5, .6, .7])
        assert [line.get_color() for line in right.lines] == list(plots.COLORS)
        assert [text.get_text() for text in right.get_legend().get_texts()] == ['accept, observations=1', 'reject, observations=3']
        assert right.get_xlabel() == 'Layer' and right.get_ylabel() == 'Mean cosine'
        assert right.get_title() == 'Four hindsight observations; not Figure 2 reproduction'
        figure.canvas.draw()
        assert report == before
    finally:
        figure.clear()


def test_input_mask_figure_omits_zero_count_observations_without_dividing_by_zero():
    report = _report()
    report['rows'][2]['raw'].update(counts={'accept': 0, 'reject': 4},
                                  sums={'accept': [9., 9., 9.], 'reject': [2., 2.4, 2.8]})
    figure = plots.baseline_figure(report)
    try:
        right = figure.axes[1]
        line, = right.lines
        assert line.get_label() == 'reject, observations=4'
        assert line.get_ydata().tolist() == pytest.approx([.5, .6, .7])
        assert line.get_color() == '#c57736'
        figure.canvas.draw()
    finally:
        figure.clear()
