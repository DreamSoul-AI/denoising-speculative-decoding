"""Fixed saved values, labels and colors retain their historical figure meaning."""

import pytest

from studies.output_space_small_matrix.structure import plots


def test_acceptance_figure_preserves_saved_counts_colors_labels_and_limits():
    data = {'study_quality_passed': True, 'question_ids': [172, 173, 174], 'per_prompt': [
        {'question_id': qid, 'cells': {case: {'accepted': i + j} for j, case in enumerate(plots.METHODS)}}
        for i, qid in enumerate((172, 173, 174))]}
    figure = plots.acceptance_figure(data)
    try:
        from matplotlib.colors import to_rgba
        axis, = figure.axes
        assert [bar.get_height() for bar in axis.patches] == [0, 1, 2, 1, 2, 3, 2, 3, 4]
        assert [bar.get_facecolor() for bar in axis.patches] == [to_rgba(color) for color in plots.COLORS for _ in range(3)]
        assert axis.get_ylim() == (0, 7)
        assert [label.get_text() for label in axis.get_xticklabels()] == ['q172', 'q173', 'q174']
        assert axis.get_ylabel() == 'Accepted draft count (all depths)'
        assert [text.get_text() for text in axis.get_legend().get_texts()] == list(plots.METHODS)
        figure.canvas.draw()
    finally:
        figure.clear()


def test_window_figure_preserves_signed_margins_and_direction_norm_grouping():
    windows = {'quality_passed': True, 'question_id': 173, 'windows': [83, 94, 97],
        'cases': [{'case': case, 'windows': [
            {'root_position': root, 'first_slot': {'target_margin_vs_best_other': j - i}}
            for j, root in enumerate((83, 94, 97))]} for i, case in enumerate(plots.METHODS)]}
    scale = {'quality_passed': True, 'question_id': 173, 'root_position': 83, 'cells': [
        {'direction': direction, 'norm_source': norm, 'first_slot': {'target_margin_vs_best_other': value}}
        for direction, norm, value in [('output', 'output', -5.5), ('input', 'output', .5),
                                      ('output', 'input', -5.6), ('input', 'input', .6)]]}
    figure = plots.windows_figure(windows, scale)
    try:
        from matplotlib.colors import to_rgba
        left, right = figure.axes
        assert [bar.get_height() for bar in left.patches] == [0, 1, 2, -1, 0, 1, -2, -1, 0]
        assert [bar.get_height() for bar in right.patches] == pytest.approx([.6, -5.6, .5, -5.5])
        assert [bar.get_facecolor() for bar in right.patches] == [to_rgba(color) for color in ('#5f8c88', '#5f8c88', '#d2a656', '#d2a656')]
        assert all(axis.get_ylim() == (-6.4, 2.7) for axis in figure.axes)
        assert [label.get_text() for label in left.get_xticklabels()] == ['root83', 'root94', 'root97']
        assert [label.get_text() for label in right.get_xticklabels()] == ['input direction', 'output direction']
        assert [text.get_text() for text in right.get_legend().get_texts()] == ['norm .191', 'norm .925']
        figure.canvas.draw()
    finally:
        figure.clear()
