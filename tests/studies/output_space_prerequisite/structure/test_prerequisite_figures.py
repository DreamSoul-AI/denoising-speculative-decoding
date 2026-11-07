"""The saved q171 count figure preserves its established labels and scale."""

from studies.output_space_prerequisite.structure import plots


def test_q171_acceptance_figure_keeps_original_counts_color_order_and_axis_rule():
    data = {'study_quality_passed': True, 'question_id': 171, 'rows': [
        {'task': 'esp_mtp', 'case': case, 'raw': {'rows': [{'accepted': accepted}]}}
        for case, accepted in zip(plots.METHODS, (17, 10, 10))]}
    data['rows'].append({'task': 'ar_baseline', 'case': 'AR'})
    figure = plots.acceptance_figure(data)
    try:
        from matplotlib.colors import to_rgba
        axis, = figure.axes
        assert [bar.get_height() for bar in axis.patches] == [17, 10, 10]
        assert [bar.get_facecolor() for bar in axis.patches] == [to_rgba(color) for color in plots.COLORS]
        assert axis.get_ylim() == (0, 21.25)
        assert [label.get_text() for label in axis.get_xticklabels()] == list(plots.METHODS)
        assert axis.get_ylabel() == 'Accepted draft count (all depths)'
        assert axis.get_title() == ('Qwen3-0.6B / q171 / FP32 / seed123 / one prompt\n'
                                    'Matched two-slot chain; all outputs exactly match AR')
        figure.canvas.draw()
    finally:
        figure.clear()
