"""Native and canonical bars retain their distinct historical meanings."""
from studies.output_space_cross_task.structure.plots import acceptance_figure


def test_cross_task_figure_keeps_native_and_canonical_values_axes_and_colors():
    native = {'study_quality_passed': True, 'question_ids': [241, 242, 243, 244], 'runs': [
        {'case': case, 'raw': {'evidence': [{'stats': {'accepted': i + j}} for i in range(4)]}}
        for j, case in enumerate(('frozen_mean_chain', 'output_order0', 'output_order1'))]}
    canonical = {'quality_passed': True, 'question_id': 241,
        'accepted': dict(zip(('native_self', 'native_graft', 'canonical_self', 'canonical_graft'), (11, 12, 14, 14)))}
    figure = acceptance_figure(native, canonical)
    try:
        left, right = figure.axes
        assert [bar.get_height() for bar in right.patches] == [11, 12, 14, 14]
        assert left.get_ylim() == (0, 40) and right.get_ylim() == (0, 18)
        assert 'different semantic paths' in right.get_title()
        figure.canvas.draw()
    finally:
        figure.clear()
