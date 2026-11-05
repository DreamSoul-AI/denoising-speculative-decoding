"""The paired count rendering retains signed bars and the declared historical interpretation."""
from studies._shared.structure import paired_report


def test_paired_count_figure_preserves_signed_differences_and_no_timing_label():
    data = {'quality_passed': True, 'decoding_timing_disabled': True, 'n': 2, 'question_ids': [1, 2], 'paired_rows': [
        {'question_id': qid, 'cells': {'input': {'accepted': 5, 'model_calls': 10},
         'zero': {'accepted': 7, 'model_calls': 9}, 'one': {'accepted': 3, 'model_calls': 11}}} for qid in (1, 2)]}
    figure = paired_report.paired_counts_figure(data)
    try:
        left, right = figure.axes
        assert [bar.get_height() for bar in left.patches] == [2, 2, -2, -2]
        assert [bar.get_height() for bar in right.patches] == [-1, -1, 1, 1]
        assert 'no timing' in figure._suptitle.get_text()
        figure.canvas.draw()
    finally:
        figure.clear()
