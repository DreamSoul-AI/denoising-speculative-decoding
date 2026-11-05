"""The fixed native q241–244 and separately declared canonical q241 count figure."""


def acceptance_figure(native, canonical):
    if (native.get('study_quality_passed') is not True or canonical.get('quality_passed') is not True
            or native.get('question_ids') != [241, 242, 243, 244] or canonical.get('question_id') != 241):
        raise ValueError('cross-task figure requires the original passing fixed prompt reports')
    import torch  # noqa: F401 - Windows torch-first native DLL order
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rows = {row['case']: row for row in native['runs']}
    figure = Figure(figsize=(11, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for j, case in enumerate(('frozen_mean_chain', 'output_order0', 'output_order1')):
        values = [evidence['stats']['accepted'] for evidence in rows[case]['raw']['evidence']]
        if len(values) != 4:
            raise ValueError('cross-task figure requires four observations per native method')
        bars = axes[0].bar([i + (j - 1) * .25 for i in range(4)], values, width=.23, label=case)
        axes[0].bar_label(bars, padding=3)
    axes[0].set_xticks(range(4), ['q241', 'q242', 'q243', 'q244'])
    axes[0].set_ylabel('Accepted drafts (before budget/EOS trimming)')
    axes[0].set_ylim(0, 40)
    axes[0].set_title('Four fixed summarization prompts')
    axes[0].legend(fontsize=8)
    labels = ['native_self', 'native_graft', 'canonical_self', 'canonical_graft']
    bars = axes[1].bar(range(4), [canonical['accepted'][label] for label in labels],
                       color=['#3868a8', '#c57736', '#3868a8', '#c57736'])
    axes[1].bar_label(bars, padding=3)
    axes[1].set_xticks(range(4), ['Native\nself', 'Native\ngraft', 'Canonical\nself', 'Canonical\ngraft'])
    axes[1].set_ylim(0, 18)
    axes[1].set_title('q241 separately frozen follow-up\nEqual totals still have different semantic paths')
    figure.suptitle('Qwen3-0.6B / FP32 / T0 / fixed two-slot chain / all outputs AR-identical', fontsize=11)
    figure.tight_layout()
    return figure
