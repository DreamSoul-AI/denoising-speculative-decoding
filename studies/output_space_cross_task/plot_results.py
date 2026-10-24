"""Plot actual native counts and the separately declared q241 count contrast.

    python studies/output_space_cross_task/plot_results.py
"""

import hashlib
import json
from pathlib import Path


def main():
    import torch  # noqa: F401 - Windows torch-first native DLL order
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    study = Path(__file__).resolve().parent
    native = json.loads((study / 'docs/execution_v1.json').read_text())
    canonical = json.loads((study / 'docs/canonical_analysis_v1.json').read_text())
    assert native['study_quality_passed'] and canonical['quality_passed']
    assert native['question_ids'] == [241, 242, 243, 244]
    rows = {r['case']: r for r in native['runs']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, case in enumerate(('frozen_mean_chain', 'output_order0', 'output_order1')):
        bars = axes[0].bar([i + (j - 1) * .25 for i in range(4)],
            [e['stats']['accepted'] for e in rows[case]['raw']['evidence']], width=.23, label=case)
        axes[0].bar_label(bars, padding=3)
    axes[0].set_xticks(range(4), ['q241', 'q242', 'q243', 'q244'])
    axes[0].set_ylabel('Accepted drafts (before budget/EOS trimming)')
    axes[0].set_ylim(0, 40)
    axes[0].set_title('Four fixed summarization prompts')
    axes[0].legend(fontsize=8)
    labels = ['native_self', 'native_graft', 'canonical_self', 'canonical_graft']
    bars = axes[1].bar(range(4), [canonical['accepted'][k] for k in labels],
                       color=['#3868a8', '#c57736', '#3868a8', '#c57736'])
    axes[1].bar_label(bars, padding=3)
    axes[1].set_xticks(range(4), ['Native\nself', 'Native\ngraft', 'Canonical\nself', 'Canonical\ngraft'])
    axes[1].set_ylim(0, 18)
    axes[1].set_title('q241 separately frozen follow-up\nEqual totals still have different semantic paths')
    fig.suptitle('Qwen3-0.6B / FP32 / T0 / fixed two-slot chain / all outputs AR-identical', fontsize=11)
    fig.tight_layout()
    folder = study / 'docs/figures'
    folder.mkdir(exist_ok=True)
    target = folder / 'acceptance_v1.png'
    assert not target.exists(), 'preserve existing figure'
    fig.savefig(target, dpi=170)
    plt.close(fig)
    source = {'native_sha256': hashlib.sha256((study / 'docs/execution_v1.json').read_bytes()).hexdigest(),
        'canonical_analysis_sha256': hashlib.sha256((study / 'docs/canonical_analysis_v1.json').read_bytes()).hexdigest(),
        'plot_tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figure_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'no_new_model_forwards': True}
    (folder / 'source_v1.json').write_text(json.dumps(source, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
