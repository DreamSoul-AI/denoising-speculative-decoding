"""Plot saved per-prompt acceptance counts; does not execute a Run.

    python studies/output_space_small_matrix/plot_results.py
"""

import json
from pathlib import Path


def main():
    import torch  # noqa: F401 - Windows torch-first native DLL order
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    study = Path(__file__).resolve().parent
    data = json.loads((study / 'docs/execution_v1.json').read_text(encoding='utf-8'))
    assert data['study_quality_passed'] and data['question_ids'] == [172, 173, 174]
    methods = ['frozen_mean_chain', 'output_order0', 'output_order1']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, (case, color) in enumerate(zip(methods, ['#3868a8', '#c57736', '#8f5a9a'])):
        x = [i + (j - 1) * .25 for i in range(3)]
        values = [p['cells'][case]['accepted'] for p in data['per_prompt']]
        bars = ax.bar(x, values, width=.23, label=case, color=color)
        ax.bar_label(bars, padding=3)
    ax.set_xticks(range(3), ['q172', 'q173', 'q174'])
    ax.set_ylim(0, 7)
    ax.set_ylabel('Accepted draft count (all depths)')
    ax.set_title('Qwen3-0.6B / FP32 / T0 / three fixed prompts\nMatched two-slot chain; all outputs exactly match AR')
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = study / 'docs/figures/acceptance_v1.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
