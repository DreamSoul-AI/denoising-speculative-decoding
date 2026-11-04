"""Plot saved matched-window and fixed scale-counterfactual metrics.

    python studies/output_space_small_matrix/plot_windows.py
"""

import json
from pathlib import Path


def main():
    import torch  # noqa: F401 - Windows torch-first DLL order
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    study = Path(__file__).resolve().parent
    windows = json.loads((study / 'docs/window_diagnosis_v1.json').read_text(encoding='utf-8'))
    scale = json.loads((study / 'docs/scale_diagnosis_v1.json').read_text(encoding='utf-8'))
    assert windows['quality_passed'] and scale['quality_passed']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (case, color) in enumerate(zip(windows['cases'], ['#3868a8', '#c57736', '#8f5a9a'])):
        values = [w['first_slot']['target_margin_vs_best_other'] for w in case['windows']]
        bars = axes[0].bar([x + (i - 1) * .25 for x in range(3)], values,
                           width=.23, label=case['case'], color=color)
        axes[0].bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
    axes[0].set_xticks(range(3), ['root83', 'root94', 'root97'])
    axes[0].set_title('Actual preceding seed-mask logits')
    axes[0].legend(fontsize=8, loc='upper left')
    for i, (norm, color) in enumerate(zip(['input', 'output'], ['#5f8c88', '#d2a656'])):
        values = [next(c['first_slot']['target_margin_vs_best_other'] for c in scale['cells']
                       if c['direction'] == direction and c['norm_source'] == norm)
                  for direction in ['input', 'output']]
        bars = axes[1].bar([x + (i - .5) * .28 for x in range(2)], values, width=.26,
                           label='norm .191' if norm == 'input' else 'norm .925', color=color)
        axes[1].bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
    axes[1].set_xticks(range(2), ['input direction', 'output direction'])
    axes[1].set_title('Root83: fixed 2 x 2 isolated-prefix queries')
    axes[1].legend(fontsize=8, loc='upper left')
    for ax in axes:
        ax.axhline(0, color='#333333', linewidth=.8)
        ax.set_ylim(-6.4, 2.7)
        ax.set_ylabel('Target logit margin vs best other token')
    fig.suptitle('Qwen3-0.6B / q173 / FP32 / T0 / fixed selected windows', fontsize=11)
    fig.tight_layout()
    target = study / 'docs/figures/windows_v1.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
