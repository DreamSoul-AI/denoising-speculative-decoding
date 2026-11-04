"""Plot the saved input-mask prerequisite evidence; does not execute any Run.

    python studies/input_mask_baseline/plot_results.py
"""

import json
from pathlib import Path


def main():
    import torch  # noqa: F401 - Windows torch-first native import order
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    study = Path(__file__).resolve().parent
    data = json.loads((study / 'docs/v4_execution.json').read_text(encoding='utf-8'))
    assert data['study_quality_passed'] and data['question_id'] == 171
    esp = [r for r in data['rows'] if r['task'] == 'esp_mtp']
    probe = next(r for r in data['rows'] if r['task'] == 'alignment_probe')
    counts = probe['raw']['counts']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = [r['mask'] for r in esp]
    values = [r['raw']['rows'][0]['accepted'] for r in esp]
    bars = axes[0].bar(labels, values, color=['#3868a8', '#c57736'], width=.55)
    axes[0].bar_label(bars, padding=4)
    axes[0].set_ylabel('Accepted draft count (all depths)')
    axes[0].set_ylim(0, max(values) * 1.25)
    axes[0].set_title('Two input-mask baselines; exact-match = 1')
    layers = list(range(1, len(probe['raw']['sums']['accept']) + 1))
    for key, color in [('accept', '#3868a8'), ('reject', '#c57736')]:
        n = counts[key]
        if n:
            values = [value / n for value in probe['raw']['sums'][key]]
            axes[1].plot(layers, values, label=f'{key}, observations={n}', color=color)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean cosine')
    axes[1].set_title('Four hindsight observations; not Figure 2 reproduction')
    axes[1].legend(loc='best')
    fig.suptitle('Qwen3-0.6B / SpecBench q171 / FP32 eager / seed 123 / one prompt', fontsize=11)
    fig.tight_layout()
    output = study / 'docs/figures/input_mask_baseline_v4.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
