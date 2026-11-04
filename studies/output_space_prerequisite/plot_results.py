"""Plot saved output-space prerequisite counts without running an experiment.

    python studies/output_space_prerequisite/plot_results.py
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
    assert data['study_quality_passed'] and data['question_id'] == 171
    rows = [r for r in data['rows'] if r['task'] == 'esp_mtp']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    accepted = [r['raw']['rows'][0]['accepted'] for r in rows]
    bars = ax.bar([r['case'] for r in rows], accepted, color=['#3868a8', '#c57736', '#8f5a9a'], width=.55)
    ax.bar_label(bars, padding=4)
    ax.set_ylim(0, max(accepted) * 1.25)
    ax.set_ylabel('Accepted draft count (all depths)')
    ax.set_title('Qwen3-0.6B / q171 / FP32 / seed123 / one prompt\nMatched two-slot chain; all outputs exactly match AR')
    fig.tight_layout()
    target = study / 'docs/figures/acceptance_v1.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
