"""Plot the preserved delivery CPU evidence; no Run is executed or updated."""

import json
from pathlib import Path


def main():
    import torch  # noqa: F401 - preserve the Windows torch-first DLL import order
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    study = Path(__file__).resolve().parent
    reference = json.loads((study / 'docs/delivery_validation.json').read_text(encoding='utf-8'))
    rows = reference['cpu']
    variants = ['esp_static', 'esp_dynamic', 'ema_velocity']
    assert reference['implementation_commit'] == '934dac3fd3b1662664ebf71d8317b49183e22f0b'
    assert [row['variant'] for row in rows] == variants
    assert all(row['exact_match_rate'] == 1 and row['prompts'] == 4 and row['seed'] == 123 for row in rows)
    assert all(row['device'] == 'cpu' and row['dtype'] == 'torch.float32' and row['backend'] == 'eager'
               for row in rows)
    accepted = [row['accepted'] for row in rows]
    depth_one = [row['accept_d1'] for row in rows]
    assert accepted == [28, 31, 8] and depth_one == [27, 31, 8]

    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    x = list(range(len(rows)))
    for offset, values, label, color in (
        (-0.19, accepted, 'All-depth accepted drafts', '#2868a0'),
        (0.19, depth_one, 'Depth-1 accepted drafts', '#81b29a'),
    ):
        bars = ax.bar([value + offset for value in x], values, width=0.36, label=label, color=color)
        ax.bar_label(bars, padding=3)
    ax.set_xticks(x, variants)
    ax.set_ylabel('Draft tokens accepted (4 prompts)')
    ax.set_ylim(0, 39)
    ax.set_title('Historical CPU tiny validation — exact match = 1 for every variant')
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0, -0.20, 'Source: delivery_validation.json / 934dac3 / n=1 seed per variant\n'
            'FP32, eager, seed=123, T=0, 32-token budget; counts before budget/EOS truncation',
            transform=ax.transAxes, fontsize=8)
    output = study / 'docs/figures/tiny_acceptance.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
