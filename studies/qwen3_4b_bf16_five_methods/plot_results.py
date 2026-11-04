"""Plot the complete paired counts; no model evaluation or timing."""
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'frozen_mean': 'Frozen mean', 'input_zero': 'Input 0', 'input_first': 'Input 1',
          'output_zero': 'Output 0', 'output_first': 'Output 1'}


def pair_matrix(body, budget, metric):
    assert body['complete'] and metric in ('accepted', 'calls') and budget in (30, 60)
    expected_cells = {'AR'} | {f'{bc}/{m}' for bc in (30, 60) for m in LABELS}
    assert set(body['cells']) == expected_cells
    assert all(c['native_status'] == 'succeeded' for c in body['cells'].values())
    assert len(body['pairs']) == 20
    pairs = [p for p in body['pairs'] if p['first'].startswith(f'{budget}/')]
    lookup = {(p['first'], p['second']): p for p in pairs}
    expected = {(f'{budget}/{a}', f'{budget}/{b}') for a, b in combinations(LABELS, 2)}
    assert len(pairs) == len(lookup) == 10 and set(lookup) == expected
    names = list(LABELS)
    grid = [[None for _ in names] for _ in names]
    for (first, second), pair in lookup.items():
        counts = pair[metric]
        assert pair['n'] == 480 and set(counts) == {'win', 'tie', 'loss'}
        assert all(type(v) is int and v >= 0 for v in counts.values()) and sum(counts.values()) == 480
        w, t, loss = (counts[k] for k in ('win', 'tie', 'loss'))
        i, j = names.index(first.split('/')[1]), names.index(second.split('/')[1])
        grid[j][i], grid[i][j] = (w, t, loss), (loss, t, w)
    return grid


def write_plot(body, path):
    # Validate every panel before opening or overwriting an image.
    panels = {(bc, metric): pair_matrix(body, bc, metric) for bc in (30, 60) for metric in ('accepted', 'calls')}
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout='constrained')
    try:
        for row, budget in enumerate((30, 60)):
            for col, metric in enumerate(('accepted', 'calls')):
                ax, grid = axes[row, col], panels[budget, metric]
                net = [[v[0]-v[2] if v is not None else float('nan') for v in line] for line in grid]
                image = ax.imshow(net, cmap='RdBu', vmin=-480, vmax=480)
                for i, line in enumerate(grid):
                    for j, counts in enumerate(line):
                        text = '—' if counts is None else '/'.join(str(v) for v in counts)
                        color = 'white' if counts is not None and abs(counts[0]-counts[2]) > 250 else 'black'
                        ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)
                rule = 'higher wins' if metric == 'accepted' else 'lower wins'
                ax.set_title(f'BC{budget}: {metric} ({rule})')
                ax.set_xticks(range(5), LABELS.values(), rotation=25, ha='right')
                ax.set_yticks(range(5), LABELS.values())
                ax.set_xlabel('Reference (column)')
                ax.set_ylabel('Compared method (row)')
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75,
                     label='Net winning prompts (W − L); positive favors row')
        fig.suptitle('Qwen3-4B paired outcomes\n'+body['scope'])
        fig.supxlabel('Cells = W/T/L out of 480 prompts; all losses retained; counts only (no timing).', fontsize=10)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
