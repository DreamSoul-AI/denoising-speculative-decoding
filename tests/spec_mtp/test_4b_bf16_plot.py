import importlib.util
from itertools import combinations
from pathlib import Path
from PIL import Image
import pytest

STUDY = Path(__file__).resolve().parents[2]/'studies/qwen3_4b_bf16_five_methods'


@pytest.fixture
def plotter():
    spec = importlib.util.spec_from_file_location('bf16_paired_plot', STUDY/'plot_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_body(plotter):
    cells = {'AR': {'native_status': 'succeeded'}}
    cells.update({f'{bc}/{m}': {'native_status': 'succeeded'} for bc in (30, 60) for m in plotter.LABELS})
    pairs = [dict(first=f'{bc}/{a}', second=f'{bc}/{b}', n=480,
        accepted=dict(win=300, tie=30, loss=150), calls=dict(win=80, tie=20, loss=380))
        for bc in (30, 60) for a, b in combinations(plotter.LABELS, 2)]
    return dict(complete=True, cells=cells, pairs=pairs,
                scope='SYNTHETIC layout fixture — not model evidence')


def test_row_vs_column_preserves_all_wins_ties_losses_without_double_inverting_calls(plotter):
    body = fixture_body(plotter)
    grid = plotter.pair_matrix(body, 30, 'accepted')
    assert grid[1][0] == (300,30,150) and grid[0][1] == (150,30,300)
    assert all(grid[i][i] is None for i in range(5))
    calls = plotter.pair_matrix(body, 60, 'calls')
    assert calls[1][0] == (80,20,380) and calls[0][1] == (380,20,80)


@pytest.mark.parametrize('failure', ['pending', 'missing_pair', 'duplicate_pair', 'bad_count', 'failed_cell', 'missing_cell'])
def test_partial_or_inconsistent_results_cannot_be_plotted(plotter, tmp_path, failure):
    body = fixture_body(plotter)
    if failure == 'pending': body['complete'] = False
    if failure == 'missing_pair': body['pairs'].pop()
    if failure == 'duplicate_pair': body['pairs'][0] = body['pairs'][1]
    if failure == 'bad_count': body['pairs'][0]['accepted']['win'] = 301
    if failure == 'failed_cell': body['cells']['30/input_first']['native_status'] = 'failed'
    if failure == 'missing_cell': body['cells'].pop('60/input_first')
    path = tmp_path/'paired.png'
    with pytest.raises(AssertionError):
        plotter.write_plot(body, path)
    assert not path.exists()


def test_synthetic_fixture_renders_png(plotter, tmp_path):
    path = tmp_path/'paired.png'
    plotter.write_plot(fixture_body(plotter), path)
    with Image.open(path) as image:
        assert image.format == 'PNG' and image.width > 1500 and image.height > 1200
