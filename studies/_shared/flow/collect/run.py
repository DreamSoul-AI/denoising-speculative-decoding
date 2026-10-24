"""Derive CSV and Tracker observations from execute evidence."""
import csv
import json
from pathlib import Path
import studies._shared.structure.evaluate as esp_mtp

def run(ctx):
    """Study collect hook: derive CSV and Tracker metrics from execute evidence."""
    assets = Path(ctx.layout.assets_dir)
    raw = json.loads((assets / 'decode_observations.json').read_text(encoding='utf-8'))
    task, out = raw['task'], raw['out']
    if task == 'esp_mtp':
        out['summaries'] = esp_mtp._aggregate(out['rows'])
    elif task == 'ar_baseline':
        rows = out['rows']
        total_time = sum(row['ar_time'] for row in rows)
        out['summary'] = {'prompts': len(rows), 'tokens_per_s': sum(row['ar_new_tokens'] for row in rows) / total_time if total_time > 0 else 0.0}
    elif task == 'alignment_probe':
        import torch

        out['rows'] = [
            {'layer': i + 1, 'cos_accepted': (torch.tensor(a) / max(out['counts']['accept'], 1)).item(),
             'cos_rejected': (torch.tensor(b) / max(out['counts']['reject'], 1)).item(),
             'n_accepted': out['n_accepted'], 'n_rejected': out['n_rejected']}
            for i, (a, b) in enumerate(zip(out['sums']['accept'], out['sums']['reject']))]
    metrics = _tracker_metrics(task, out)
    tracker = ctx.state['tracker']
    tracker.append('test', values=metrics)
    tracker.save('test')
    tracker.flush_state()
    _write_assets(assets, task, out)
    ctx.state['collected']['metrics'].update({f'test_{k.lower()}': v for k, v in metrics.items()})



def _tracker_metrics(task, out):
    if task == 'alignment_probe':
        return {
            'Accepted': float(out.get('n_accepted', 0)),
            'Rejected': float(out.get('n_rejected', 0)),
        }
    if task == 'ar_baseline':
        summary = out.get('summary') or {}
        return {'TokensPerS': float(summary.get('tokens_per_s', 0.0))}
    rows = out.get('rows') or []
    totals = esp_mtp.totals(rows)
    summaries = out.get('summaries') or {}
    overall = summaries.get('overall', {})
    metrics = {
        'Accepted': float(totals['accepted']),
        'Committed': float(totals['committed']),
        'DecodeCalls': float(totals['decode_calls']),
    }
    for label, key in (('Tau', 'tau'), ('ExactMatchRate', 'exact_match_rate')):
        if overall.get(key) is not None:
            metrics[label] = float(overall[key])
    return metrics


def _write_assets(assets_dir: Path, task: str, out: dict) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    rows = out.get('rows')
    if rows:
        path = assets_dir / ('alignment_metrics.csv' if task == 'alignment_probe' else 'spec_metrics.csv')
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summaries = out.get('summaries')
    if summaries:
        flat = [{'category': k, **v} for k, v in summaries.items()]
        path = assets_dir / 'summary.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)
    summary = out.get('summary')
    if summary:
        path = assets_dir / 'summary.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
