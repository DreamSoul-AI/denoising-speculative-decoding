"""CPU-only completed-cell accounting; never launch or retry GPU work.

Usage: python tools/audit_4b_norm_queue.py --data-root <old-checkout> --output-dir <new-dir>
Add --continuation-study studies/qwen3_4b_bf16_bias_norm_continuation --watch for the seven-cell continuation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/qwen3_4b_bf16_bias_norm_scan'
READER = STUDY / 'analyze_results.py'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_once(data_root, output_dir, continuation_study=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries = read(STUDY / 'index.json')['experiments']
    assert len(entries) == 8 and all(len(e['runs']) == 1 for e in entries)
    continuation = Path(continuation_study).resolve() if continuation_study else None
    selected = [(STUDY, e) for e in entries]
    if continuation:
        newer = read(continuation / 'index.json')['experiments']
        assert len(newer) == 7
        selected = [(STUDY, entries[0]), *[(continuation, e) for e in newer]]
    done, failed = [], []
    for cell_study, entry in selected:
        run = entry['runs'][0]['id']
        cfg = entry['factors']['algorithm.config']
        path = cell_study / 'runs' / run / 'result.json'
        result = read(path) if path.exists() else {}
        item = dict(run_id=run, case=f"{cfg['direction']}/{cfg['spec']['ema']['step_scale']:g}")
        if result.get('status') == 'succeeded':
            done.append(dict(item, result_sha256=sha(path)))
        elif result.get('status') == 'failed':
            failed.append(item)
    state = dict(complete=False, native_completed=len(done), total=8, completed=done, failed=failed,
                 no_model_forward=True, no_timing=True, audited_cells=[])
    signature = hashlib.sha256(json.dumps([done, sha(READER), sha(STUDY / 'docs/source_refs.json'), sha(continuation / 'docs/source_refs.json') if continuation else None], sort_keys=True).encode()).hexdigest()
    latest = output / 'ledger.json'
    previous = read(latest) if latest.exists() else {}
    if previous.get('signature') == signature and previous.get('failed') == failed:
        return previous
    if done and not failed:
        process = (continuation or STUDY) / 'process.json'
        if len(done) == 8 and (not process.exists() or not read(process).get('complete')):
            state['phase'] = 'waiting_for_native_study_process'
        else:
            target = output / f"{len(done):02d}-{done[-1]['run_id']}-{signature[:8]}"
            target.mkdir(parents=True, exist_ok=False)
            env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(ROOT / 'src'), CUDA_VISIBLE_DEVICES='',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
            with (target / 'audit.stdout.log').open('w') as stdout, (target / 'audit.stderr.log').open('w') as stderr:
                subprocess.run([sys.executable, str(READER), '--data-root', str(Path(data_root).resolve()),
                                '--study-dir', str(STUDY), '--output-dir', str(target),
                                *(['--continuation-study', str(continuation)] if continuation else [])], cwd=ROOT, env=env,
                               stdout=stdout, stderr=stderr, check=True)
            body = read(target / ('norm_evidence_v1.json' if len(done) == 8 else 'norm_pending.json'))
            assert body['complete'] == (len(done) == 8)
            for item in done:
                cell = body['cells'][item['case']]
                assert cell['native_status'] == 'succeeded' and cell['run_id'] == item['run_id']
                assert len(cell['rows']) == 480
                totals = cell['totals']
                assert totals['argmax_checked_tokens'] == totals['new_tokens']
                state['audited_cells'].append(dict(item, totals=totals, files_sha256=cell['files_sha256']))
            state.update(complete=body['complete'], phase='accepted_completed_cells', signature=signature,
                         readout_dir=str(target), readout_sha256=sha(target / ('norm_evidence_v1.json' if body['complete'] else 'norm_pending.json')))
    if not done or failed:
        state.update(phase='native_failure' if failed else 'waiting_for_first_complete_cell', signature=signature)
    latest.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--continuation-study', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        state = audit_once(args.data_root, args.output_dir, args.continuation_study)
        print(json.dumps({k: state[k] for k in ('complete', 'native_completed', 'total', 'phase')}), flush=True)
        if not args.watch or state['complete'] or state['failed']:
            break
        time.sleep(30)

    if state['failed']:
        raise SystemExit(2)
