"""Validate only the current Study index after RPipe aggregation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from rpipe.flow.context import FlowContext, StudyProcessContext

from studies._shared.structure.validation import (
    GATE_KEY, STUDY_CHECK_NAME, SpecMtpGateError, declared_factors, verify_evidence,
)
def _normalised(items: list[dict[str, Any]]) -> list[str]:
    return sorted(json.dumps(item, sort_keys=True) for item in items)

def study_gate(ctx: StudyProcessContext | FlowContext, expected_factors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Study process hook (``ctx.scope == 'study'``); writes ``spec_mtp_study_check.json``."""
    from rpipe.structure.artifact._atomic import atomic_write_text
    from rpipe.structure.artifact.index import load_index

    if ctx.scope != 'study':
        return {}
    study_dir = Path(ctx.study_dir)
    index = load_index(study_dir)
    rows, missing, failed, digests, seen = [], [], [], set(), []
    for experiment in index.get('experiments') or []:
        seen.append(experiment.get('factors') or {})
        for run in experiment.get('runs') or []:
            root = study_dir / 'runs' / str(run.get('run_dir') or run['id'])
            result_path = root / 'result.json'
            if not result_path.is_file():
                missing.append(run['id'])
                continue
            result = json.loads(result_path.read_text(encoding='utf-8'))
            gate = dict(result.get(GATE_KEY) or {})
            try:
                from rpipe.structure.artifact.config import load_config
                verify_evidence(load_config(root / 'config.yaml'), root / 'assets')
                saved_sources = json.loads((root / 'assets' / 'spec_mtp_sources.json').read_text(encoding='utf-8'))
                if gate.get('sources_digest') != saved_sources.get('digest'):
                    raise SpecMtpGateError('result source digest mismatch')
            except (RuntimeError, ValueError, KeyError, OSError) as error:
                gate['passed'] = False
                gate['evidence_error'] = str(error)
            row = {'run': run['id'], 'seed': run.get('seed'), 'factors': experiment.get('factors') or {},
                   'status': result.get('status'), 'gate_passed': gate.get('passed') is True,
                   'exact_match_rate': gate.get('exact_match_rate'), 'accept_d1': gate.get('accept_d1'),
                   'argmax_all_passed': gate.get('argmax_all_passed'),
                   'argmax_checked_tokens': gate.get('argmax_checked_tokens'),
                   'validation_policy': gate.get('validation_policy'),
                   'sources_digest': gate.get('sources_digest'), 'evidence_error': gate.get('evidence_error')}
            rows.append(row)
            if row['status'] != 'succeeded' or not row['gate_passed'] or not row['sources_digest']:
                failed.append(run['id'])
            if row['sources_digest']:
                digests.add(row['sources_digest'])
    expected_ok = True
    if expected_factors is None and (study_dir / 'study.yaml').is_file():
        from rpipe.structure.make.expand import load_study_yaml

        from rpipe.structure.artifact.config import load_config

        expected_factors = declared_factors(load_study_yaml(study_dir),
                                            load_config(study_dir / 'experiment_config.yaml'))
    seeds_ok = True
    if (study_dir / 'study.yaml').is_file():
        from rpipe.structure.make.expand import load_study_yaml

        from rpipe.structure.make.expand import expand_patches

        declared_seeds = sorted({patch['seed'] for patch in expand_patches(load_study_yaml(study_dir))})
        for experiment in index.get('experiments') or []:
            if declared_seeds and sorted(r.get('seed') for r in experiment.get('runs') or []) != declared_seeds:
                seeds_ok = False
    if expected_factors is not None:
        expected_ok = _normalised(seen) == _normalised(expected_factors)
    complete = not missing and bool(rows) and expected_ok and seeds_ok
    body = {
        'scope': 'study',
        'complete': complete,
        'passed': complete and not failed and len(digests) == 1,
        'runs': rows,
        'missing_runs': missing,
        'failed_runs': failed,
        'expected_factors_match': expected_ok,
        'declared_seeds_match': seeds_ok,
        'sources_digests': sorted(digests),
    }
    atomic_write_text(study_dir / STUDY_CHECK_NAME, json.dumps(body, indent=2, ensure_ascii=False) + '\n')
    ctx.state[GATE_KEY] = body
    process = ctx.state.get('process')
    if isinstance(process, dict):
        from rpipe.structure.artifact.result.format import encode_result

        process[GATE_KEY] = body
        process['complete'] = bool(process.get('complete')) and body['passed']
        atomic_write_text(study_dir / 'process.json', encode_result(process))
    if not body['passed']:
        state = 'incomplete' if not complete else 'failed'
        raise SpecMtpGateError(f'SpecMTP Study gate {state}; see {study_dir / STUDY_CHECK_NAME}')
    return body

run = study_gate
