"""Pure SpecMTP result and evidence validation; no Flow orchestration."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from spec_mtp.audit import SpecMtpGateError, _check_argmax
GATE_KEY = 'spec_mtp'
STUDY_CHECK_NAME = 'spec_mtp_study_check.json'


def _overall(assets: Path) -> dict[str, str] | None:
    path = assets / 'summary.csv'
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('category') == 'overall']
    return rows[0] if len(rows) == 1 else None



def _argmax_evidence(assets: Path) -> dict[str, Any]:
    raw = json.loads((assets / 'decode_observations.json').read_text(encoding='utf-8'))
    evidence, rows = raw['out']['evidence'], raw['out']['rows']
    inputs = json.loads((assets / 'decode_inputs.json').read_text(encoding='utf-8'))
    if not evidence or len(evidence) != len(rows) or len(evidence) != len(inputs['prompts']):
        raise SpecMtpGateError('missing prompt decode evidence')
    checked = sum(_check_argmax(item) for item in evidence)
    audit_path = assets / 'argmax_audit.jsonl'
    if audit_path.is_file():
        recorded = [json.loads(line) for line in audit_path.read_text(encoding='utf-8').splitlines()]
        if recorded != [item['stats']['verification_argmax_audit'] for item in evidence]:
            raise SpecMtpGateError('verifier argmax JSONL does not match raw evidence')
    diagnostics, mismatches = [], []
    for index, (item, row) in enumerate(zip(evidence, rows)):
        ar, output = item.get('ar_ids'), item['output_ids']
        exact = ar == output if ar is not None else None
        if ar is not None and bool(row.get('exact_match')) != exact:
            mismatches.append(index)
        first = None if ar is None or exact else next(
            (i for i, (a, b) in enumerate(zip(output, ar)) if a != b), min(len(output), len(ar))) - len(item['input_ids'])
        diagnostics.append({'prompt_index': index, 'exact': exact, 'first_new_difference_index': first})
    return {'argmax_all_passed': True, 'argmax_checked_prompts': len(evidence),
            'argmax_checked_tokens': checked, 'AR_diagnostics': diagnostics,
            'AR_reported_exact_mismatches': mismatches,
            'exact_match_rate': sum(d['exact'] is True for d in diagnostics)/len(diagnostics)
                if all(d['exact'] is not None for d in diagnostics) else None}


def evaluate_run(config: dict[str, Any], assets: Path) -> dict[str, Any]:
    """Pure check of one Run's assets; no decoding."""
    algorithm = config.get('algorithm') or {}
    settings = algorithm.get('config') or {}
    task = str(settings.get('task', 'esp_mtp')).lower()
    out: dict[str, Any] = {'task': task, 'passed': False, 'reasons': [],
                           'validation_policy': 'verifier-argmax-v1',
                           'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    sources = assets / 'spec_mtp_sources.json'
    if sources.is_file():
        out['sources_digest'] = json.loads(sources.read_text(encoding='utf-8')).get('digest')
    else:
        out['reasons'].append('missing assets/spec_mtp_sources.json')
    if task == 'esp_mtp':
        overall = _overall(assets)
        if overall is None:
            out['reasons'].append('summary.csv has no single overall row')
        else:
            raw = overall.get('exact_match_rate')
            rate = float(raw) if raw is not None and raw != '' else None
            out['reported_exact_match_rate'] = rate
            for key in ('prompts', 'accept_d1', 'accept_d2', 'accept_d3', 'tau', 'call_reduction'):
                if overall.get(key) not in (None, ''):
                    out[key] = float(overall[key])
        try:
            out.update(_argmax_evidence(assets))
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            out['reasons'].append(str(error))
    elif task == 'ar_baseline':
        if _overall_any(assets) is None:
            out['reasons'].append('missing summary.csv')
    elif task == 'alignment_probe':
        if not (assets / 'alignment_metrics.csv').is_file():
            out['reasons'].append('missing alignment_metrics.csv')
    if task not in ('esp_mtp', 'ar_baseline', 'alignment_probe'):
        out['reasons'].append(f'unsupported task: {task}')
    out['passed'] = not out['reasons']
    return out

def _overall_any(assets: Path) -> Path | None:
    path = assets / 'summary.csv'
    return path if path.is_file() else None

def declared_factors(study: dict[str, Any], base_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Expected Experiment factors from declarations, never from the current index.

    With a base, reproduce RPipe's effective merge/group contract. The one-arg
    compatibility form exposes raw axes only; Study verification always supplies
    its base so inherited fields inside mapping-valued axes are checked too.
    """
    from rpipe.structure.make.expand import cartesian

    axes = dict(study.get('axes') or {})
    axes.pop('seed', None)
    if base_config is None:
        return cartesian(axes)
    from rpipe.structure.artifact.index import experiment_entries
    from rpipe.structure.control import experiment_config_from_mapping, run_config_from_merge
    from rpipe.structure.make.expand import axis_keys_from_study, expand_patches

    base = experiment_config_from_mapping(base_config)
    runs = [run_config_from_merge(base, patch) for patch in expand_patches(study)]
    entries = experiment_entries(
        configs=[(Path('runs') / str(run.id) / 'config.yaml', run.to_mapping()) for run in runs],
        axis_keys=axis_keys_from_study(study),
    )
    return [entry['factors'] for entry in entries]

def verify_evidence(config: dict[str, Any], assets: Path) -> None:
    """Recompute correctness from raw tokens, and reject execution/finalisation drift."""
    from studies._shared.structure.evidence import check_inputs, digest
    from studies._shared.structure.sources import changed_files, manifest

    saved = json.loads((assets / 'spec_mtp_sources.json').read_text(encoding='utf-8'))
    current = manifest()
    if saved.get('digest') != current['digest'] or changed_files(saved, current) or saved.get('rpipe') != current['rpipe']:
        raise SpecMtpGateError('SpecMTP sources changed since prepare')
    if saved.get('config_digest') != digest(config):
        raise SpecMtpGateError('Run configuration changed since prepare')
    inputs = json.loads((assets / 'decode_inputs.json').read_text(encoding='utf-8'))
    raw = json.loads((assets / 'decode_observations.json').read_text(encoding='utf-8'))
    if raw['inputs_digest'] != digest(inputs):
        raise SpecMtpGateError('decode input evidence digest mismatch')
    check_inputs(inputs)
    study_dir = assets.parents[2]
    if (study_dir / 'study.yaml').is_file():
        from rpipe.structure.artifact.provenance import source_files
        from rpipe.structure.make.expand import load_study_yaml
        if saved.get('study_files') != source_files(study_dir, load_study_yaml(study_dir)):
            raise SpecMtpGateError('Study sources changed since prepare')
    settings = inputs['settings']
    if str(settings['run']['seed']) != str(config.get('seed', settings['run']['seed'])):
        raise SpecMtpGateError('decode seed changed')
    algorithm = config.get('algorithm') or {}
    from rpipe.structure.control.layers import AlgorithmConfig
    from studies._shared.structure.algorithm import effective_spec, task_name
    from studies._shared.structure.algorithm import legacy_cfg
    from types import SimpleNamespace

    algo = AlgorithmConfig.from_mapping(algorithm)
    expected = legacy_cfg(SimpleNamespace(meta={}), SimpleNamespace(meta={}),
                          SimpleNamespace(device=settings['eval']['device']), algo, settings['run']['seed'])
    expected.spec = effective_spec(expected.spec, algo)
    if digest(expected.spec) != digest(settings['spec']) or digest(expected.eval) != digest(settings['eval']):
        raise SpecMtpGateError('decode configuration changed')
    task = task_name(algo)
    if raw['task'] != task:
        raise SpecMtpGateError('decode task mismatch')
    # Runtime gates apply inside a Run; later readout compares the saved inputs and raw evidence.
    if task == 'esp_mtp':
        _argmax_evidence(assets)
        evidence, rows = raw['out']['evidence'], raw['out']['rows']
        if not evidence or len(evidence) != len(inputs['prompts']) or len(rows) != len(evidence):
            raise SpecMtpGateError('missing prompt decode evidence')
        for item, row, prompt in zip(evidence, rows, inputs['prompts']):
            if item['input_ids'] != prompt['input_ids'] or item['output_ids'][:len(item['input_ids'])] != item['input_ids']:
                raise SpecMtpGateError('prompt token evidence mismatch')
            stats = item['stats']
            if stats['new_tokens'] != len(item['output_ids']) - len(item['input_ids']):
                raise SpecMtpGateError('generated token count mismatch')
            if len(stats['trace']) != stats['decode_calls'] or sum(len(t['accepted_nodes']) for t in stats['trace']) != stats['accepted']:
                raise SpecMtpGateError('acceptance trace mismatch')
            continuation = item['output_ids'][len(item['input_ids']):]
            if sum(len(t['committed_tokens']) for t in stats['trace']) != stats['committed']:
                raise SpecMtpGateError('committed trace count mismatch')
            if continuation[1:] != [token for t in stats['trace'] for token in t['committed_tokens']]:
                raise SpecMtpGateError('committed token trace mismatch')
            for trace in stats['trace']:
                if trace['cache_before'] != trace['root_position']:
                    raise SpecMtpGateError('cache prefix mismatch')
                if not trace['stopped'] and trace['cache_after'] != trace['root_position'] + 1 + len(trace['accepted_nodes']):
                    raise SpecMtpGateError('cache eviction mismatch')
            if settings['spec'].get('method') == 'output_hidden':
                import math
                import torch

                prefill = stats.get('output_hidden_prefill') or {}
                scale = prefill.get('scale', 0)
                k = int(settings['spec']['num_masks'])
                tree = settings['spec'].get('tree', {})
                expected_owners = (1 + sum(tree['branches']) if tree.get('mode', 'dynamic') == 'static'
                                   else int(settings['spec']['block_complexity']) // (k + 1))
                # Actual masks are cast to model dtype after FP32 normalization.
                mapping_tol = max(1e-5, torch.finfo(getattr(torch, inputs['dtype'].split('.')[-1])).eps)
                if (prefill.get('observed_positions') != len(item['input_ids'])
                        or prefill.get('state_dtype') != 'float32' or prefill.get('final_normalized') is not True
                        or not math.isfinite(scale) or scale <= 0):
                    raise SpecMtpGateError('output hidden prefill evidence invalid')
                for trace in stats['trace']:
                    observation = trace.get('output_hidden') or {}
                    expected_rows = [] if trace['stopped'] else [0] + trace['accepted_nodes']
                    if (observation.get('observed_block_rows') != expected_rows
                            or observation.get('owner_count') != expected_owners or observation.get('slot_count') != k):
                        raise SpecMtpGateError('output hidden owner/observation evidence invalid')
                    for key in ('mask_norm_min', 'mask_norm_max'):
                        value = observation.get(key, float('nan'))
                        if not math.isfinite(value) or not math.isclose(value, scale, rel_tol=mapping_tol):
                            raise SpecMtpGateError('output hidden mapping norm evidence invalid')
            for key in ('accepted', 'committed', 'decode_calls', 'new_tokens'):
                if row[key] != stats[key]:
                    raise SpecMtpGateError(f'{key} evidence mismatch')
        from studies._shared.structure.evaluate import _aggregate
        expected_summary = _aggregate(rows)['overall']
        actual = _overall(assets)
        def same_value(key, value):
            raw_value = actual.get(key) if actual is not None else None
            if value is None:
                return raw_value in ('', None)
            if type(value) is bool:
                return raw_value == str(value)
            return raw_value not in ('', None) and float(raw_value) == float(value)
        if actual is None or any(not same_value(k, v) for k, v in expected_summary.items() if k != 'exact_match_rate'):
            raise SpecMtpGateError('summary does not match raw decode evidence')
    elif task == 'ar_baseline':
        evidence, rows = raw['out']['evidence'], raw['out']['rows']
        if not evidence or len(evidence) != len(inputs['prompts']) or len(rows) != len(evidence):
            raise SpecMtpGateError('missing AR token evidence')
        for item, row, prompt in zip(evidence, rows, inputs['prompts']):
            if item['input_ids'] != prompt['input_ids'] or item['output_ids'][:len(item['input_ids'])] != item['input_ids']:
                raise SpecMtpGateError('AR prompt evidence mismatch')
            if row['ar_new_tokens'] != len(item['output_ids']) - len(item['input_ids']):
                raise SpecMtpGateError('AR generated token count mismatch')
        with (assets / 'summary.csv').open(newline='', encoding='utf-8') as handle:
            summary_rows = list(csv.DictReader(handle))
        total_time = sum(r['ar_time'] for r in rows)
        expected_speed = sum(r['ar_new_tokens'] for r in rows) / total_time if total_time > 0 else 0
        if len(summary_rows) != 1 or float(summary_rows[0]['prompts']) != len(evidence) or float(summary_rows[0]['tokens_per_s']) != expected_speed:
            raise SpecMtpGateError('AR summary does not match raw evidence')
    elif task == 'alignment_probe':
        out = raw['out']
        for key, field in (('accept', 'n_accepted'), ('reject', 'n_rejected')):
            count = out['counts'].get(key)
            if (type(count) is not int or count < 0 or type(out.get(field)) is not int
                    or out[field] != count):
                raise SpecMtpGateError('alignment observation count mismatch')
        if out['n_accepted'] + out['n_rejected'] <= 0:
            raise SpecMtpGateError('alignment probe has no observations')
        with (assets / 'alignment_metrics.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        import torch

        if len(rows) != len(out['sums']['accept']):
            raise SpecMtpGateError('alignment layer evidence missing')
        for i, row in enumerate(rows):
            if any(row.get(field) != str(out[field]) for field in ('n_accepted', 'n_rejected')):
                raise SpecMtpGateError('alignment CSV count mismatch')
            for key, column in (('accept', 'cos_accepted'), ('reject', 'cos_rejected')):
                expected_value = (torch.tensor(out['sums'][key][i]) / max(out['counts'][key], 1)).item()
                if float(row[column]) != expected_value:
                    raise SpecMtpGateError('alignment CSV does not match raw evidence')
