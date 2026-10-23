"""Versioned scientific observations and their read-only audit."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from spec_mtp.contracts import AuditResult


class JsonlSink:
    """Exclusive output identity; each acknowledged record is flushed."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = self.path.open('x', encoding='utf-8', newline='\n')

    def write(self, observation):
        self.handle.write(json.dumps(dict(observation), ensure_ascii=False, allow_nan=False) + '\n')
        self.handle.flush()

    def close(self):
        self.handle.close()


def _invalid_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def verify_files(root, declared):
    """Recheck frozen members without modifying or recreating missing evidence."""
    root = Path(root).resolve()
    for relative, expected in declared.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('evidence member escapes the explicit root')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'evidence member changed: {relative}')


def audit_run(raw, expected_sources):
    """Check existing observations only; never regenerate missing logits."""
    reasons = []
    try:
        if isinstance(raw, (str, Path)):
            with Path(raw).open(encoding='utf-8') as handle:
                records = []
                for line in handle:
                    if not line.endswith('\n') or len(line) > 8 * 1024 * 1024:
                        raise ValueError('truncated or oversized observation line')
                    records.append(json.loads(line, parse_constant=_invalid_constant))
        else:
            records = list(raw)
        if any(not isinstance(row, dict) for row in records):
            raise ValueError('each observation must be a JSON object')
        if (len(records) < 2 or records[0].get('event') != 'header'
                or records[-1].get('event') != 'end'
                or any(row.get('event') != 'commit' for row in records[1:-1])):
            raise ValueError('missing, duplicate or out-of-order observation envelope')
        header, end = records[0], records[-1]
        if header.get('schema') != 1 or type(header.get('schema')) is not int:
            raise ValueError('unknown observation schema')
        if not header.get('run_id') or not header.get('qid'):
            raise ValueError('missing run/prompt identity')
        source_integrity = header.get('sources') == dict(expected_sources)
        if not source_integrity:
            reasons.append('source digest mismatch')
        tokens, commits = end['committed'], records[1:-1]
        if (not isinstance(tokens, list) or len(tokens) != len(commits)
                or any(type(t) is not int or t < 0 for t in tokens)
                or any(type(row.get('index')) is not int or row['index'] != index
                       or type(row.get('token')) is not int or row['token'] != token
                       for index, (row, token) in enumerate(zip(commits, tokens)))):
            raise ValueError('committed token coverage/order mismatch')
        if end.get('execution') not in ('completed', 'cancelled'):
            raise ValueError('incomplete execution')
        temperature = header['temperature']
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if any(type(token) is not int or token < 0 for token in header['input_ids']) or not header['input_ids']:
            raise ValueError('invalid prompt token evidence')
        if temperature != 0:
            return AuditResult(source_integrity, False, False, tuple(reasons + ['T0 audit is inapplicable to sampling']))
        stats = end['stats']
        if not isinstance(stats, dict):
            raise ValueError('stats must be a JSON object')
        if tokens:
            for row in commits:
                if any(type(row.get(key)) is not int or row[key] < 0 for key in ('forward', 'row', 'argmax')):
                    raise ValueError('forward/row/argmax must be nonnegative integers')
            verification = stats.get('verification_argmax_audit')
            if not isinstance(verification, dict) or not isinstance(verification.get('committed'), list):
                raise ValueError('invalid verifier audit object')
            if any(not isinstance(row, dict) or any(type(row.get(key)) is not int or row[key] < 0
                   for key in ('forward', 'row', 'chosen', 'argmax')) for row in verification['committed']):
                raise ValueError('invalid verifier audit record')
            item = {'input_ids': header['input_ids'], 'output_ids': header['input_ids'] + tokens, 'stats': stats}
            _check_argmax(item)
            expected = stats['verification_argmax_audit']['committed']
            if any((row['forward'], row['row'], row['argmax'], row['token']) !=
                   (ref['forward'], ref['row'], ref['argmax'], ref['chosen'])
                   for row, ref in zip(commits, expected)):
                raise ValueError('streamed commit differs from forward/row audit')
        elif stats.get('new_tokens') != 0 or stats.get('prefill_calls') != 0 or stats.get('decode_calls') != 0:
            raise ValueError('empty result has inconsistent forward counts')
        return AuditResult(source_integrity, True, True, tuple(reasons))
    except (ValueError, TypeError, KeyError, OSError, SpecMtpGateError) as error:
        return AuditResult(False, False, False, tuple(reasons + [str(error)]))

class SpecMtpGateError(RuntimeError):
    pass

def _check_argmax(item: dict[str, Any]) -> int:
    stats = item['stats']
    audit = stats.get('verification_argmax_audit') or {}
    prompt, output = item['input_ids'], item['output_ids']
    tokens = output[len(prompt):]
    records = audit.get('committed') or []
    token_hash = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if (audit.get('passed') is not True or not tokens
            or audit.get('checked_committed_tokens') != len(tokens)
            or len(records) != len(tokens) or stats['new_tokens'] != len(tokens)
            or audit.get('input_sha256') != token_hash(prompt)
            or audit.get('output_sha256') != token_hash(output)):
        raise SpecMtpGateError('missing or invalid verifier argmax coverage/hash')
    owners = [(0, len(prompt) - 1)]
    for step, trace in enumerate(stats['trace'], 1):
        owners.extend((step, row) for row in ([0] + trace['accepted_nodes'])[:len(trace['committed_tokens'])])
    if len(owners) != len(tokens):
        raise SpecMtpGateError('verifier argmax owner coverage mismatch')
    for record, token, owner in zip(records, tokens, owners):
        if (record.get('chosen') != token or record.get('argmax') != token
                or (record.get('forward'), record.get('row')) != owner):
            raise SpecMtpGateError('committed token differs from verifier argmax/owner')
    return len(tokens)
