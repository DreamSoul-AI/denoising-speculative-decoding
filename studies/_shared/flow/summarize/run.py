"""Reject invalid evidence before success is finalised."""
from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from rpipe.flow.context import FlowContext

from studies._shared.structure.validation import GATE_KEY, SpecMtpGateError, evaluate_run, verify_evidence
def run_gate(ctx: FlowContext) -> dict[str, Any]:
    """Study write hook: gate this Run before RPipe writes result.json."""
    report = evaluate_run(ctx.config, Path(ctx.layout.assets_dir))
    try:
        verify_evidence(ctx.config, Path(ctx.layout.assets_dir))
        if ctx.state.get('model') is not None:
            from studies._shared.structure.evidence import model_digest, runtime
            inputs = json.loads((Path(ctx.layout.assets_dir) / 'decode_inputs.json').read_text(encoding='utf-8'))
            if inputs['runtime'] != runtime() or inputs['model_digest'] != model_digest(ctx.state['model'].module):
                raise SpecMtpGateError('model or runtime changed before finalisation')
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        report['reasons'].append(str(error))
        report['passed'] = False
    draft = ctx.state.get('result_draft')
    if draft is not None:
        draft[GATE_KEY] = report
        if report['task'] == 'esp_mtp' and report.get('exact_match_rate') is not None:
            draft.setdefault('metrics', {})['test_exactmatchrate'] = report['exact_match_rate']
        sources = Path(ctx.layout.assets_dir) / 'spec_mtp_sources.json'
        if sources.is_file():
            body = json.loads(sources.read_text(encoding='utf-8'))
            draft[GATE_KEY]['sources'] = {k: body.get(k) for k in ('digest', 'git', 'rpipe')}
    if not report['passed']:
        raise SpecMtpGateError(f"SpecMTP gate failed: {'; '.join(report['reasons'])} "
                               f'(assets: {ctx.layout.assets_dir}; see docs/architecture.md "Run path and gates")')
    return report

run = run_gate
