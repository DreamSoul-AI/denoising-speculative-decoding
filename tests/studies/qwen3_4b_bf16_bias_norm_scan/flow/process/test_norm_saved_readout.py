"""cpu pure IO/stats of the actual archived reused endpoint report."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

reader = importlib.import_module('studies.qwen3_4b_bf16_bias_norm_scan.flow.process.run')
ROOT = Path(__file__).resolve().parents[5]
COMMIT = 'b8b20b5ada7fd860894114cbf224ce749af430cf'


def test_actual_reused_report_reaggregates_without_native_acceptance(historical_file, tmp_path):
    proof = historical_file('studies/qwen3_4b_bf16_bias_norm_scan/docs/reused_readiness_v1.json')
    direction = historical_file('studies/qwen3_4b_bf16_directions/docs/direction_evidence_v1.json')
    output = tmp_path/'readout'
    body = reader.published(proof, direction, ROOT, COMMIT, hashlib.sha256(proof.read_bytes()).hexdigest(),
        hashlib.sha256(direction.read_bytes()).hexdigest(), output)
    assert body['raw_argmax_checked_tokens'] == 405083
    scope = json.loads((output/'counter_scope.json').read_bytes())
    assert scope['cells_checked'] == 5 and scope['native_run_revalidated'] is False
    assert scope['raw_files_revalidated'] is False
    with pytest.raises(FileExistsError):
        reader.published(proof, direction, ROOT, COMMIT, hashlib.sha256(proof.read_bytes()).hexdigest(),
            hashlib.sha256(direction.read_bytes()).hexdigest(), output)


def test_native_output_cannot_enter_the_separate_continuation_input_root(tmp_path):
    data, study, continuation = (tmp_path/name for name in ('data', 'study', 'continuation'))
    for folder in (data, study, continuation):
        folder.mkdir()
    output = continuation/'new-output'
    with pytest.raises(ValueError, match='separate'):
        reader.native_readout(data, study, output, continuation)
    assert not output.exists()


@pytest.mark.parametrize('fault', ['plot_failure', 'input_drift'])
def test_native_transaction_does_not_publish_complete_after_plot_failure_or_input_drift(tmp_path, fault):
    from studies.qwen3_4b_bf16_paired_revalidation.flow.process.run import PublishedReadout
    inputs = tmp_path/'inputs'; inputs.mkdir()
    source = inputs/'source.json'; source.write_text('{}', encoding='utf-8')
    output = tmp_path/'output'
    session = PublishedReadout(inputs, output, ROOT)
    session.read_json(source)
    target = output/'norm_evidence_v1.json'
    session.write_text(target, '{"complete": true}\n')
    def render(folder):
        raise RuntimeError('injected plot failure')
    if fault == 'input_drift':
        source.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises((RuntimeError, ValueError)):
        session.finish(render=render)
    assert not target.exists() and not (output/'readout_inputs.json').exists()
