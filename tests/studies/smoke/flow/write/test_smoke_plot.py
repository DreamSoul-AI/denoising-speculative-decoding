"""CPU saved-scalar Agg rendering, input binding and completion failure boundaries."""
import json
import pytest
from spec_mtp.structure.artifact.provenance import file_sha256
from studies.smoke.flow.write.run import write_plot
from studies.smoke.structure import validation as protocol


def test_historical_smoke_plot_uses_explicit_input_and_new_root_only(tmp_path,historical_file):
    source=historical_file('studies/smoke/docs/delivery_validation.json');before=source.read_bytes()
    output=tmp_path/'render';image=write_plot(source,file_sha256(source),output)
    assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    receipt=json.loads((output/'plot_complete.json').read_bytes())
    assert receipt['runtime_input']['sha256']==file_sha256(source)
    assert receipt['implementation_commit']==protocol.IMPLEMENTATION_COMMIT and source.read_bytes()==before
    with pytest.raises(FileExistsError):write_plot(source,file_sha256(source),output)


@pytest.mark.parametrize('fault',['hash','input_drift','figure_failure'])
def test_historical_smoke_plot_refuses_drift_and_never_publishes_partial_completion(tmp_path,historical_file,monkeypatch,fault):
    source=tmp_path/'inputs/report.json';source.parent.mkdir()
    source.write_bytes(historical_file('studies/smoke/docs/delivery_validation.json').read_bytes())
    expected=file_sha256(source);output=tmp_path/'render'
    original=protocol.historical_cpu_figure
    def figure(data):
        value=original(data)
        if fault=='input_drift':source.write_bytes(source.read_bytes()+b' ')
        if fault=='figure_failure':
            def broken(*a,**kw):raise OSError('injected save failure')
            monkeypatch.setattr(value,'savefig',broken)
        return value
    monkeypatch.setattr(protocol,'historical_cpu_figure',figure)
    with pytest.raises((ValueError,OSError)):
        write_plot(source,'0'*64 if fault=='hash' else expected,output)
    assert not (output/'plot_complete.json').exists()
    if fault!='figure_failure':assert not output.exists()


@pytest.mark.parametrize('fault', ['cleanup_only', 'body_plus_cleanup', 'cleanup_mutates_input'])
def test_historical_smoke_figure_cleanup_is_part_of_completion(tmp_path, historical_file, monkeypatch, fault):
    # Copy the already-bound fixture into this case's input root, never mutate its original Git material.
    source = tmp_path / 'inputs/report.json'
    source.parent.mkdir()
    source.write_bytes(historical_file('studies/smoke/docs/delivery_validation.json').read_bytes())
    before = source.read_bytes()
    expected_sha = file_sha256(source)
    output = tmp_path / 'render'
    primary = OSError('injected Figure body failure')
    cleanup = RuntimeError('injected Figure cleanup failure')
    calls = []

    class Figure:
        def savefig(self, stream, **kwargs):
            calls.append('save')
            stream.write(b'partial PNG retained')
            if fault == 'body_plus_cleanup':
                raise primary
        def clear(self):
            calls.append('clear')
            if fault == 'cleanup_mutates_input':
                source.write_bytes(b'changed during Figure cleanup')
            else:
                raise cleanup

    monkeypatch.setattr(protocol, 'historical_cpu_figure', lambda data: Figure())
    expected = ValueError if fault == 'cleanup_mutates_input' else OSError if fault == 'body_plus_cleanup' else RuntimeError
    with pytest.raises(expected) as caught:
        write_plot(source, expected_sha, output)
    if fault == 'body_plus_cleanup':
        assert caught.value is primary and caught.value.figure_cleanup_error is cleanup
    elif fault == 'cleanup_only':
        assert caught.value is cleanup
    else:
        assert 'changed' in str(caught.value)
    assert calls == ['save', 'clear']
    assert (output / 'tiny_acceptance.png').read_bytes() == b'partial PNG retained'
    assert not (output / 'plot_complete.json').exists()
    if fault != 'cleanup_mutates_input':
        assert source.read_bytes() == before
