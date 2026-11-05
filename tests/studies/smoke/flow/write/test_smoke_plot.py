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
