"""Explicit saved-counter rendering preserves identities and refuses source drift."""
import json
from pathlib import Path

import pytest

from studies.qwen3_4b_bf16_five_methods.flow.write.run import write_published

ROOT = Path(__file__).resolve().parents[5]
COMMIT = 'b8b20b5ada7fd860894114cbf224ce749af430cf'
MEMBER = 'studies/qwen3_4b_bf16_five_methods/docs/paired_evidence_v1.json'
SHA = '8e85ac6ea9984047abf0ffd5d6653b28b6aadd74aa876139fa10465dc5bbf274'


def test_bound_saved_counter_plot_records_identity_without_overwrite(historical_file, tmp_path):
    from PIL import Image
    from spec_mtp.structure.artifact.provenance import file_sha256
    source = historical_file(MEMBER)
    before = source.read_bytes()
    output = tmp_path / 'figure'
    target = write_published(source, ROOT, COMMIT, SHA, output)
    receipt = json.loads((output / 'plot_complete.json').read_bytes())
    assert receipt['native_run_revalidated'] is False
    assert receipt['historical_input']['sha256'] == SHA
    assert receipt['figure']['sha256'] == file_sha256(target)
    with Image.open(target) as image:
        assert image.format == 'PNG' and image.width > 1500 and image.height > 1200
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        write_published(source, ROOT, COMMIT, SHA, output)


def test_plot_source_changed_during_readout_is_rejected_before_figure(historical_file, tmp_path, monkeypatch):
    import importlib
    processor = importlib.import_module('studies.qwen3_4b_bf16_five_methods.flow.process.run')
    from studies.qwen3_4b_bf16_five_methods.structure import plots
    fake = tmp_path / 'fake_plotter.py'
    fake.write_text('before', encoding='utf-8')
    monkeypatch.setattr(plots, '__file__', str(fake))
    original = processor.read_published

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        fake.write_text('after', encoding='utf-8')
        return result

    monkeypatch.setattr(processor, 'read_published', change)
    monkeypatch.setattr(plots, 'paired_figure', lambda body: pytest.fail('source drift must precede figure construction'))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='source changed'):
        write_published(historical_file(MEMBER), ROOT, COMMIT, SHA, output)
    assert not output.exists()
