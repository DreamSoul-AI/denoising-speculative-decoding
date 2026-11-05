"""CPU Study declaration/config/index/provenance publication through the actual make writer."""
import json
from pathlib import Path

import pytest
import yaml

from spec_mtp.structure.make.write import expand_study
from spec_mtp.structure.control.run_config import RunConfig
from spec_mtp.structure.control.hashing import compute_run_id
from spec_mtp.structure.artifact.provenance import file_sha256


def declaration(tmp_path):
    source=tmp_path/'source'
    source.mkdir()
    (source/'protocol.txt').write_text('fixed engineering-only configuration fixture\n')
    (source/'study.yaml').write_text(yaml.safe_dump({'study':'wiring-fixture','seeds':[7],
        'provenance':{'include':['protocol.txt']},'flow':{'study_phases':True},
        'axes':{'model.config.width':[4,8]}}))
    (source/'experiment_config.yaml').write_text(yaml.safe_dump({'experiment':'wiring-fixture','seed':7,
        'data':{'name':'fixture'},'model':{'name':'fixture'},'algorithm':{'mode':'eval',
        'config':{'version':'wiring-fixture-v1'}},'system':{'config':{'device':'cpu'}}}))
    return source


@pytest.mark.parametrize('external',[False,True])
def test_expand_writer_preserves_content_ids_factors_and_fixed_source_inputs(tmp_path,external):
    source=declaration(tmp_path)
    original={p.name:p.read_bytes() for p in source.iterdir()}
    result=expand_study(source,run_root=tmp_path/'unrelated-runtime' if external else None)
    target=result['study_dir']
    assert len(result['configs'])==2 and result['index'].is_file() and result['provenance'].is_file()
    widths=[]
    for filename in result['configs']:
        config=yaml.safe_load(filename.read_text())
        assert RunConfig.from_mapping(config).id==compute_run_id(config)==config['id']==filename.parent.name
        assert config['algorithm']['config']['version']=='wiring-fixture-v1' and config['seed']==7
        widths.append(config['model']['config']['width'])
    assert widths==[4,8]
    index=json.loads(result['index'].read_bytes())
    assert [row['factors']['model.config.width'] for row in index['experiments']]==[4,8]
    assert all((source/name).read_bytes()==raw for name,raw in original.items())
    if external:
        marker=json.loads((target/'study_source.json').read_bytes())
        assert Path(marker['source_dir'])==source


def test_expand_writer_refuses_changed_snapshot_before_rewriting_existing_artifacts(tmp_path):
    source=declaration(tmp_path);runtime=tmp_path/'external-runtime'
    expand_study(source,run_root=runtime)
    before={p.relative_to(runtime):file_sha256(p) for p in runtime.rglob('*') if p.is_file()}
    (source/'protocol.txt').write_text('changed frozen source\n')
    with pytest.raises(ValueError):expand_study(source,run_root=runtime)
    assert {p.relative_to(runtime):file_sha256(p) for p in runtime.rglob('*') if p.is_file()}==before
