"""Revalidate only: python tools/revalidate_run.py studies/<name> <run> --validator /path/validation.py."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import patch


def load_validator(path):
    spec = importlib.util.spec_from_file_location('revalidation_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def revalidate_run(study, run_id, validator_path):
    from rpipe.flow.context import FlowContext
    from rpipe.structure.artifact.config import load_config
    from rpipe.structure.artifact.layout import artifact_layout
    from rpipe.structure.artifact.provenance import check_frozen
    from rpipe.structure.make.expand import load_study_yaml
    from rpipe.flow.write.run import run as write
    summarize = import_module('studies._shared.flow.summarize.run')

    study = Path(study).resolve()
    validator_path = Path(validator_path).resolve()
    layout = artifact_layout(study, run_id)
    config = load_config(layout.config_path)
    validator = load_validator(validator_path)
    check_frozen(study, load_study_yaml(study))
    # Uses the installed execution package's unchanged source manifest. The new
    # validator is loaded separately, never substituted into that source lock.
    report = validator.evaluate_run(config, layout.assets_dir)
    validator.verify_evidence(config, layout.assets_dir)
    if not report['passed']:
        raise RuntimeError(report['reasons'])
    names = ('decode_observations.json','decode_inputs.json','argmax_audit.jsonl','spec_mtp_sources.json')
    hashes = {name:hashlib.sha256((layout.assets_dir/name).read_bytes()).hexdigest()
              for name in names if (layout.assets_dir/name).is_file()}
    previous_bytes = layout.result_path.read_bytes()
    previous = json.loads(previous_bytes)
    if previous.get('status') == 'succeeded':
        migration = previous.get('spec_mtp',{}).get('revalidation',{})
        if migration.get('evidence_sha256') != hashes or previous['spec_mtp'].get('validator_sha256') != report['validator_sha256']:
            raise RuntimeError('existing success has different validation/evidence; do not re-sign')
        return previous
    if previous.get('status') != 'failed':
        raise RuntimeError('only a completed failed Run can be revalidated')
    previous_sha = hashlib.sha256(previous_bytes).hexdigest()
    history = layout.assets_dir/'gate_revalidation'/f'{previous_sha}.json'
    history.parent.mkdir(exist_ok=True)
    if history.exists():
        assert history.read_bytes() == previous_bytes
    else:
        with history.open('xb') as handle:
            handle.write(previous_bytes)
    draft = deepcopy(previous)
    draft['status'] = 'succeeded'
    draft.pop('error',None)
    ctx = FlowContext(study_dir=study,layout=layout,config=config,state={'result_draft':draft})
    with ExitStack() as stack:
        stack.enter_context(patch.object(summarize,'evaluate_run',validator.evaluate_run))
        stack.enter_context(patch.object(summarize,'verify_evidence',validator.verify_evidence))
        summarize.run_gate(ctx)
    draft['spec_mtp']['revalidation'] = {'previous_result_sha256':previous_sha,
        'previous_status':previous['status'],'previous_error':previous.get('error'),
        'previous_result_asset':str(history.relative_to(layout.assets_dir)),
        'evidence_sha256':hashes,'validator_path':str(validator_path),'execute_rerun':False}
    assert all(hashlib.sha256((layout.assets_dir/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    write(ctx)
    return ctx.state['result']


def process_study(study, validator_path):
    from rpipe.flow.process import run_study
    process = import_module('studies._shared.flow.process.run')
    validator = load_validator(Path(validator_path).resolve())
    with patch.object(process,'verify_evidence',validator.verify_evidence):
        return run_study(Path(study).resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('run_id',nargs='?')
    parser.add_argument('--validator',required=True,type=Path)
    parser.add_argument('--process',action='store_true')
    args=parser.parse_args()
    if args.run_id:
        result=revalidate_run(args.study,args.run_id,args.validator)
        print(json.dumps({'run':args.run_id,'status':result['status'],'gate':result['spec_mtp']}))
    if args.process:
        print(process_study(args.study,args.validator))
    if not args.run_id and not args.process:
        parser.error('provide a completed Run id or --process')


if __name__=='__main__':
    main()
