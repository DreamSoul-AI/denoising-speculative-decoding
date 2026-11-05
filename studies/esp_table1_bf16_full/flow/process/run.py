"""Read original full-AR saved Runs into a new external output; never decode."""
from studies._shared.flow.process import run
__all__ = ['run']

import argparse
import json
import os
from pathlib import Path

import yaml
from studies._shared.structure.evidence import digest
from studies._shared.structure.sources import files_digest
from studies.esp_table1_bf16_full_argmax.flow.process.run import SavedReadout
from studies.esp_table1_bf16_full_argmax.structure.readout import count_summary, ordered_cells
from studies.esp_table1_bf16_full.structure import readout as protocol


def _producer_sources(session, study, execution, assets):
    """Read declared original source bytes, independently of this readout's sources."""
    source = session.read_json(assets / 'spec_mtp_sources.json')
    files, study_files = source.get('files'), source.get('study_files')
    if (source.get('package') != 'spec_mtp' or not isinstance(files, dict) or not files
            or not isinstance(study_files, dict) or not study_files
            or source.get('digest') != files_digest(files)):
        raise ValueError('full readout requires a complete original producer source binding')
    for table, package in ((files, True), (study_files, False)):
        for member, expected in table.items():
            path = Path(member)
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(char not in '0123456789abcdef' for char in expected)
                    or path.is_absolute() or path.drive or '..' in path.parts):
                raise ValueError('invalid original producer source member or SHA256')
            if member.startswith(('spec_mtp/', 'rpipe/')):
                actual = execution / 'src' / path
                root = execution
            elif member.startswith('studies/_shared/'):
                actual = execution / path
                root = execution
            elif package:
                raise ValueError('unknown original producer package member')
            else:
                actual = study / path
                root = study
            if not actual.resolve(strict=True).is_relative_to(root):
                raise ValueError('original producer source member resolves outside its declared root')
            if session.sha(actual) != expected:
                raise ValueError('original producer source changed: ' + member)
    return source


def analyze(study_dir, output_root, *, execution_root):
    study, execution = Path(study_dir).resolve(strict=True), Path(execution_root).resolve(strict=True)
    session = SavedReadout(study_dir, output_root, execution_root)
    output = session.output
    manifest = session.read_json(study / 'docs/prompts_manifest.json')
    prompts = manifest['rendered_prompts']
    if not prompts:
        raise ValueError('full readout requires its nonempty fixed prompt manifest')
    observed = {}
    reused = study / 'docs/reused_preflight.json'
    if session.exists(reused):
        reuse = session.read_json(reused)
        source = execution / reuse['source_evidence']
        if session.sha(source) != reuse['evidence_sha256']:
            raise ValueError('reused preflight evidence identity differs')
        original = session.read_json(source)
        for cell in reuse['cells']:
            matches = [row for row in original['cells'] if row['BC'] == cell['BC']
                       and row['actual_spec']['num_masks'] == cell['actual_spec']['num_masks']]
            if (len(matches) != 1 or any(matches[0].get(key) != cell.get(key)
                    for key in ('rows', 'raw_tokens', 'actual_spec', 'model_digest', 'n'))
                    or cell['n'] != len(prompts) or len(cell['rows']) != len(prompts)
                    or len(cell['raw_tokens']) != len(prompts) or cell['dtype'] != 'torch.bfloat16'
                    or cell['actual_spec']['mask'] != {'init': 'mean_prompt', 'lam': .1, 'update': True}
                    or any(prompt['input_ids'] != evidence['input_ids']
                           for prompt, evidence in zip(prompts, cell['raw_tokens']))):
                raise ValueError('reused preflight cells differ from their bound source or fixed prompts')
            copied = dict(cell, provenance='reused frozen preflight; not rerun or re-signed')
            copied.update(count_summary(cell['rows']))
            observed[cell['BC'], cell['actual_spec']['num_masks']] = copied
    index = session.read_json(study / 'index.json')
    snapshot = session.read_json(study / 'docs/modelscope_snapshot.json')
    declared, run_ids = set(), set()
    for entry in index['experiments']:
        spec = entry['factors']['algorithm.config']['spec']
        key = spec['block_complexity'], spec['num_masks']
        if key not in {item[:2] for item in protocol.CELLS} or key in declared or len(entry['runs']) != 1:
            raise ValueError('full readout index must keep its four unique predeclared cells')
        declared.add(key)
        run_id = entry['runs'][0]['id']
        if not isinstance(run_id, str) or Path(run_id).name != run_id or run_id in ('', '.', '..') or run_id in run_ids:
            raise ValueError('full readout index contains an invalid or duplicated Run identity')
        run_ids.add(run_id)
        folder = study / 'runs' / run_id
        assets = folder / 'assets'
        raw_file = assets / 'decode_observations.json'
        if not session.exists(raw_file):
            continue
        sources = _producer_sources(session, study, execution, assets)
        config = yaml.safe_load(session.read_text(folder / 'config.yaml'))
        raw, inputs = session.read_json(raw_file), session.read_json(assets / 'decode_inputs.json')
        if sources.get('config_digest') != digest(config):
            raise ValueError('original producer configuration digest differs from the saved Run configuration')
        if (raw['inputs_digest'] != digest(inputs) or inputs['dtype'] != 'torch.bfloat16'
                or inputs['backend'] != 'eager' or inputs['settings']['spec'] != spec
                or config['seed'] != 123 or config['algorithm']['config']['spec'] != spec
                or spec['tree'] != {'mode': 'dynamic', 'pruning': True}
                or spec['mask'] != {'init': 'mean_prompt', 'lam': .1, 'update': True}):
            raise ValueError('full native input, configuration or scientific protocol differs')
        settings = inputs['settings']['eval']
        if inputs['settings'].get('run', {}).get('seed') != config['seed']:
            raise ValueError('captured native run seed differs from the saved Run configuration')
        if (settings['temperature'] != 0 or settings['max_new_tokens'] != 256 or settings['warmup'] is not False
                or settings['run_ar_baseline'] is not True or settings['stop_on_eos'] is not True):
            raise ValueError('full native readout requires its original independent-AR evaluation settings')
        if (len(inputs['prompts']) != len(prompts) or any(a['input_ids'] != b['input_ids']
                or a['category'] != b['category'] for a, b in zip(inputs['prompts'], prompts))):
            raise ValueError('full native input prompt list differs from the fixed manifest')
        landed = {name.replace('\\', '/').rsplit('/', 1)[-1]: sha for name, sha in inputs['files'].items()}
        if not snapshot.get('files') or any(landed[item['path']] != item['sha256'] for item in snapshot['files']):
            raise ValueError('full native model source identity differs from its original snapshot')
        policy = session.read_json(assets / 'numeric_policy.json')
        if policy != {'dtype': 'torch.bfloat16', 'bf16_reduced_precision_reduction': True, 'deterministic': False}:
            raise ValueError('full native numeric policy differs')
        rows, audits = protocol.native_rows(raw['out'], prompts, spec)
        if key in observed:
            raise ValueError('full readout cannot replace a reused cell with a different native Run')
        cell = {'run_id': run_id, 'actual_spec': spec, 'dtype': inputs['dtype'], 'model_digest': inputs['model_digest'],
            'runtime': inputs['runtime'], 'numeric_policy': policy, 'rows': rows, 'token_audits': audits,
            'raw_sha256': session.sha(raw_file), 'inputs_sha256': session.sha(assets / 'decode_inputs.json'),
            'original_source_digest': sources['digest'], 'producer_sources_sha256': session.sha(assets / 'spec_mtp_sources.json'),
            'provenance': 'original saved native Run; not executed or resigned by this readout'}
        cell.update(count_summary(rows))
        observed[key] = cell
    if declared != {item[:2] for item in protocol.CELLS}:
        raise ValueError('full readout index is missing a predeclared cell')
    cells = ordered_cells(observed, declarations=protocol.CELLS)
    if len({cell['model_digest'] for cell in cells if not cell['pending']}) > 1:
        raise ValueError('full native cells use different model identities')
    body = {'scope': f'{len(prompts)} fixed SpecBench prompts; BC60/1 primary and BC60/2 control predeclared',
        'fixed_prompts': len(prompts), 'all_cells_observed': all(not cell['pending'] for cell in cells),
        'quality_contract': 'all saved speculative output tokens equal independent AR; argmax is not a substitute',
        'quality_passed': all(not cell['pending'] and cell['exact_count'] == len(prompts) for cell in cells),
        'no_best_mask_selection': True, 'readout_only': True, 'readout_model_forwards': 0, 'cells': cells}
    filename = 'protocol_evidence_v1.json' if body['all_cells_observed'] else 'protocol_pending.json'
    session.write_text(output / filename, json.dumps(body, indent=2, allow_nan=False) + '\n')
    links = {}
    for label, path in (('固定 prompt 清单', study / 'docs/prompts_manifest.json'), ('原 index', study / 'index.json')):
        try:
            links[label] = Path(os.path.relpath(path, output)).as_posix()
        except ValueError:
            links[label] = path.as_posix()
    session.write_text(output / 'FULL_REPORT.md', protocol.render_report(body, filename, links))
    session.finish()
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study-dir', 'execution-root', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.study_dir, args.output_root, execution_root=args.execution_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
