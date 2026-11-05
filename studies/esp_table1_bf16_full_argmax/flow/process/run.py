"""Read saved evidence into a fresh external report; never execute a model."""
from studies._shared.flow.process import run as run

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from studies._shared.structure.evidence import digest
from studies.esp_table1_bf16_full_argmax.structure.readout import summarize, ordered_cells
from studies._shared.structure.esp_controls import token_digest


class SavedReadout:
    """Bind actual saved bytes and publish one new readout without rewriting inputs."""
    def __init__(self, study, output, *roots):
        from spec_mtp.structure.artifact.layout import validate_study_root
        from spec_mtp.structure.artifact.paths import validate_output_root
        from studies._shared.structure.sources import package_files

        if not __debug__:
            raise RuntimeError('saved evidence readout requires Python assertions enabled')
        self.roots = tuple(Path(root).resolve(strict=True) for root in (study, *roots))
        self.output = validate_study_root(validate_output_root(output, *self.roots))
        if self.output.exists() or self.output.is_symlink():
            raise FileExistsError('readout requires a new external output root')
        self.inputs = {}
        self.resolved = {}
        self.historical = []
        self.missing = set()
        self.outputs = {}
        self.package = package_files()
        self.source_files = self._source_files()

    def _source_files(self):
        from spec_mtp.structure.artifact.provenance import file_sha256

        checkout = Path(__file__).resolve().parents[4]
        names = ('esp_table1_bf16_full_argmax', 'esp_table1_bf16_qwen_protocol', 'esp_table1_nonofficial_revalidation', 'esp_table1_bf16_full')
        paths = [path for name in names for pillar in ('structure', 'flow')
                 for path in (checkout / 'studies' / name / pillar).rglob('*.py')]
        from studies._shared.structure import esp_controls
        assert esp_controls.__file__ is not None
        paths.append(Path(esp_controls.__file__))
        return {str(path.resolve()): file_sha256(path) for path in paths}

    def _input(self, path, *, strict=True):
        alias = Path(path).absolute()
        path = alias.resolve(strict=strict)
        if alias in self.resolved and self.resolved[alias] != path:
            raise ValueError('saved input alias changed its resolved target')
        if not any(path.is_relative_to(root) for root in self.roots) and str(path) not in self.source_files:
            raise ValueError('saved readout input escapes the explicitly supplied roots')
        self.resolved[alias] = path
        return path

    def read_bytes(self, path):
        path = self._input(path)
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if path in self.inputs and self.inputs[path] != sha:
            raise ValueError('saved input changed between reads')
        self.inputs[path] = sha
        return data

    def read_text(self, path, encoding='utf-8'):
        return self.read_bytes(path).decode(encoding)

    def read_json(self, path):
        return json.loads(self.read_bytes(path))

    def sha(self, path):
        from spec_mtp.structure.artifact.provenance import file_sha256

        path = self._input(path)
        sha = file_sha256(path)
        if path in self.inputs and self.inputs[path] != sha:
            raise ValueError('saved input changed between reads')
        self.inputs[path] = sha
        return sha

    def exists(self, path):
        path = Path(path).absolute()
        if path.resolve().is_relative_to(self.output):
            return False
        path = self._input(path, strict=False)
        present = path.exists()
        if not present:
            self.missing.add(path)
        return present

    def write_text(self, path, text, encoding='utf-8'):
        import os
        import re

        path = Path(path).resolve()
        path.relative_to(self.output)
        if path.name == 'STUDY_REPORT.md':
            text = re.sub(r'\[PLAN\]\(<\.\./\.\./\.\./docs/[^>]+>\)',
                          '原冻结 PLAN（本次保存输入未提供固定 Git 身份，来源未知）', text)
            for name in ('prompts_manifest.json', 'native-process-revalidation-v1.json'):
                source = self.roots[0] / 'docs' / name
                if f']({name})' not in text:
                    continue
                if self.exists(source):
                    self.sha(source)
                    try:
                        target = Path(os.path.relpath(source, self.output)).as_posix()
                    except ValueError:
                        target = source.as_posix()
                    text = text.replace(f']({name})', f'](<{target}>)')
                else:
                    text = re.sub(r'\[([^]]+)\]\(' + re.escape(name) + r'\)',
                                  r'\1（本次输入缺失，来源未知）', text)
        if path in self.outputs:
            raise ValueError('readout attempted to publish the same file twice')
        self.outputs[path] = text.encode(encoding)
        return len(text)

    def verify(self):
        from spec_mtp.structure.artifact.provenance import file_sha256
        from studies._shared.structure.sources import package_files

        if (self.package != package_files() or self.source_files != self._source_files()
                or any(file_sha256(path) != sha for path, sha in self.inputs.items())
                or any(path.exists() for path in self.missing)
                or any(alias.resolve() != path for alias, path in self.resolved.items())):
            raise ValueError('saved inputs, pending artifacts or current readout sources changed')

    def finish(self, figure=None, *, render=None):
        from spec_mtp.structure.artifact.provenance import file_sha256

        self.verify()
        self.output.mkdir(parents=True, exist_ok=False)
        staged = {}
        for path, data in self.outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.partial')
            with partial.open('xb') as handle:
                handle.write(data)
            staged[path] = partial
        figure_sources = {}
        if figure is not None:
            from studies.esp_table1_bf16_full_argmax.flow.write.run import write_plot
            figure_path = self.output / 'figures/tau_acceptance.png'
            write_plot(figure, figure_path)
            figure_sources[figure_path.relative_to(self.output).as_posix()] = file_sha256(figure_path)
        if render is not None:
            for figure_path in render(self.output):
                figure_path = Path(figure_path).resolve(strict=True)
                figure_sources[figure_path.relative_to(self.output).as_posix()] = file_sha256(figure_path)
        self.verify()
        if any(file_sha256(self.output / name) != sha for name, sha in figure_sources.items()):
            raise ValueError('generated readout figure changed before publication')
        binding = {'schema': 1, 'scope': 'saved-count readout; no model forward or historical source re-signing',
                   'input_sources': {str(path): sha for path, sha in self.inputs.items()},
                   'missing_inputs': sorted(str(path) for path in self.missing),
                   'resolved_inputs': {str(alias): str(path) for alias, path in self.resolved.items()},
                   'historical_sources': self.historical,
                   'output_sources': {**{path.relative_to(self.output).as_posix(): hashlib.sha256(data).hexdigest()
                                         for path, data in self.outputs.items()}, **figure_sources},
                   'readout_sources': self.source_files, 'package_sources': self.package}
        marker = self.output / 'readout_inputs.json'
        marker_partial = marker.with_name(marker.name + '.partial')
        with marker_partial.open('x', encoding='utf-8') as handle:
            json.dump(binding, handle, indent=2)
        # Publish the ready evidence last, after every fallible rendering and source check.
        ready = [path for path in staged if path.name.endswith('_evidence_v1.json')]
        ordered = [path for path in staged if path not in ready]
        for path in ordered:
            if path.exists():
                raise FileExistsError(path)
            staged[path].rename(path)
        marker_partial.rename(marker)
        for path in ready:
            if path.exists():
                raise FileExistsError(path)
            staged[path].rename(path)

def analyze(study_dir, output_root, *, execution_root):
    study = Path(study_dir).resolve()
    output = Path(output_root).resolve()
    session = SavedReadout(study, output, Path(execution_root).resolve())
    figure = None
    manifest = json.loads(session.read_text(study / 'docs/prompts_manifest.json'))
    prompts = manifest['rendered_prompts']
    observed = {}
    reused = study / 'docs/reused_preflight.json'
    if session.exists(reused):
        reuse = json.loads(session.read_text(reused))
        source = Path(execution_root).resolve() / reuse['source_evidence']
        assert hashlib.sha256(session.read_bytes(source)).hexdigest() == reuse['evidence_sha256']
        for c in reuse['cells']:
            assert c['n'] == len(prompts) and len(c['rows']) == len(prompts)
            assert c['dtype'] == 'torch.bfloat16' and c['actual_spec']['mask'] == {'init': 'mean_prompt', 'lam': 0.1, 'update': True}
            assert all((p['input_ids'] == e['input_ids'] for p, e in zip(prompts, c['raw_tokens'])))
            cell = dict(c, provenance='reused frozen preflight; not rerun or re-signed')
            cell.update(summarize(c['rows']))
            observed[c['BC'], c['actual_spec']['num_masks']] = cell
    index = json.loads(session.read_text(study / 'index.json'))
    snapshot = json.loads(session.read_text(study / 'docs/modelscope_snapshot.json'))
    for entry in index['experiments']:
        spec = entry['factors']['algorithm.config']['spec']
        run_id = entry['runs'][0]['id']
        assets = study / 'runs' / run_id / 'assets'
        raw_file = assets / 'decode_observations.json'
        if not session.exists(raw_file):
            continue
        raw = json.loads(session.read_text(raw_file))
        input_file = assets / 'decode_inputs.json'
        inputs = json.loads(session.read_text(input_file))
        assert raw['inputs_digest'] == digest(inputs) and inputs['dtype'] == 'torch.bfloat16'
        assert inputs['backend'] == 'eager' and inputs['settings']['eval']['temperature'] == 0
        assert inputs['settings']['eval']['max_new_tokens'] == 256 and inputs['settings']['eval']['warmup'] is False
        assert inputs['settings']['spec'] == spec and spec['tree'] == {'mode': 'dynamic', 'pruning': True}
        assert spec['mask'] == {'init': 'mean_prompt', 'lam': 0.1, 'update': True}
        landed = {name.replace('\\', '/').rsplit('/', 1)[-1]: sha for name, sha in inputs['files'].items()}
        assert all((landed[f['path']] == f['sha256'] for f in snapshot['files']))
        policy = json.loads(session.read_text(assets / 'numeric_policy.json'))
        assert policy == {'dtype': 'torch.bfloat16', 'bf16_reduced_precision_reduction': True, 'deterministic': False}
        out = raw['out']
        assert len(out['rows']) == len(out['evidence']) == len(prompts)
        audits = [json.loads(line) for line in session.read_text(assets / 'argmax_audit.jsonl').splitlines()]
        assert len(audits) == len(prompts)
        rows: list[dict[str, Any]] = []
        for p, r, e in zip(prompts, out['rows'], out['evidence']):
            assert p['input_ids'] == e['input_ids'] and r['spec_time'] == r['ar_time'] == 0
            assert e['stats']['block_complexity'] == spec['block_complexity']
            assert bool(r['exact_match']) == (e['output_ids'] == e['ar_ids'])
            audit = e['stats']['verification_argmax_audit']
            assert audit == audits[len(rows)] and audit['passed'] is True
            assert audit['input_sha256'] == token_digest(e['input_ids']) and audit['output_sha256'] == token_digest(e['output_ids'])
            continuation = e['output_ids'][len(e['input_ids']):]
            assert audit['checked_committed_tokens'] == len(continuation) == e['stats']['new_tokens']
            assert len(audit['committed']) == len(continuation)
            assert all((a['chosen'] == a['argmax'] == token for a, token in zip(audit['committed'], continuation)))
            first = next((i - len(e['input_ids']) for i, (a, b) in enumerate(zip(e['output_ids'], e['ar_ids'])) if a != b), min(len(e['output_ids']), len(e['ar_ids'])) - len(e['input_ids'])) if e['output_ids'] != e['ar_ids'] else None
            rows.append(dict(r, question_id=p['question_id'], exact=bool(r['exact_match']), first_new_difference_index=first, argmax_passed=True, argmax_checked_tokens=len(continuation)))
        result_file = assets.parent / 'result.json'
        native = json.loads(session.read_text(result_file)) if session.exists(result_file) else {}
        native_status = native.get('status', 'pending finalisation')
        gate = native.get('spec_mtp') or {}
        cell = {'run_id': run_id, 'native_status': native_status, 'native_validation_policy': gate.get('validation_policy'), 'validator_sha256': gate.get('validator_sha256'), 'gate_revalidation': {k: v for k, v in (gate.get('revalidation') or {}).items() if k in ('previous_result_sha256', 'previous_status', 'evidence_sha256', 'execute_rerun')}, 'argmax_audit_sha256': hashlib.sha256(session.read_bytes(assets / 'argmax_audit.jsonl')).hexdigest(), 'actual_spec': spec, 'dtype': inputs['dtype'], 'model_digest': inputs['model_digest'], 'runtime': inputs['runtime'], 'numeric_policy': policy, 'rows': rows, 'token_audits': [{'question_id': p['question_id'], 'output_sha256': hashlib.sha256(json.dumps(e['output_ids']).encode()).hexdigest(), 'AR_sha256': hashlib.sha256(json.dumps(e['ar_ids']).encode()).hexdigest(), 'first_new_difference_index': next((i - len(e['input_ids']) for i, (a, b) in enumerate(zip(e['output_ids'], e['ar_ids'])) if a != b), min(len(e['output_ids']), len(e['ar_ids'])) - len(e['input_ids'])) if e['output_ids'] != e['ar_ids'] else None} for p, e in zip(prompts, out['evidence'])], 'raw_sha256': hashlib.sha256(session.read_bytes(raw_file)).hexdigest(), 'inputs_sha256': hashlib.sha256(session.read_bytes(input_file)).hexdigest(), 'provenance': 'new native Run'}
        cell.update(summarize(rows))
        key = (spec['block_complexity'], spec['num_masks'])
        assert key not in observed
        observed[key] = cell
    cells = ordered_cells(observed)
    assert len({c['model_digest'] for c in cells if not c['pending']}) <= 1
    body = {'scope': f'{len(prompts)} fixed SpecBench prompts; BC60/1 primary and BC60/2 control predeclared', 'all_cells_observed': all((not c['pending'] for c in cells)), 'quality_contract': 'tau with full committed-token verification-forward argmax audit; AR diagnostic separately', 'quality_passed': all((not c['pending'] and c['argmax_all_passed'] for c in cells)), 'AR_all_exact': all((not c['pending'] and c['exact_count'] == len(prompts) for c in cells)), 'no_best_mask_selection': True, 'cells': cells}
    formal_ready = body['all_cells_observed'] and all((c.get('native_status') == 'succeeded' and c.get('native_validation_policy') == 'verifier-argmax-v1' for c in cells))
    body['native_finalisation_complete'] = formal_ready
    filename = 'protocol_evidence_v1.json' if formal_ready else 'protocol_pending.json'
    path = output / filename
    text = json.dumps(body, indent=2) + '\n'
    if session.exists(path) and formal_ready:
        assert session.read_text(path) == text
    else:
        session.write_text(path, text, encoding='utf-8')
    reference = f'[{filename}]({filename})' if formal_ready else f'本机临时文件 `{filename}`（未入Git）'
    lines = ['# ESP Table 1 Qwen3 BF16 protocol report', '', f"{len(prompts)} prompts; completed={sum((not c['pending'] for c in cells))}/4; 全四格验收={('passed' if body['quality_passed'] else 'pending')}. 已完成格的τ/argmax有效性单列，AR失配不否定该格。BC60/1 is primary and BC60/2 is control, fixed before results.", '', '| BC/masks | role | τ/argmax结果 | native状态 | AR exact/n (rate) | argmax checked/pass | pooled τ | τ − target | prompt mean τ | mean − target | category mean τ | category − target |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in cells:
        values = ['n/a' if key.endswith('_target_delta') and c['target'] is None else 'pending' if c.get(key) is None else f'{c[key]:.6f}' for key in ('pooled_tau', 'pooled_tau_target_delta', 'prompt_mean_tau', 'prompt_mean_tau_target_delta', 'category_mean_pooled_tau', 'category_mean_pooled_tau_target_delta')]
        exact = 'pending' if c['pending'] else f"{c['exact_count']}/{c['n']} ({c['AR_exact_rate']:.2%})"
        audit = 'pending' if c['pending'] else f"{c['argmax_checked_tokens']}/{c['argmax_all_passed']}"
        valid = 'pending' if c['pending'] else 'valid' if c['argmax_all_passed'] else 'invalid'
        lines.append(f"| {c['BC']}/{c['num_masks']} | {c['role']} | {valid} | {c.get('native_status', 'pending')} | {exact} | {audit} | " + ' | '.join(values) + ' |')
    lines += ['', '目标BC30=1.63，BC60双方均列相对1.74差值，不按τ选主配置。AR独立审计失配单列，不能宣称与独立AR一致；不阻止τ计算。τ验收要求全480、全部提交token的验证前向argmax自检通过，不以预检替代结果。', '', '聚合定义：pooled τ=Σcommitted/Σdecode_calls；prompt mean τ为逐条committed/max(decode_calls,1)的等权均值；category mean τ先按原始category各算pooled τ，再对类别等权。类别样本数不等，不能把该权重称为论文已确认的任务权重；分组见[prompt清单](prompts_manifest.json)。', '', '计数定义：committed为解码阶段实际提交数，包含bonus，经过EOS/预算裁剪，不含prefill首token。argmax审计覆盖完整新输出（含prefill首token），因此全480条的argmax checked比Σcommitted多480；实际model calls=Σdecode_calls+480。accepted为候选接受数，不能替代committed或τ。', '', '[PLAN](<../../../docs/研究计划 Denoising Speculative Decoding.md#legacy-studies-esp-table1-bf16-full-argmax-docs-plan>)引用G.2具体说明：Qwen3 BC60单mask/29节点，双mask/19节点为对照；保持动态树剪枝，BF16/T0/256/seed123，无测速/预热。', '', 'Mean mask按Eq5 λ=.1更新，与后续4B用户指定的ESP frozen mean不同。四格全部新Run，旧冻结PLAN不回写；旧AR门失败result归档；仅重新验收后记录真实native状态及新validator SHA，不重跑execute。新增argmax必须全过；独立AR差异不阻塞native summarize/process。', '', f'逐prompt计数、AR首分叉位置和自检覆盖及哈希见{reference}。完整原始tokens/argmax选择记录/Run仅在本机保留，未运行格不能用预检外推。']
    if formal_ready and body['quality_passed']:
        figure = body
        matched = all((round(c['pooled_tau'], 2) == c['target'] for c in cells if c['role'] == 'primary' and c['target'] is not None))
        lines += ['', f"结论：主配置pooled τ按两位小数与论文目标{('一致' if matched else '不一致')}。数学自检与原生生命周期通过不等于论文τ数值复现成功；三种聚合和双mask对照全部保留，不事后择优。", '', '[原生process验收证据](native-process-revalidation-v1.json)记录四格成功状态及旧process归档哈希。', '', '![Full480 tau and accepted draft counts](figures/tau_acceptance.png)', '', f'图：同一四格480条，三种τ聚合及真实accepted draft总数（EOS/预算裁剪前）；黑菱形为论文目标。来源[{filename}]({filename})，AR一致率另列，不含测速。']
    session.write_text(output / 'STUDY_REPORT.md', '\n'.join(lines) + '\n', encoding='utf-8')
    session.finish(figure=figure)
    return body

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study-dir','output-root','execution-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    analyze(args.study_dir,args.output_root,execution_root=args.execution_root)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
