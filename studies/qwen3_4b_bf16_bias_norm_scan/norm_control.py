"""Observe the existing fixed direction at native EMA step_scale; no decoder changes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BASE = ROOT / 'studies/qwen3_4b_bf16_five_methods'
sys.path.insert(0, str(STUDY))
for folder in ('qwen3_4b_bf16_five_methods', 'qwen3_4b_bf16_directions', 'esp_table1_nonofficial_revalidation'):
    sys.path.insert(0, str(ROOT / 'studies' / folder))
from untimed_pairing import PairedUntimedEval, build_cuda  # noqa: E402
from direction_control import control  # noqa: E402
from norm_admission import wait_for_admission  # noqa: E402
import spec_mtp.decode as decoding  # noqa: E402
from rpipe.structure.algorithm.factory import AlgorithmRegistry  # noqa: E402
from rpipe.structure.model.factory import ModelRegistry  # noqa: E402


def check_refs():
    refs = json.loads((STUDY / 'docs/source_refs.json').read_text(encoding='utf-8'))
    for name, expected in refs['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    plan = refs['project_plan']
    blob = subprocess.check_output(['git', 'show', plan['commit'] + ':' + plan['path']], cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest() == plan['sha256']


def observe_initial_bias(provider, mode, info):
    control(provider, mode, None, info)
    initialize = provider.init_from_prompt

    def init(embeds):
        initialize(embeds)
        step = provider._step_vector(provider._velocity, provider._ref_norm)
        masks = provider.masks()
        info.update(step_scale=provider.step_scale, initial_mean_norm=float(provider._level.float().norm()),
            effective_step_norm=float(step.float().norm()),
            rounded_slot_displacement_norms=(masks.float() - provider._level.float()).norm(dim=-1).tolist())
        assert all(torch.isfinite(x).all() for x in (step, masks))

    provider.init_from_prompt = init
    return provider


def admitted_build(*args, **kwargs):
    check_refs()
    assets = kwargs.get('assets_dir', args[1] if len(args) > 1 else None)
    admission = wait_for_admission(assets, initial=True)
    model = build_cuda(*args, **kwargs)
    model.meta['norm_scan_admission'] = admission
    return model


class NormEval(PairedUntimedEval):
    def run(self, data, model, system, tracker):
        check_refs()
        expected = json.loads((BASE / 'docs/prompts_manifest.json').read_text(encoding='utf-8'))['rendered_prompts']
        mode = self.config.setting('direction')
        scale = self.config.setting('spec')['ema']['step_scale']
        assert mode in ('template', 'random0', 'random1', 'random2') and scale in (.5, 2.)
        assert len(expected) == 480
        builder, generate = decoding.build_mask_provider, decoding.SpecDecoder.generate
        current, progress = {'index': 0, 'bias': {}}, []
        assets = Path(system.assets_dir)
        (assets / 'norm_admission_initial.json').write_text(json.dumps(model.meta['norm_scan_admission']) + '\n')

        def controlled_builder(*args, **kwargs):
            provider = builder(*args, **kwargs)
            assert provider.step_scale == scale
            return observe_initial_bias(provider, mode, current['bias'])

        def observed_generate(decoder, prompt):
            index = current['index']
            assert prompt[0].tolist() == expected[index]['input_ids']
            admission = wait_for_admission(assets, initial=False)
            current['bias'] = {}
            output, stats = generate(decoder, prompt)
            audit = stats['verification_argmax_audit']
            assert audit['passed'] and audit['checked_committed_tokens'] == stats['new_tokens']
            assert stats['prefill_time'] == stats['decode_time'] == 0
            row = {'index': index, 'question_id': expected[index]['question_id'], 'bias': current['bias'],
                'accepted': stats['accepted'], 'committed': stats['committed'], 'decode_calls': stats['decode_calls'],
                'new_tokens': stats['new_tokens'], 'argmax_checked_tokens': audit['checked_committed_tokens'],
                'attempt_by_depth': stats['attempt_by_depth'], 'accept_by_depth': stats['accept_by_depth'],
                'first_block_accepted': len(stats['trace'][0]['accepted_nodes']) if stats['trace'] else None,
                'gpu_admission': admission}
            progress.append(row)
            with (assets / 'norm_partial_private.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'output_ids': output[0].tolist(), 'stats': stats, **row}) + '\n')
            (assets / 'norm_progress.json').write_text(json.dumps({'direction': mode, 'step_scale': scale,
                'completed': len(progress), 'total': 480, 'rows': progress}, indent=2) + '\n', encoding='utf-8')
            if (index + 1) % 10 == 0:
                print('Norm scan', mode, scale, index + 1, '/480', flush=True)
            current['index'] += 1
            return output, stats

        with patch.object(decoding, 'build_mask_provider', side_effect=controlled_builder), \
                patch.object(decoding.SpecDecoder, 'generate', observed_generate):
            result = super().run(data, model, system, tracker)
        assert current['index'] == 480
        check_refs()
        return result


def register_overrides():
    ModelRegistry.register('hf_causal', 'spec_mtp', admitted_build)
    AlgorithmRegistry.register('eval', 'spec_mtp', NormEval)
