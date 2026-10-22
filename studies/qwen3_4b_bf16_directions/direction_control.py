"""Study-scoped requested direction controls; native decoder and gates stay intact."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
BASE = ROOT / 'studies/qwen3_4b_bf16_five_methods'
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'tools'))
from untimed_pairing import PairedUntimedEval, build_cuda  # noqa: E402
from diagnose_4b_mask_windows import admission, content_start_of  # noqa: E402
import spec_mtp.decode as decoding  # noqa: E402
from rpipe.structure.algorithm.factory import AlgorithmRegistry  # noqa: E402
from rpipe.structure.model.factory import ModelRegistry  # noqa: E402


def check_refs():
    refs = json.loads((STUDY / 'docs/source_refs.json').read_text())
    for name, digest in refs.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def control(provider, mode, start, info):
    """Replace only velocity construction, retaining native mean/anchor/EMA code."""
    assert mode in ('template', 'random0', 'random1', 'random2', 'content')
    assert not provider.normalize and not provider.extrapolate_ema and provider.level_mode == 'order0'
    original_init, original_commit = provider.init_from_prompt, provider.on_token_committed

    def initialize(embeds):
        original_init(embeds)
        full_mean, template = provider._level.clone(), (embeds[1] - embeds[0]).clone()
        if mode == 'content':
            assert 0 <= start < len(embeds) - 1
            original_init(embeds[start:])
            provider._level = full_mean
        elif mode == 'template':
            provider._velocity = template
        else:
            seed = int(mode[-1])
            vector = torch.randn(embeds.shape[-1], generator=torch.Generator().manual_seed(seed))
            vector *= template.float().norm().cpu() / vector.norm()
            provider._velocity = vector.to(device=embeds.device, dtype=embeds.dtype)
            assert abs(float(provider._velocity.float().norm() / template.float().norm()) - 1) < .01
        v = provider._velocity.float()
        info.update(mode=mode, content_start=start if mode == 'content' else None,
            velocity_norm=float(v.norm()), template_norm=float(template.float().norm()),
            norm_ratio=float(v.norm() / template.float().norm()),
            cosine_to_template=float(torch.nn.functional.cosine_similarity(v, template.float(), dim=0)),
            full_prompt_mean_preserved=bool(torch.equal(provider._level, full_mean)),
            velocity_sha256=hashlib.sha256(provider._velocity.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
        assert info['full_prompt_mean_preserved']

    def commit(new, previous):
        velocity = provider._velocity
        original_commit(new, previous)
        if mode != 'content':
            provider._velocity = velocity

    provider.init_from_prompt, provider.on_token_committed = initialize, commit
    return provider


def admitted_cuda(*args, **kwargs):
    check_refs()
    check = admission()
    assert check['passed'], f'Nonpreemptive GPU admission deferred: {check}'
    model = build_cuda(*args, **kwargs)
    model.meta['direction_admission'] = check
    return model


class DirectionEval(PairedUntimedEval):
    def run(self, data, model, system, tracker):
        check_refs()
        from transformers import AutoTokenizer
        manifest = json.loads((BASE / 'docs/prompts_manifest.json').read_text())
        records = {p['question_id']: p for p in map(json.loads,
            (ROOT / 'datasets/spec_bench/question.jsonl').read_text(encoding='utf-8').splitlines())}
        assert hashlib.sha256((ROOT / 'datasets/spec_bench/question.jsonl').read_bytes()).hexdigest() == manifest['data_sha256']
        expected = manifest['rendered_prompts']
        assert len(expected) == 480
        tokenizer = AutoTokenizer.from_pretrained(BASE / 'shared/model/modelscope/Qwen/Qwen3-4B', local_files_only=True)
        starts = [content_start_of(tokenizer, records[p['question_id']]['turns'][0], p['input_ids']) for p in expected]
        mode = self.config.setting('direction')
        builder, generate = decoding.build_mask_provider, decoding.SpecDecoder.generate
        current, progress = {'index': 0, 'direction': {}}, []
        assets = Path(system.assets_dir)
        (assets / 'direction_admission.json').write_text(json.dumps(model.meta['direction_admission'], indent=2) + '\n')

        def controlled_builder(*args, **kwargs):
            return control(builder(*args, **kwargs), mode, starts[current['index']], current['direction'])

        def observed_generate(decoder, prompt):
            index = current['index']
            assert prompt[0].tolist() == expected[index]['input_ids']
            check = admission(initial=False)
            assert check['passed'], f'Foreign GPU work appeared: {check}'
            current['direction'] = {}
            output, stats = generate(decoder, prompt)
            assert stats['verification_argmax_audit']['passed']
            assert stats['prefill_time'] == stats['decode_time'] == 0
            row = {'index': index, 'question_id': expected[index]['question_id'], 'direction': current['direction'],
                'gpu_admission': check,
                'accepted': stats['accepted'], 'committed': stats['committed'], 'decode_calls': stats['decode_calls'],
                'new_tokens': stats['new_tokens'], 'argmax_checked_tokens': stats['verification_argmax_audit']['checked_committed_tokens']}
            progress.append(row)
            with (assets / 'direction_partial_private.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'output_ids': output[0].tolist(), 'stats': stats, **row}) + '\n')
            (assets / 'direction_progress.json').write_text(json.dumps({'direction': mode, 'completed': len(progress),
                'total': 480, 'rows': progress}, indent=2) + '\n', encoding='utf-8')
            if (index + 1) % 10 == 0:
                print('Direction', mode, 'completed', index + 1, '/ 480', flush=True)
            current['index'] += 1
            return output, stats

        with patch.object(decoding, 'build_mask_provider', side_effect=controlled_builder), \
                patch.object(decoding.SpecDecoder, 'generate', observed_generate):
            result = super().run(data, model, system, tracker)
        assert current['index'] == 480
        check_refs()
        return result


def register_overrides():
    ModelRegistry.register('hf_causal', 'spec_mtp', admitted_cuda)
    AlgorithmRegistry.register('eval', 'spec_mtp', DirectionEval)
