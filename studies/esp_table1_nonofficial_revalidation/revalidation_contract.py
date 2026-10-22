"""Source/input guards around the unchanged untimed8B decoder."""
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import spec_mtp.decode as decoding
from spec_mtp.structure.artifact import evidence
from untimed_bf16 import UntimedEval, build_cuda
from resource_admission import wait_for_admission

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent


def check_refs():
    refs = json.loads((STUDY / 'docs/source_refs.json').read_text(encoding='utf-8'))
    for name, expected in refs['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    plan = refs['project_plan']
    blob = subprocess.check_output(['git', 'show', plan['commit'] + ':' + plan['path']], cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest() == plan['sha256']
    return refs


def admitted_build(*args, **kwargs):
    check_refs()
    assets = kwargs.get('assets_dir', args[1] if len(args) > 1 else None)
    admission = wait_for_admission(assets, initial=True)
    print('Initial host-RAM/GPU admission', json.dumps(admission), flush=True)
    model = build_cuda(*args, **kwargs)
    model.meta['initial_resource_admission'] = admission
    return model


class RevalidationEval(UntimedEval):
    def run(self, data, model, system, tracker):
        check_refs()
        (Path(system.assets_dir) / 'resource_admission_initial.json').write_text(
            json.dumps(model.meta['initial_resource_admission'], indent=2) + '\n', encoding='utf-8')
        original_capture, original_generate = evidence.capture, decoding.SpecDecoder.generate
        baseline = ROOT / 'studies/esp_table1_bf16_full_argmax/docs'
        expected = json.loads((baseline / 'prompts_manifest.json').read_text(encoding='utf-8'))['rendered_prompts']
        formal = json.loads((baseline / 'protocol_evidence_v1.json').read_text(encoding='utf-8'))
        count = [0]

        def capture(*args, **kwargs):
            saved = original_capture(*args, **kwargs)
            assert saved['dtype'] == 'torch.bfloat16' and saved['backend'] == 'eager'
            assert saved['model_digest'] == formal['cells'][0]['model_digest']
            assert len(saved['prompts']) == len(expected) == 480
            assert all(a['input_ids'] == b['input_ids'] and a['category'] == b['category']
                       for a, b in zip(saved['prompts'], expected))
            return saved

        def generate(decoder, prompt):
            wait_for_admission(system.assets_dir, initial=False)
            assert prompt[0].tolist() == expected[count[0]]['input_ids']
            output, stats = original_generate(decoder, prompt)
            count[0] += 1
            progress = {'completed': count[0], 'total': 480, 'BC': decoder.nominal_block_complexity,
                        'last_committed': stats['committed'], 'last_decode_calls': stats['decode_calls']}
            (Path(system.assets_dir) / 'revalidation_progress.json').write_text(json.dumps(progress) + '\n')
            if count[0] % 10 == 0:
                print('Nonofficial Table1', progress, flush=True)
            return output, stats

        with patch.object(evidence, 'capture', capture), patch.object(decoding.SpecDecoder, 'generate', generate):
            result = super().run(data, model, system, tracker)
        assert count[0] == 480
        check_refs()
        return result
