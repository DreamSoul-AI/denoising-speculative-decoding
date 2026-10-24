import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]


def auditor(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    spec = importlib.util.spec_from_file_location('audit_esp_third_party', ROOT / 'tools/audit_esp_third_party.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_files_hash_and_compile(monkeypatch):
    m = auditor(monkeypatch)
    _, origin = m.load_original()
    assert len(origin['files']) == 8
    for path in (m.VENDOR / 'upstream/code').glob('*.py'):
        compile(path.read_bytes(), str(path), 'exec')


def test_recorded_preflight_cannot_be_reported_as_valid_tau(historical_file):
    report = json.loads(historical_file('docs/research/esp-third-party-audit-2026-10-15.json').read_text(encoding='utf-8'))
    assert not report['author_official'] and not report['quality_passed'] and not report['full_480_executed']
    assert [r['bc'] for r in report['rows']] == [10, 30, 60]
    assert sum(r['completed_model_calls'] for r in report['rows']) == 6
    for row in report['rows']:
        assert row['status'] == 'verifier_argmax_failed' and row['valid_tau'] is None
        assert row['stopped_before_bad_commit'] and row['first_rejection']['output_index'] == 0
        assert row['first_rejection']['argmax_minus_chosen_logit'] > 0
    for path, key in [('tools/audit_esp_third_party.py', 'tool_sha256'),
                      ('studies/esp_table1_bf16_full_argmax/docs/THIRD_PARTY_PLAN.md', 'plan_sha256')]:
        original = subprocess.check_output(['git', 'show', f'aed4eee23ea5e6b9772b388eec9aa50a38452089:{path}'], cwd=ROOT)
        candidates = (original, original.replace(b'\n', b'\r\n'))
        assert report[key] in {hashlib.sha256(data).hexdigest() for data in candidates}


def test_audit_uses_prefix_parent_and_limits_remaining(monkeypatch):
    m = auditor(monkeypatch)
    logits = torch.zeros(5, 6, dtype=torch.bfloat16)
    logits[1, 1] = logits[2, 4] = 2
    records = m.audit_step(logits, 2, [1, 2], (1, 0, 4, 0.), 2)
    assert all(r['passed'] for r in records)
    assert [r['verifier_row'] for r in records] == [1, 2]
    assert len(m.audit_step(logits, 2, [1, 2], (1, 0, 4, 0.), 1)) == 1
    logits[1, 0] = 3
    assert not m.audit_step(logits, 2, [1, 2], (1, 0, 4, 0.), 2)[0]['passed']


def test_unchanged_source_self_match_is_not_prefix_acceptance(monkeypatch):
    m = auditor(monkeypatch)
    original, _ = m.load_original()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(6, 3).bfloat16()
            self.logits = torch.zeros(1, 5, 6, dtype=torch.bfloat16)
            self.logits[0, 1, 0] = self.logits[0, 2, 1] = 2
            self.logits[0, 3, 3] = self.logits[0, 4, 4] = 2

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, **kwargs):
            assert kwargs['inputs_embeds'].dtype == torch.bfloat16
            return SimpleNamespace(logits=self.logits, hidden_states=())

    model = Model()
    decoder = original.ESPDecoder(model, SimpleNamespace(pad_token_id=0, eos_token_id=5), 'cpu')
    prompt = torch.tensor([[0, 0]])
    result = decoder.verify_single_pass(prompt, [1, 2], decoder.get_mask_embedding(prompt))
    assert result[:3] == (1, 0, 4)
    records = m.audit_step(model.logits[0], 2, [1, 2], result, 2)
    assert [r['chosen'] for r in records] == [1, 4]
    assert [r['argmax'] for r in records] == [0, 1]
    assert not any(r['passed'] for r in records)
