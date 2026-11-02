"""Count observed repeat-root exclusions; never generates tokens or loads a model.

    python studies/_shared/structure/analyze_esp_repeat_roots.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def count_repeats(observations):
    calls = accepted = repeats = nonterminal = prompts = 0
    for observation in observations:
        found = False
        for step in observation['stats']['trace']:
            calls += 1
            accepted += len(step['accepted_nodes'])
            if step['committed_tokens'][0] == observation['output_ids'][step['root_position']]:
                assert not step['accepted_nodes'], 'pruned single-mask tree cannot accept a parent repeat'
                repeats += 1
                found = True
                nonterminal += not step['stopped']
        prompts += found
    assert calls > 0
    return dict(calls=calls, accepted=accepted, repeat_root_fallbacks=repeats,
                nonterminal_repeat_root_fallbacks=nonterminal, prompts_with_repeat_root_fallback=prompts,
                repeat_fraction=repeats / calls, optimistic_extra_one_same_calls=nonterminal / calls)


def main():
    study = ROOT / 'studies/esp_table1_bf16_full_argmax'
    formal = json.loads((study / 'docs/protocol_evidence_v1.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT / 'docs/research/esp-table1-attribution-hypotheses-2026-10-16.json').read_text(encoding='utf-8'))
    result = {}
    for cell in formal['cells']:
        if cell['num_masks'] != 1:
            continue
        assert cell['actual_spec']['tree']['pruning'] and cell['argmax_all_passed']
        raw = study / 'runs' / cell['run_id'] / 'assets/decode_observations.json'
        blob = raw.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == cell['raw_sha256']
        observations = json.loads(blob)['out']['evidence']
        assert len(observations) == len(cell['rows']) == 480
        counts = count_repeats(observations)
        assert counts['calls'] == sum(r['decode_calls'] for r in cell['rows'])
        assert counts['accepted'] == sum(r['accepted'] for r in cell['rows'])
        expected = audit['pruning_observed_opportunity'][str(cell['BC'])]
        assert all(expected[key] == value for key, value in counts.items())
        result[str(cell['BC'])] = counts
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
