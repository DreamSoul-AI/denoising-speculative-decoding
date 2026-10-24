"""Observe the unchanged T0 decoder's verification choices, scoped to this Study."""
import hashlib
import json
from contextlib import contextmanager
from unittest.mock import patch
import spec_mtp.decode as decoding


def token_digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def bind_committed(prompt_ids, output_ids, stats, selections):
    """Check forward/row ownership and EOS/budget-trimmed commitment coverage."""
    assert len(selections) == 1 + sum(len(s['accepted_nodes']) + 1 for s in stats['trace'])
    assert selections[0]['forward'] == 0 and selections[0]['row'] == len(prompt_ids) - 1
    committed = [selections[0]]
    offset = 1
    for step, trace in enumerate(stats['trace'], 1):
        count = len(trace['accepted_nodes']) + 1
        block = selections[offset:offset + count]
        assert [s['forward'] for s in block] == [step] * count
        assert [s['row'] for s in block] == [0] + trace['accepted_nodes']
        assert trace['committed_tokens'] == [s['chosen'] for s in block[:len(trace['committed_tokens'])]]
        committed.extend(block[:len(trace['committed_tokens'])])
        offset += count
    assert output_ids[:len(prompt_ids)] == prompt_ids
    assert output_ids[len(prompt_ids):] == [s['chosen'] for s in committed]
    assert len(committed) == stats['new_tokens'] == 1 + stats['committed']
    assert all(s['chosen'] == s['argmax'] for s in selections)
    return {'passed': True, 'checked_committed_tokens': len(committed),
            'selected_tokens_including_trimmed_tail': len(selections),
            'input_sha256': token_digest(prompt_ids), 'output_sha256': token_digest(output_ids),
            'committed': committed}


@contextmanager
def observe_argmax(assets):
    original_generate = decoding.SpecDecoder.generate
    original_select = decoding.select_token
    audit_path = assets / 'argmax_audit.jsonl'
    assert not audit_path.exists(), 'never append a new execution to old audit evidence'
    with audit_path.open('x', encoding='utf-8') as stream:
        def generate(decoder, prompt_ids):
            assert decoder.temperature == 0 and decoder.model.dtype.__str__() == 'torch.bfloat16'
            latest = [None]
            forward = [-1]
            selections = []
            def capture(_module, _args, output):
                latest[0] = output.logits[0]
                forward[0] += 1
            def select(row, temperature, generator):
                assert temperature == 0
                matrix = latest[0]
                assert matrix is not None and row.ndim == 1 and row.numel() == matrix.shape[1]
                offset = (row.data_ptr() - matrix.data_ptr()) // row.element_size()
                assert offset >= 0 and offset % matrix.stride(0) == 0
                index = offset // matrix.stride(0)
                assert index < matrix.shape[0] and row.stride(0) == matrix.stride(1)
                chosen = original_select(row, temperature, generator)
                expected = int(row.argmax(dim=-1).item())
                assert chosen == expected, 'verification choice differs from its forward argmax'
                selections.append({'forward': forward[0], 'row': index, 'chosen': chosen, 'argmax': expected})
                return chosen
            handle = decoder.model.register_forward_hook(capture)
            try:
                with patch.object(decoding, 'select_token', select):
                    output, stats = original_generate(decoder, prompt_ids)
                assert forward[0] == stats['decode_calls']
                audit = bind_committed(prompt_ids[0].tolist(), output[0].tolist(), stats, selections)
                stats['verification_argmax_audit'] = audit
                stream.write(json.dumps(audit) + '\n')
                stream.flush()
                return output, stats
            finally:
                handle.remove()
                latest[0] = None
        with patch.object(decoding.SpecDecoder, 'generate', generate):
            yield
