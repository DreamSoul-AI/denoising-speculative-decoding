"""Pure cell comparison and the unchanged default fingerprint matrix."""

from studies._shared.structure import decode_fingerprint as fp


def test_compare_reports_field_level_differences():
    old = {'cells': {'a': {'tokens': [1, 2], 'accepted': 1}, 'b': {'tokens': [3]}}}
    new = {'cells': {'a': {'tokens': [1, 2], 'accepted': 2}, 'c': {'tokens': [3]}}}
    assert fp.compare(old, old) == []
    assert fp.compare(old, new) == ["a: ['accepted']", "b: ['tokens']", "c: ['tokens']"]


def test_fingerprint_default_matrix_and_seeds_are_unchanged_without_execution(monkeypatch):
    calls = []
    def cell(model, prompt, spec, config, seed):
        calls.append((model, prompt, config['temperature'], config['max_new_tokens'], seed))
        return {'fixture': len(calls)}
    monkeypatch.setattr(fp, 'fingerprint_cell', cell)
    report = fp.fingerprint({1024: 'long', 64: 'limited'}, ['p0', 'p1', 'p2'], 'cuda:0')
    assert report['count'] == len(fp.default_presets()) * 15
    assert {call[:4] for call in calls} == {
        (model, prompt, temperature, tokens) for model, tokens, temperatures in
        [('long', 32, [0., .7, 1.]), ('limited', 48, [0., 1.])]
        for temperature in temperatures for prompt in ('p0', 'p1', 'p2')}
    assert all(seed == 100 + int(prompt[1]) for _model, prompt, _temperature, _tokens, seed in calls)
