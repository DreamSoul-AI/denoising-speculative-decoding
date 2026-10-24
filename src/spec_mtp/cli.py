"""Scientific decode and read-only audit/compare commands."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any


def _json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _decode(args: argparse.Namespace) -> dict[str, Any]:
    from spec_mtp.audit import JsonlSink
    from spec_mtp.contracts import DecodeRequest, Prompt
    from spec_mtp.decode import generate
    from spec_mtp.masks import build_mask_provider
    from spec_mtp.model import HFCausalLM

    body = _json(args.request)
    prompt = body.pop('prompt')
    prompt = Prompt(prompt['qid'], tuple(prompt['input_ids']), prompt.get('category', 'unknown'))
    request = DecodeRequest(prompt=prompt, **body)
    # No implicit Hub or device fallback; decode only the explicitly supplied local model.
    if not Path(args.model).is_dir():
        raise ValueError('--model must name a local model directory')
    model = None
    sink = None
    primary = None
    try:
        sink = JsonlSink(args.output)
        model = HFCausalLM(args.model, local_files_only=True)
        result = generate(request, model, build_mask_provider(request.spec, model.embedding_table), sink)
        return asdict(result)
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = []
        for resource in (sink, model):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    failures.append(error)
        if failures:
            if primary is None:
                raise failures[0]
            setattr(primary, 'cleanup_failures', failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='spec-mtp', description=__doc__)
    commands = parser.add_subparsers(dest='command')
    decode = commands.add_parser('decode', help='decode one request with an explicit local model')
    decode.add_argument('--request', required=True, help='DecodeRequest JSON')
    decode.add_argument('--model', required=True)
    decode.add_argument('--output', required=True, help='new JSONL observation path')
    audit = commands.add_parser('audit', help='audit existing observations, without a model')
    audit.add_argument('raw')
    audit.add_argument('--sources', required=True)
    compare = commands.add_parser('compare', help='compare frozen rows, without a model')
    compare.add_argument('left')
    compare.add_argument('right')
    compare.add_argument('--pairing', required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] and argv[0] in ('make', 'run', 'launch', 'process', 'report', 'study', 'status'):
        parser.error('Study execution moved to python -m studies._shared.flow.cli; scientific commands are decode/audit/compare')
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == 'decode':
        result = _decode(args)
    elif args.command == 'audit':
        from spec_mtp.audit import audit_run

        result = asdict(audit_run(args.raw, _json(args.sources)))
    else:
        from spec_mtp.analysis import compare as compare_records

        result = compare_records(_json(args.left), _json(args.right), _json(args.pairing))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if args.command == 'audit':
        return 0 if all(result[key] for key in ('source_integrity', 'argmax_coverage', 'argmax_correctness')) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
