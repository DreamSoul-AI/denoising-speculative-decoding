"""Prompt text -> token ids [1, L] with the model's tokenizer."""


def render(tokenizer, text, apply_template=True, template_kwargs=None):
    if apply_template:
        encoded = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}],
            add_generation_prompt=True, return_tensors='pt',
            **dict(template_kwargs or {}))
        # transformers 5 returns a BatchEncoding; earlier versions a Tensor.
        return encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
    return tokenizer(text, return_tensors='pt').input_ids

"""SpecBench prompts (Xia et al., 2024), the paper's evaluation suite.

Expects the `question.jsonl` from the official Spec-Bench repo at
`datasets/spec_bench/question.jsonl` (copied into the Study `shared/data`
on prepare), one JSON object per line:

    {"question_id": ..., "category": "...", "turns": ["...", ...]}

Categories cover writing, roleplay, reasoning, math, coding, extraction,
stem, humanities (MT-Bench), plus translation (WMT14 DE-EN), summarization
(CNN/DM), qa (Natural Questions), math_reasoning (GSM8K) and rag. Prompts
are the first turn, rendered through the model's chat template (all paper
models are Instruct variants).
"""

import json
import os



def load_questions(path, categories=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'SpecBench question file not found: {path}\n'
            'Place it at datasets/spec_bench/question.jsonl')
    questions: list[dict] = []
    qids: set[str] = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 1024 * 1024:
                raise ValueError('prompt input exceeds bounded size')
            line = line.strip()
            if not line:
                continue
            if len(qids) >= 100000:
                raise ValueError('prompt input exceeds bounded size')
            item = json.loads(line)
            if (not isinstance(item, dict) or not isinstance(item.get('category'), str)
                    or not isinstance(item.get('turns'), list) or not item['turns']
                    or not all(isinstance(text, str) and text for text in item['turns'])
                    or type(item.get('question_id')) not in (str, int)):
                raise ValueError('invalid prompt record')
            qid = str(item['question_id'])
            if not qid or qid in qids:
                raise ValueError('duplicate or empty prompt identity')
            qids.add(qid)
            if categories and item['category'] not in categories:
                continue
            questions.append(item)
    return questions


def window_by_category(questions, skip=0, take=None):
    """Contiguous window inside each category.

    Categories are sorted by name. Within a category, order is file order.
    ``take`` None keeps the tail after ``skip``. first 2, next 4, and last 4
    are skip/take ``0/2``, ``2/4``, and ``6/4``.
    """
    if type(skip) is not int or skip < 0 or (take is not None and (type(take) is not int or take < 0)):
        raise ValueError('prompt windows require nonnegative integers')
    by_category: dict = {}
    for item in questions:
        by_category.setdefault(item['category'], []).append(item)
    picked = []
    for category in sorted(by_category):
        rows = by_category[category]
        window = rows[skip:] if take is None else rows[skip:skip + int(take)]
        if take is not None and len(window) < int(take):
            raise ValueError(
                f'{category} has {len(rows)} questions, need {skip + int(take)}')
        picked.extend(window)
    return picked


def read_specbench(data_cfg):
    path = data_cfg.get('path', 'datasets/spec_bench/question.jsonl')
    questions = load_questions(path, data_cfg.get('categories', None))
    questions = window_by_category(
        questions,
        skip=data_cfg.get('skip_per_category', 0),
        take=data_cfg.get('take_per_category', None),
    )
    return [{
        'question_id': item.get('question_id', None),
        'category': item['category'],
        'text': item['turns'][0],
    } for item in questions]


def encode_specbench(records, tokenizer, data_cfg):
    max_prompt_len = int(data_cfg.get('max_prompt_len', 2048))
    apply_template = bool(data_cfg.get('apply_chat_template', True))
    # e.g. {enable_thinking: false} for Qwen3.
    template_kwargs = dict(data_cfg.get('chat_template_kwargs', {}) or {})
    prompts = []
    for item in records:
        input_ids = render(tokenizer, item['text'], apply_template, template_kwargs)
        if input_ids.size(1) > max_prompt_len:
            continue
        prompts.append({
            'question_id': item['question_id'],
            'category': item['category'],
            'input_ids': input_ids,
        })
    return prompts

"""Deterministic prompts that need no download.

Two modes:
  - synthetic: random token-id sequences for the `tiny_llama` smoke model
    (which has no tokenizer). Seeded, so runs are reproducible.
  - text: a small builtin prompt list tokenized with the real tokenizer,
    for quick pretrained-model sanity checks without SpecBench.
"""

import torch


_TEXT_PROMPTS = [
    'Explain speculative decoding in two sentences.',
    'Write a haiku about parallel token prediction.',
    'What is the capital of France, and why is it famous?',
    'List three uses of dynamic programming.',
]


def is_synthetic(data_cfg):
    return str(data_cfg.get('mode', 'synthetic')).lower() == 'synthetic'


def read_local_prompts(data_cfg):
    mode = str(data_cfg.get('mode', 'synthetic')).lower()
    num_prompts = data_cfg.get('num_prompts', 8)
    if type(num_prompts) is not int or not 0 <= num_prompts <= 100000:
        raise ValueError('num_prompts outside bounded integer range')

    if mode == 'synthetic':
        vocab_size = data_cfg.get('vocab_size', 256)
        prompt_len = data_cfg.get('prompt_len', 32)
        if (type(vocab_size) is not int or vocab_size < 1 or type(prompt_len) is not int
                or prompt_len < 1 or prompt_len * num_prompts > 10000000):
            raise ValueError('invalid synthetic prompt shape/vocabulary')
        generator = torch.Generator().manual_seed(int(data_cfg.get('seed', 0)))
        return [{
            'question_id': i,
            'category': 'synthetic',
            'input_ids': torch.randint(
                0, vocab_size, (1, prompt_len), generator=generator),
        } for i in range(num_prompts)]

    if mode == 'text':
        return [{
            'question_id': i,
            'category': 'local',
            'text': _TEXT_PROMPTS[i % len(_TEXT_PROMPTS)],
        } for i in range(num_prompts)]

    raise ValueError(f'Unsupported local_prompts mode: {mode}')


def encode_local_prompts(records, tokenizer, data_cfg):
    if is_synthetic(data_cfg):
        return [dict(item) for item in records]
    apply_template = bool(data_cfg.get('apply_chat_template', True))
    return [{
        'question_id': item['question_id'],
        'category': item['category'],
        'input_ids': render(tokenizer, item['text'], apply_template),
    } for item in records]

"""Dolly-Databricks creative-writing prompts (Conover et al., 2023).

Used by the Figure 2 alignment probe: 100 creative-writing samples.
Requires the `datasets` package. Hugging Face / ModelScope caches go under the Study `shared/model`."""



def read_dolly(data_cfg):
    from datasets import load_dataset

    category = data_cfg.get('dolly_category', 'creative_writing')
    ds = load_dataset('databricks/databricks-dolly-15k', split='train')
    records = []
    for item in ds:
        if item['category'] != category:
            continue
        text = item['instruction']
        if item.get('context'):
            text = f"{item['context']}\n\n{text}"
        records.append({'category': category, 'text': text})
    return records


def encode_dolly(records, tokenizer, data_cfg):
    num_prompts = int(data_cfg.get('num_prompts', 100))
    max_prompt_len = int(data_cfg.get('max_prompt_len', 2048))
    prompts: list = []
    for item in records:
        input_ids = render(tokenizer, item['text'])
        if input_ids.size(1) > max_prompt_len:
            continue
        prompts.append({
            'question_id': len(prompts),
            'category': item['category'],
            'input_ids': input_ids,
        })
        if len(prompts) >= num_prompts:
            break
    return prompts

"""Prompt sets. Loading and tokenizing are separate steps.

`load_records(data_cfg)` reads raw prompts without a tokenizer (the RPipe
Data layer). `encode_prompts(records, tokenizer, data_cfg)` turns them into
{'question_id', 'category', 'input_ids' [1, L]} once the model's tokenizer is
known. `get_prompts(cfg)` composes both for the single-config CLI.

EOS ids come from the tokenizer (plus generation-config extras) unless the
dataset has no tokenizer (synthetic smoke prompts)."""

_REGISTRY = {
    'specbench': (read_specbench, encode_specbench),
    'local_prompts': (read_local_prompts, encode_local_prompts),
    'dolly': (read_dolly, encode_dolly),
}

def _module(data_cfg):
    name = str(data_cfg.get('name', 'specbench')).lower()
    if name not in _REGISTRY:
        raise ValueError(f'Unknown dataset: {name}. Available: {list(_REGISTRY)}')
    return _REGISTRY[name]

def needs_tokenizer(data_cfg):
    return not (str(data_cfg.get('name', 'specbench')).lower() == 'local_prompts' and is_synthetic(data_cfg))

def load_records(data_cfg):
    return _module(data_cfg)[0](data_cfg)

def encode_prompts(records, tokenizer, data_cfg):
    prompts = _module(data_cfg)[1](records, tokenizer, data_cfg)
    seen = set()
    for prompt in prompts:
        qid = prompt.get('question_id')
        ids = prompt.get('input_ids')
        if type(qid) not in (str, int) or str(qid) in seen:
            raise ValueError('invalid or duplicate encoded prompt identity')
        seen.add(str(qid))
        if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1
                or ids.dtype not in (torch.int32, torch.int64) or (ids < 0).any()):
            raise ValueError('invalid encoded prompt tokens')
    return prompts


def read_prompts(path, *, categories=None, skip=0, take=None):
    """Read bounded local SpecBench records; loading never downloads or creates a model."""
    return window_by_category(load_questions(path, categories), skip, take)

def load_tokenizer(name, local_files_only=False):
    from transformers import AutoTokenizer

    if name is None:
        return None
    return AutoTokenizer.from_pretrained(name, local_files_only=bool(local_files_only))


def eos_ids(tokenizer):
    if tokenizer is None:
        return []
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    # Instruct models often stop on extra ids (e.g. <|eot_id|> for LLaMA3).
    for name in ('eot_token_id',):
        extra = getattr(tokenizer, name, None)
        if extra is not None:
            ids.add(int(extra))
    return sorted(ids)


def get_prompts(cfg):
    _module(cfg.data)
    tokenizer = None
    if needs_tokenizer(cfg.data):
        name = cfg.data.get('tokenizer_name', None) or cfg.model.get('name_or_path', None)
        tokenizer = load_tokenizer(name, cfg.model.get('local_files_only', False))
    prompts = encode_prompts(load_records(cfg.data), tokenizer, cfg.data)
    return prompts, eos_ids(tokenizer)
