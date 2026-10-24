"""Resolve tokenized prompts and EOS ids for an RPipe Run."""

from spec_mtp.data import encode_prompts, eos_ids, load_tokenizer, needs_tokenizer


def resolve_prompts(data, model):
    loader = data._loaders.get('test')
    items = list(loader) if loader is not None else []
    data_cfg = dict(data.meta.get('data_cfg') or {})
    if items and isinstance(items[0], dict) and 'input_ids' in items[0]:
        return items
    if not items:
        return []
    if not needs_tokenizer(data_cfg):
        return items
    model_cfg = dict(model.meta.get('model_cfg') or {})
    name = data_cfg.get('tokenizer_name') or model_cfg.get('name_or_path')
    tokenizer = load_tokenizer(name, model_cfg.get('local_files_only', False))
    data.meta['tokenizer'] = {'name': name, 'commit': tokenizer.init_kwargs.get('_commit_hash'),
                              'class': type(tokenizer).__name__, 'vocab': tokenizer.get_vocab(),
                              'chat_template': tokenizer.chat_template,
                              'special_tokens': tokenizer.special_tokens_map}
    return encode_prompts(items, tokenizer, data_cfg)


def resolve_eos(data, model, eval_cfg):
    explicit = eval_cfg.get('eos_token_ids', None)
    if explicit:
        return explicit
    cached = data.meta.get('eos_ids')
    if cached is not None:
        return cached
    data_cfg = dict(data.meta.get('data_cfg') or {})
    if not needs_tokenizer(data_cfg):
        return []
    model_cfg = dict(model.meta.get('model_cfg') or {})
    name = data_cfg.get('tokenizer_name') or model_cfg.get('name_or_path')
    tokenizer = load_tokenizer(name, model_cfg.get('local_files_only', False))
    return eos_ids(tokenizer)
