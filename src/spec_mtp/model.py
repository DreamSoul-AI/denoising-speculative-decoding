"""Explicit HF model sessions and their physical cache operations."""


import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def makedir_exist_ok(path, is_file=False):
    if is_file:
        path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)


def get_device(cfg_device):
    if cfg_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg_device


def get_dtype(tag):
    tag = str(tag).lower()
    if tag in ('auto', 'none'):
        return 'auto'
    mapping = {
        'float32': torch.float32, 'fp32': torch.float32,
        'float16': torch.float16, 'fp16': torch.float16,
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
    }
    if tag not in mapping:
        raise ValueError(f'Unsupported torch_dtype: {tag}')
    return mapping[tag]


def sync_device(device):
    """Synchronize before wall-clock timing so speedup ratios are honest."""
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


from collections.abc import Iterator
from typing import Any

def new_cache() -> Any:
    from transformers.cache_utils import DynamicCache

    return DynamicCache()


def _layer_tensors(cache: Any) -> Iterator[tuple[Any, str, str]]:
    """Yields (get_k, get_v, set_k, set_v) accessors per layer."""
    if hasattr(cache, 'layers'):          # transformers >= 5
        for layer in cache.layers:
            yield layer, 'keys', 'values'
    elif hasattr(cache, 'key_cache'):     # transformers 4.x
        class _Shim:
            def __init__(self, cache: Any, idx: int) -> None:
                self._c, self._i = cache, idx

            @property
            def keys(self) -> Any:
                return self._c.key_cache[self._i]

            @keys.setter
            def keys(self, t: Any) -> None:
                self._c.key_cache[self._i] = t

            @property
            def values(self) -> Any:
                return self._c.value_cache[self._i]

            @values.setter
            def values(self, t: Any) -> None:
                self._c.value_cache[self._i] = t

        for idx in range(len(cache.key_cache)):
            yield _Shim(cache, idx), 'keys', 'values'
    else:
        raise TypeError(f'Unsupported cache type: {type(cache)}')


def cache_len(cache: Any) -> int:
    return cache.get_seq_length()


def crop(cache: Any, keep_len: int) -> None:
    """Drop all cache entries from position keep_len onward."""
    if hasattr(cache, 'crop'):
        cache.crop(keep_len)
        return
    for layer, kname, vname in _layer_tensors(cache):
        setattr(layer, kname, getattr(layer, kname)[:, :, :keep_len])
        setattr(layer, vname, getattr(layer, vname)[:, :, :keep_len])


def keep_block_positions(cache: Any, prefix_len: int, block_keep_indices: list[int]) -> None:
    """Rewrite cache to prefix + the block entries at `block_keep_indices`
    (block-relative, in commit order). Everything else is evicted."""
    if len(block_keep_indices) == 0:
        crop(cache, prefix_len)
        return
    device_index = None
    for layer, kname, vname in _layer_tensors(cache):
        keys = getattr(layer, kname)
        values = getattr(layer, vname)
        if device_index is None or device_index.device != keys.device:
            device_index = torch.tensor(
                block_keep_indices, dtype=torch.long, device=keys.device) + prefix_len
        setattr(layer, kname, torch.cat(
            [keys[:, :, :prefix_len], keys[:, :, device_index]], dim=2))
        setattr(layer, vname, torch.cat(
            [values[:, :, :prefix_len], values[:, :, device_index]], dim=2))


import torch.nn as nn



class HFCausalLM(nn.Module):

    def __init__(self, name_or_path=None, local_files_only=True, torch_dtype='auto',
                 attn_implementation='eager', tiny_config=None, cache_dir=None, *, module=None):
        super().__init__()
        self.hf: Any
        self._owns_model = module is None
        self.closed = False
        if attn_implementation not in ('eager', 'sdpa'):
            raise ValueError('tree attention requires eager or sdpa')
        if module is not None:
            self.hf = module
        elif tiny_config is not None:
            from transformers import LlamaConfig, LlamaForCausalLM
            config = LlamaConfig(  # type: ignore[call-arg]  # attn_implementation goes to PretrainedConfig kwargs
                vocab_size=int(tiny_config.get('vocab_size', 256)),
                tie_word_embeddings=bool(tiny_config.get('tie_word_embeddings', False)),
                hidden_size=int(tiny_config.get('hidden_size', 64)),
                intermediate_size=int(tiny_config.get('intermediate_size', 128)),
                num_hidden_layers=int(tiny_config.get('num_layers', 2)),
                num_attention_heads=int(tiny_config.get('num_heads', 4)),
                num_key_value_heads=int(tiny_config.get('num_kv_heads', 2)),
                max_position_embeddings=int(tiny_config.get('max_position_embeddings', 512)),
                attn_implementation=attn_implementation,
            )
            self.hf = LlamaForCausalLM(config)
        else:
            from transformers import AutoModelForCausalLM
            kwargs = dict(
                local_files_only=bool(local_files_only),
                attn_implementation=attn_implementation,
            )
            if cache_dir:
                kwargs['cache_dir'] = cache_dir
            try:  # transformers >= 4.56 prefers `dtype`; older only knows `torch_dtype`
                self.hf = AutoModelForCausalLM.from_pretrained(
                    name_or_path, dtype=get_dtype(torch_dtype), **kwargs)
            except TypeError:
                self.hf = AutoModelForCausalLM.from_pretrained(
                    name_or_path, torch_dtype=get_dtype(torch_dtype), **kwargs)
        self.hf.eval()
        self.config = self.hf.config
        self.vocab_size = self.config.vocab_size
        self.max_positions = getattr(self.config, 'max_position_embeddings', 4096)

    @property
    def embedding_table(self):
        """Input embedding matrix E in R^{V x d} (Section 3.1)."""
        return self.hf.get_input_embeddings().weight

    @property
    def tied_embeddings(self):
        return self.hf.get_output_embeddings().weight.data_ptr() == self.embedding_table.data_ptr()

    new_cache = staticmethod(new_cache)
    cache_len = staticmethod(cache_len)
    crop_cache = staticmethod(crop)
    keep_cache = staticmethod(keep_block_positions)

    def close(self):
        """Release only an owned model; never destroy a borrowed module."""
        if not self.closed:
            self.closed = True
            if self._owns_model:
                self.hf = None

    def describe(self):
        """Actual loaded session facts, independent of caller source labels."""
        return {'model_class': type(self.hf).__name__, 'config': self.config.to_dict(),
                'attention_backend': self.config._attn_implementation,
                'device': str(self.device), 'dtype': str(self.dtype),
                'vocab_size': self.vocab_size, 'max_positions': self.max_positions,
                'tied_embeddings': self.tied_embeddings}

    @property
    def device(self):
        return self.embedding_table.device

    @property
    def dtype(self):
        return self.embedding_table.dtype

    def embed(self, input_ids):
        """Token ids -> input embeddings e_i = E[x_i]."""
        return self.hf.get_input_embeddings()(input_ids)

    @torch.no_grad()
    def forward(self, inputs_embeds, position_ids, attention_mask_4d=None,
                past_key_values=None, output_hidden_states=False, last_hidden_only=False):
        """One forward pass over a (tree-)block of embeddings.

        attention_mask_4d: additive float mask [1, 1, q_len, kv_len] with 0 for
        attend and dtype-min for blocked, or None for the plain causal prefill
        path. Returns the full model output (logits at every block position).
        """
        if self.closed:
            raise RuntimeError('model session is closed')
        captured = []
        handle = None
        if last_hidden_only:
            # Llama/Qwen expose their final normalization here. Capture only
            # its output; request no intermediate layer hidden-state tuple.
            norm = getattr(getattr(self.hf, 'model', None), 'norm', None)
            if norm is None or output_hidden_states:
                raise ValueError('last_hidden_only requires Llama/Qwen final norm and no all-layer output')
            handle = norm.register_forward_hook(lambda _module, _args, output: captured.append(output))
        try:
            out = self.hf(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask_4d,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
                output_hidden_states=output_hidden_states,
            )
            if last_hidden_only:
                if len(captured) != 1:
                    raise RuntimeError('expected one final norm observation')
                out.last_hidden_state = captured[0]
            return out
        finally:
            if handle is not None:
                handle.remove()

    def min_mask_value(self):
        return torch.finfo(self.dtype).min


def build_model(cfg):
    model_cfg = cfg.model
    model_type = str(model_cfg.get('type', 'hf_causal')).lower()
    if model_type == 'tiny_llama':
        return HFCausalLM(
            attn_implementation=model_cfg.get('attn_implementation', 'eager'),
            tiny_config=model_cfg.get('tiny', {}),
        )
    if model_type == 'hf_causal':
        return HFCausalLM(
            name_or_path=model_cfg.name_or_path,
            local_files_only=model_cfg.get('local_files_only', False),
            torch_dtype=model_cfg.get('torch_dtype', 'auto'),
            attn_implementation=model_cfg.get('attn_implementation', 'eager'),
            cache_dir=model_cfg.get('cache_dir'),
        )
    raise ValueError(f'Unsupported model.type: {model_type}')
