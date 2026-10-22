"""Inputs shared by every evaluation: model on device, prompt slice, EOS ids."""

from typing import Any

import torch

from spec_mtp.data import get_prompts
from spec_mtp.model import build_model
from spec_mtp.model import get_device


def make_generator(device: Any, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def select_prompts(prompts: list, data_eos: Any, eval_cfg: Any) -> tuple[list, Any]:
    num_eval = int(eval_cfg.get('num_eval_prompts', len(prompts)))
    eos_ids = eval_cfg.get('eos_token_ids', None) or data_eos
    return prompts[:num_eval], eos_ids


def place_model(model: Any, device: Any) -> Any:
    model = model.to(device)
    model.eval()
    return model


def legacy_inputs(cfg: Any, model: Any = None) -> tuple[Any, list, Any, Any]:
    """Single-config CLI path: build the model, then tokenize its prompts."""
    device = get_device(cfg.eval.get('device', 'auto'))
    if model is None:
        model = build_model(cfg)
    model = place_model(model, device)
    prompts, data_eos = get_prompts(cfg)
    prompts, eos_ids = select_prompts(prompts, data_eos, cfg.eval)
    return model, prompts, eos_ids, device
