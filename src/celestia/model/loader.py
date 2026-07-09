"""Model and tokenizer loading with 4-bit QLoRA configuration.

All BnB / device logic is centralised here so that the rest of the
pipeline doesn't need to import bitsandbytes directly.
"""

from __future__ import annotations

import logging
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from celestia.config import BNBConfig

log = logging.getLogger(__name__)


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer for *model_id*.

    Automatically sets ``pad_token = eos_token`` if the tokenizer has
    no pad token (common for causal LMs).
    """
    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("  Vocab size : %d", tokenizer.vocab_size)
    log.info("  Pad token  : %r (id=%s)", tokenizer.pad_token, tokenizer.pad_token_id)
    log.info("  EOS token  : %r (id=%s)", tokenizer.eos_token, tokenizer.eos_token_id)
    return tokenizer


def _build_bnb_config(cfg: BNBConfig) -> BitsAndBytesConfig:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=dtype_map.get(cfg.bnb_4bit_compute_dtype, torch.bfloat16),
    )


def load_model(
    model_id: str,
    quant_cfg: BNBConfig | None = None,
    device_map: str | dict | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> PreTrainedModel:
    """Load a causal LM with optional 4-bit quantisation.

    Parameters
    ----------
    model_id, quant_cfg:
        HF model ID and quantisation config.
    device_map:
        - ``None``: don't set (for DDP where placement happens by rank).
        - ``"auto"``: HF will split layers (single-GPU).
        - ``{"": rank}``: place full model on one device (DDP per-rank).
    """
    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if quant_cfg is not None:
        kwargs["quantization_config"] = _build_bnb_config(quant_cfg)
    if device_map is not None:
        kwargs["device_map"] = device_map

    log.info("Loading model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    mem = model.get_memory_footprint()
    log.info("Model memory footprint: %.2f GB", mem / 1024**3)
    return model


def load_model_for_ddp(
    model_id: str,
    quant_cfg: BNBConfig,
    rank: int,
) -> PreTrainedModel:
    """Load the **full** model on a single device identified by *rank*.

    This is the correct pattern for DDP: each GPU gets a replica of the
    entire model (no layer splitting).
    """
    device = torch.device(f"cuda:{rank}")
    model = load_model(
        model_id,
        quant_cfg=quant_cfg,
        device_map={"": rank},
        torch_dtype=torch.bfloat16,
    )
    log.info("DDP rank %d: model on %s", rank, device)
    return model
