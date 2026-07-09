"""LoRA adapter application.

Wraps a 4-bit base model with PEFT LoRA.  Only the low-rank matrices
are trainable; the quantised base weights stay frozen.
"""

from __future__ import annotations

import logging

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel
from transformers import PreTrainedModel

from celestia.config import LoRAConfig

log = logging.getLogger(__name__)


def apply_lora(model: PreTrainedModel, cfg: LoRAConfig) -> PreTrainedModel:
    """Wrap *model* with a LoRA adapter.

    Steps:
      1. ``prepare_model_for_kbit_training`` -- casts certain modules
         to fp32 for gradient stability under 4-bit.
      2. ``get_peft_model`` -- injects trainable LoRA matrices into
         the target modules.

    Returns the PEFT-wrapped model (same interface as ``PreTrainedModel``).
    """
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_adapter(
    base_model: PreTrainedModel,
    adapter_path: str,
) -> PeftModel:
    """Load a previously saved LoRA adapter on top of *base_model*."""
    log.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model
