"""Merge a LoRA adapter into the base model and save full weights.

Useful before:
  - Converting to GGUF (which expects standalone weights)
  - Pushing to HuggingFace Hub as a single repo
  - Deploying with plain ``AutoModelForCausalLM`` (no PEFT dep)
"""

from __future__ import annotations

import logging
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

log = logging.getLogger(__name__)


def merge_adapter(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "cpu",
) -> PreTrainedModel:
    """Load the base model, apply & merge the LoRA adapter, save.

    Parameters
    ----------
    base_model_id:
        HF model ID of the original base model (e.g. ``Qwen/Qwen2.5-3B``
        or your CPT-uploaded model).
    adapter_path:
        Directory containing the saved LoRA adapter (``adapter_config.json``
        + ``adapter_model.safetensors``).
    output_dir:
        Where to write the merged model + tokenizer.
    torch_dtype:
        Cast to this precision for the merge (fp16 is usually sufficient).
    device_map:
        ``"cpu"`` to merge without GPU memory pressure.
    """
    log.info("Loading base model: %s", base_model_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    log.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)

    log.info("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    log.info("Merged model saved to %s", output_dir)
    return merged
