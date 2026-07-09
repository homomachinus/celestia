"""Pydantic-backed configuration loader.

Hyperparameters are defined in YAML and validated at load time
so that misconfigurations are caught before training starts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ??????????????????????????????????????????????
#  Shared fields
# ??????????????????????????????????????????????

_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: _LORA_TARGETS[:])


class BNBConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # "float16" | "bfloat16"


# ??????????????????????????????????????????????
#  CPT config
# ??????????????????????????????????????????????

class CPTConfig(BaseModel):
    model_id: str = Field(description="HuggingFace model ID (base, not instruct)")
    corpus_dir: str = Field(description="Directory containing .txt books")
    output_dir: str = Field(description="Where to save adapter + checkpoints")

    # Training
    max_seq_len: int = 2048
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    max_grad_norm: float = 1.0

    # Multi-GPU
    use_ddp: bool = True
    num_gpus: int = 2
    ddp_port: str = "12355"

    # Adapter
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quant: BNBConfig = Field(default_factory=BNBConfig)

    # Misc
    seed: int = 42
    eval_split: float = 0.05

    @model_validator(mode="after")
    def _derive_grad_accum(self) -> "CPTConfig":
        """Auto-derive grad_accum if only effective_batch is given."""
        return self

    @property
    def effective_batch_size(self) -> int:
        gpu_count = self.num_gpus if self.use_ddp else 1
        return gpu_count * self.batch_size * self.grad_accum


# ??????????????????????????????????????????????
#  SFT config
# ??????????????????????????????????????????????

class SFTConfig(BaseModel):
    model_id: str = Field(description="Your CPT-merged model on HF or local")
    output_dir: str = Field(description="Where to save SFT adapter, merge, GGUF")
    dataset_path: str = Field(description="Path to .jsonl with messages")

    # Training
    max_seq_len: int = 1024
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 50
    max_grad_norm: float = 1.0

    # Adapter
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quant: BNBConfig = Field(default_factory=BNBConfig)

    # SFT specific
    default_system_prompt: str = "You are Plato."
    eval_split: float = 0.05

    # Post-training
    push_to_hub: bool = False
    hub_repo_id: str = ""
    hub_private: bool = True
    convert_gguf: bool = False
    gguf_quant_type: str = "Q4_K_M"  # or "f16"

    # Misc
    seed: int = 42

    @model_validator(mode="after")
    def _check_hub(self) -> "SFTConfig":
        if self.push_to_hub and not self.hub_repo_id:
            raise ValueError("hub_repo_id required when push_to_hub=True")
        return self


# ??????????????????????????????????????????????
#  Loaders
# ??????????????????????????????????????????????

def load_cpt_config(path: str | os.PathLike) -> CPTConfig:
    """Load and validate a CPT YAML config."""
    raw = _read_yaml(path)
    return CPTConfig(**raw)


def load_sft_config(path: str | os.PathLike) -> SFTConfig:
    """Load and validate an SFT YAML config."""
    raw = _read_yaml(path)
    return SFTConfig(**raw)


def _read_yaml(path: str | os.PathLike) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
