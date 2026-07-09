"""SFT trainer — Supervised Fine-Tuning with response-only loss.

Entry point: :func:`run_sft`
"""

from __future__ import annotations

import json
import logging
import os

import torch
from transformers import Trainer, TrainingArguments, set_seed

from celestia.config import SFTConfig
from celestia.data.sft import build_sft_dataset, collate_fn
from celestia.model.loader import load_model, load_tokenizer
from celestia.model.lora import apply_lora

log = logging.getLogger(__name__)


def run_sft(cfg: SFTConfig) -> None:
    """Run the full Supervised Fine-Tuning pipeline."""
    set_seed(cfg.seed)

    # ── Tokenizer & Model ──────────────────────────────────
    tokenizer = load_tokenizer(cfg.model_id)
    model = load_model(cfg.model_id, quant_cfg=cfg.quant, device_map="auto")
    model = apply_lora(model, cfg.lora)

    # ── Data ────────────────────────────────────────────────
    train_dataset, eval_dataset = build_sft_dataset(
        jsonl_path=cfg.dataset_path,
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        default_system_prompt=cfg.default_system_prompt,
        eval_split=cfg.eval_split,
        seed=cfg.seed,
    )

    # ── Collator (closure over pad_token_id) ────────────────
    def _collate(batch):
        return collate_fn(batch, pad_token_id=tokenizer.pad_token_id)

    # ── Training args ───────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=True,
        max_grad_norm=cfg.max_grad_norm,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        log_level="info",
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=cfg.save_steps,
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    # ── Trainer ─────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_collate,
    )

    log.info("Starting SFT training…")
    trainer.train()
    log.info("SFT training completed!")

    # ── Save adapter ────────────────────────────────────────
    adapter_path = os.path.join(cfg.output_dir, "sft_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(cfg.output_dir)

    log.info("SFT adapter saved to %s", adapter_path)
    log.info("Tokenizer saved to %s", cfg.output_dir)

    # ── Training metadata ───────────────────────────────────
    meta = {
        "base_model": cfg.model_id,
        "method": "QLoRA SFT",
        "dataset": cfg.dataset_path,
        "num_epochs": cfg.num_epochs,
        "max_seq_len": cfg.max_seq_len,
        "effective_batch_size": cfg.batch_size * cfg.grad_accum,
        "lora_r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "default_system_prompt": cfg.default_system_prompt,
        "completed_at": __import__("datetime").datetime.now().isoformat(),
    }
    meta_path = os.path.join(cfg.output_dir, "sft_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("SFT metadata saved to %s", meta_path)
