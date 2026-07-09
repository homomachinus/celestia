"""CPT trainer -- Continual Pre-Training, single-GPU and DDP.

The primary entry point is :func:`run_cpt`, which:
  1. Loads + cleans + tokenises + packs a raw text corpus.
  2. Prints dataset statistics and time estimates.
  3. Launches training (inline or via ``notebook_launcher`` for DDP).

Works with any text corpus -- just point ``CPTConfig.corpus_dir`` at your
directory of ``.txt`` files.
"""

from __future__ import annotations
import datetime
import glob
import json
import logging
import os

import torch
from accelerate import notebook_launcher
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from celestia.config import CPTConfig
from celestia.data.builder import (
    build_dataset,
    estimate_training_time,
    print_dataset_stats,
    preview_random_chunk,
    tokenize_and_pack,
)
from celestia.data.loader import clean_text, load_books
from celestia.model.loader import load_model, load_model_for_ddp, load_tokenizer
from celestia.model.lora import apply_lora

log = logging.getLogger(__name__)


def _prepare_dataset(cfg: CPTConfig, tokenizer):
    """Load -> clean -> build -> tokenise -> pack.  Returns packed Dataset."""
    log.info("Step 1 -- Loading books ...")
    raw_books = load_books(cfg.corpus_dir)
    log.info("  Found %d book(s).\n", len(raw_books))

    log.info("Step 2 -- Cleaning text ...")
    cleaned_books = {
        name: clean_text(text, book_name=name)
        for name, text in raw_books.items()
    }

    log.info("Step 3 -- Building dataset ...")
    dataset = build_dataset(cleaned_books)

    log.info("Step 4 -- Tokenising & packing ...")
    packed = tokenize_and_pack(dataset, tokenizer, max_seq_len=cfg.max_seq_len)

    print_dataset_stats(packed, tokenizer)
    preview_random_chunk(packed, tokenizer)

    steps_ps = 0.50 if cfg.use_ddp else 0.25
    estimate_training_time(
        num_sequences=len(packed),
        batch_size=cfg.batch_size,
        grad_accum=cfg.grad_accum,
        num_epochs=cfg.num_epochs,
        steps_per_second=steps_ps,
    )

    return packed


def _train_single_gpu(cfg: CPTConfig, packed_dataset: Dataset, tokenizer) -> None:
    """Single-GPU / Colab-style training."""
    set_seed(cfg.seed)

    split = packed_dataset.train_test_split(test_size=cfg.eval_split, seed=cfg.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    log.info("Train sequences : %s", f"{len(train_dataset):,}")
    log.info("Eval  sequences : %s", f"{len(eval_dataset):,}")

    model = load_model(cfg.model_id, quant_cfg=cfg.quant)
    model = apply_lora(model, cfg.lora)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

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
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    checkpoints = sorted(glob.glob(os.path.join(cfg.output_dir, "checkpoint-*")))
    resume = checkpoints[-1] if checkpoints else None
    if resume:
        log.info("Resuming from %s", resume)
    else:
        log.info("Starting fresh training")
    trainer.train(resume_from_checkpoint=resume)

    _save_adapter(cfg, model, tokenizer)


def _train_ddp(cfg: CPTConfig, packed_dataset: Dataset, tokenizer) -> None:
    """Save data to disk and launch DDP via ``notebook_launcher``."""
    data_path = os.path.join(cfg.output_dir, "_train_data")
    packed_dataset.save_to_disk(data_path)
    log.info("Packed dataset saved to %s", data_path)

    set_seed(cfg.seed)

    # The function that runs on every DDP rank
    def train_per_rank():
        rank = int(os.environ["LOCAL_RANK"])
        is_main = rank == 0

        # Tokenizer (same on every rank)
        tok = load_tokenizer(cfg.model_id)

        # Full model on this rank's device
        model = load_model_for_ddp(cfg.model_id, cfg.quant, rank)
        model = apply_lora(model, cfg.lora)

        # Load dataset from disk
        packed = Dataset.load_from_disk(data_path)
        split = packed.train_test_split(test_size=cfg.eval_split, seed=cfg.seed)
        train_data = split["train"]
        eval_data = split["test"]

        if is_main:
            log.info("Rank 0: %d train / %d eval sequences",
                     len(train_data), len(eval_data))

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tok, mlm=False,
        )

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
            log_level="info" if is_main else "error",
            report_to=["tensorboard"] if is_main else [],
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=cfg.save_steps,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            seed=cfg.seed,
            ddp_backend="nccl",
            local_rank=rank,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
            processing_class=tok,
        )

        checkpoints = sorted(glob.glob(os.path.join(cfg.output_dir, "checkpoint-*")))
        resume = checkpoints[-1] if checkpoints else None
        if resume and is_main:
            log.info("Resuming from %s", resume)
        trainer.train(resume_from_checkpoint=resume)

        if is_main:
            _save_adapter(cfg, model, tok)

    log.info("Launching DDP training on %d GPUs...", cfg.num_gpus)
    notebook_launcher(
        train_per_rank,
        args=(),
        num_processes=cfg.num_gpus,
        mixed_precision="fp16",
        use_port=cfg.ddp_port,
    )
    log.info("DDP training completed!")


def _save_adapter(cfg: CPTConfig, model, tokenizer) -> None:
    """Save LoRA adapter, tokenizer, and training metadata."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    adapter_path = os.path.join(cfg.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    log.info("LoRA adapter saved to %s", adapter_path)

    tokenizer.save_pretrained(cfg.output_dir)
    log.info("Tokenizer saved to %s", cfg.output_dir)

    meta = {
        "base_model": cfg.model_id,
        "method": "QLoRA (4-bit NF4)" + (" + DDP" if cfg.use_ddp else ""),
        "hardware": f"{cfg.num_gpus}xGPU" if cfg.use_ddp else "single GPU",
        "objective": "Causal Language Modeling",
        "corpus_dir": cfg.corpus_dir,
        "num_epochs": cfg.num_epochs,
        "max_seq_len": cfg.max_seq_len,
        "effective_batch_size": cfg.effective_batch_size,
        "lora_r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "completed_at": datetime.datetime.now().isoformat(),
    }
    meta_path = os.path.join(cfg.output_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Training metadata saved to %s", meta_path)

    print()
    print("=" * 55)
    print("  SAVE COMPLETE")
    print("=" * 55)
    print(f"  LoRA adapter    : {adapter_path}/")
    print(f"  Tokenizer       : {cfg.output_dir}/")
    print(f"  Metadata        : {meta_path}")
    print("=" * 55)


# ??????????????????????????????????????????????
#  Public entry point
# ??????????????????????????????????????????????

def run_cpt(cfg: CPTConfig) -> None:
    """Run the full Continual Pre-Training pipeline."""
    tokenizer = load_tokenizer(cfg.model_id)
    packed = _prepare_dataset(cfg, tokenizer)

    if cfg.use_ddp and torch.cuda.device_count() >= 2:
        _train_ddp(cfg, packed, tokenizer)
    else:
        log.info("Single-GPU mode (DDP disabled or not enough GPUs)")
        _train_single_gpu(cfg, packed, tokenizer)
