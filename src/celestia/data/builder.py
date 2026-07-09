"""Dataset building and tokenisation for CPT.

Core workflow:
  1. ``build_dataset(books)`` -- convert ``{name: text}`` -> HuggingFace Dataset
  2. ``tokenize_and_pack(dataset, tokenizer, max_seq_len)`` -- tokenise all text,
     concatenate, and chop into fixed-length sequences (no padding).
"""

from __future__ import annotations

import logging
import math
import random
import textwrap

from datasets import Dataset
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def build_dataset(books: dict[str, str]) -> Dataset:
    """Convert a ``{name: cleaned_text}`` dict into a HuggingFace Dataset.

    Each row stores one book's cleaned text and its name.
    Tokenisation happens separately to keep cleaning and tokenisation
    decoupled.
    """
    records = [
        {"text": text, "name": name}
        for name, text in books.items()
    ]
    dataset = Dataset.from_list(records)
    total_chars = sum(len(r["text"]) for r in records)
    log.info("Dataset created: %d books, ~%s chars", len(dataset), f"{total_chars:,}")
    return dataset


def tokenize_and_pack(
    dataset: Dataset,
    tokenizer,
    max_seq_len: int = 2048,
) -> Dataset:
    """Tokenise all books and pack tokens into fixed-length sequences.

    Steps
    -----
    1. Tokenise each book (no truncation, no padding).
    2. Concatenate every book's token IDs into one long 1-D array.
    3. Chop into chunks of *max_seq_len*, discarding the final
       incomplete chunk.

    This is the standard packing strategy for causal-LM pre-training:
    it eliminates padding and maximises throughput.
    """
    all_input_ids: list[int] = []

    for example in tqdm(dataset, desc="Tokenising + packing", unit="book"):
        encoded = tokenizer(
            example["text"],
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        all_input_ids.extend(encoded["input_ids"])

    total_tokens = len(all_input_ids)
    total_seqs = total_tokens // max_seq_len
    kept_tokens = total_seqs * max_seq_len
    discarded = total_tokens - kept_tokens

    log.info("  Total tokens : %s", f"{total_tokens:,}")
    log.info("  Sequences    : %s x %d", f"{total_seqs:,}", max_seq_len)
    log.info("  Discarded    : %s tail tokens (%.1f%%)",
             f"{discarded:,}", discarded / max(1, total_tokens) * 100)

    # Chop
    input_ids_list = [
        all_input_ids[i: i + max_seq_len]
        for i in range(0, kept_tokens, max_seq_len)
    ]

    # For CLM the labels are identical to input_ids (the shift happens
    # inside the model's forward pass automatically).
    packed = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": input_ids_list,
    })
    return packed


# ??????????????????????????????????????????????
#  Diagnostic helpers
# ??????????????????????????????????????????????

def print_dataset_stats(packed_dataset: Dataset, tokenizer) -> None:
    """Print human-readable statistics about the packed dataset."""
    n = len(packed_dataset)
    n_tokens = n * len(packed_dataset[0]["input_ids"]) if n > 0 else 0

    print()
    print("=" * 55)
    print("  DATASET STATISTICS")
    print("=" * 55)
    print(f"  Sequences        : {n:,}")
    print(f"  Tokens           : {n_tokens:,}")
    if n > 0:
        sample = packed_dataset[0]["input_ids"][:8]
        decoded = tokenizer.decode(sample)
        print(f"  Sample decode     : {decoded!r}...")
    print("=" * 55)


def estimate_training_time(
    num_sequences: int,
    batch_size: int,
    grad_accum: int,
    num_epochs: int,
    steps_per_second: float = 0.25,
) -> dict:
    """Estimate total training time and print a summary.

    *steps_per_second* is an empirical throughput value:
      - T4 single-GPU + Qwen2.5-3B + LoRA ? 0.25 step/s
      - 2xT4 DDP                                          ? 0.50 step/s
    """
    steps_per_epoch = max(1, math.ceil(num_sequences / (batch_size * grad_accum)))
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps / steps_per_second

    m, s = divmod(int(total_seconds), 60)
    h, m = divmod(m, 60)

    print()
    print("=" * 55)
    print("  TRAINING TIME ESTIMATE")
    print("=" * 55)
    print(f"  Sequences / epoch : {num_sequences:,}")
    print(f"  Effective batch   : {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"  Steps / epoch     : {steps_per_epoch:,}")
    print(f"  Total steps       : {total_steps:,}")
    print(f"  Epochs            : {num_epochs}")
    print(f"  Est. step rate    : {steps_per_second:.2f} step/s")
    print(f"  Est. duration     : {h}h {m:02d}m {s:02d}s")
    print(f"  Per epoch         : ~{h // num_epochs}h {m // num_epochs:02d}m")
    print("=" * 55)

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "total_seconds": total_seconds,
    }


def preview_random_chunk(packed_dataset: Dataset, tokenizer) -> None:
    """Decode and print a random training sequence."""
    idx = random.randint(0, len(packed_dataset) - 1)
    ids = packed_dataset[idx]["input_ids"]
    text = tokenizer.decode(ids, skip_special_tokens=True)

    print()
    print("=" * 55)
    print(f"  RANDOM TRAINING SEQUENCE  (index {idx})")
    print("=" * 55)
    print(textwrap.fill(text, width=100))
    print("=" * 55)
