"""Chat-style data loading and tokenisation for SFT.

Loads a JSONL file where each line is:
.. code:: json
   {
     "messages": [
       {"role": "system", "content": "…"},
       {"role": "user",   "content": "…"},
       {"role": "assistant", "content": "…"}
     ]
   }

Key features:
  - Ensures every example has a system message (injects default if missing).
  - Uses the model's chat template for consistent formatting.
  - Masks the *prompt* portion of the labels (``-100``) so the loss is
    computed only on the assistant's response.
  - Dynamic padding collator for variable-length sequences.
"""

from __future__ import annotations

import json
import logging

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load every non-empty line of a JSONL file into a list of dicts."""
    examples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    log.info("Loaded %d raw examples from %s", len(examples), path)
    return examples


def normalize_messages(
    example: dict,
    default_system_prompt: str = "You are Plato.",
) -> dict:
    """Ensure every example has a system message.

    If no message has ``role == "system"``, prepend the default.
    """
    messages = example["messages"]
    has_system = any(m["role"] == "system" for m in messages)
    if not has_system:
        messages = [{"role": "system", "content": default_system_prompt}] + messages
    return {"messages": messages}


def build_sft_dataset(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    default_system_prompt: str = "You are Plato.",
    eval_split: float = 0.05,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load, normalise, tokenise, and split an SFT dataset.

    Returns ``(train_dataset, eval_dataset)`` with columns:
      - ``input_ids``
      - ``attention_mask``
      - ``labels``   (prompt tokens masked to -100)
    """
    raw = load_jsonl(jsonl_path)

    # Normalise system messages
    normalized = [normalize_messages(ex, default_system_prompt) for ex in raw]
    dataset = Dataset.from_list(normalized)
    log.info("Dataset ready: %d examples", len(dataset))

    # Tokenise with label masking
    def _tokenize(example: dict) -> dict:
        messages = example["messages"]

        # Full conversation (system + user + assistant)
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        # Prompt only (everything *before* the assistant reply)
        prompt_only = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )

        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]

        prompt_len = len(prompt_ids)

        # Truncate if necessary
        full_ids = full_ids[:max_seq_len]

        labels = full_ids.copy()
        # Mask prompt tokens: -100 is ignored by CrossEntropyLoss
        mask_len = min(prompt_len, len(labels))
        labels[:mask_len] = [-100] * mask_len

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    tokenized = dataset.map(
        _tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenising + masking",
    )
    log.info("Tokenised %d examples", len(tokenized))

    split = tokenized.train_test_split(test_size=eval_split, seed=seed)
    log.info("Train: %d  |  Eval: %d", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


# ──────────────────────────────────────────────
#  Dynamic padding collator
# ──────────────────────────────────────────────

def collate_fn(
    batch: list[dict],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the longest in the batch."""
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, attention_mask, labels = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
