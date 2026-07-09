# Celestia

A two-stage pipeline that **turns any base LLM into any historical figure** by fine-tuning on their original writings.

> The name comes from the original use case: turning Qwen into **Plato**. But the pipeline is fully generic. Drop in any corpus directory, change the model ID, and you're fine-tuning on Aristotle, Kant, Nietzsche, Shakespeare, Darwin, or anyone whose texts you can collect as `.txt` files.

## Pipeline

```
Raw .txt corpus (their complete works)
    ?
    ?
????????????????????????????
?  Stage 1: CPT             ?  Causal Language Modelling
?  (Continual Pre-Training) ?  QLoRA (4-bit NF4) + LoRA rank 16
?                           ?  The model absorbs their vocabulary,
?                           ?  style, sentence rhythms, and worldview
????????????????????????????
           ? LoRA adapter
           ?
????????????????????????????
?  Stage 2: SFT             ?  Chat-style instruction tuning
?  (Supervised Fine-Tuning) ?  Response-only loss (prompt masked)
?                           ?  Makes the model *answer as* them
????????????????????????????
           ? Merged weights
           ?
????????????????????????????
?  Export                   ?  HuggingFace Hub push
?                           ?  GGUF conversion (f16 -> Q4_K_M)
?                           ?  GGUF quantization for local inference
????????????????????????????
```

**What you need:**
- A base LLM (e.g. Qwen2.5-3B, Llama-3-8B, etc.)
- A corpus of `.txt` files (Project Gutenberg works great)
- An SFT dataset in JSONL format (conversations where the assistant speaks as your figure)
- A GPU with ?12 GB VRAM (one T4 works; 2xT4 with DDP is better)

## Quickstart

```bash
# Install
pip install -e .

# 1. Continual Pre-Training -- learns the figure's style from raw text
celestia cpt --config config/cpt.yaml

# 2. Supervised Fine-Tuning -- teaches it to *answer as* the figure
celestia sft --config config/sft.yaml

# 3. Merge adapter into base (required before GGUF export)
celestia merge --adapter /path/to/sft_adapter --output /path/to/merged

# 4. Convert to GGUF + quantize for local inference
celestia quantize --model /path/to/merged --outdir /path/to/gguf

# 5. Quick inference test
celestia generate --model /path/to/merged --prompt "What is justice?"
```

## Project Structure

```
src/celestia/
??? cli.py                 # CLI entry: 6 subcommands
??? config.py              # Pydantic models (CPTConfig, SFTConfig)
??? data/
?   ??? loader.py          # load_books + clean_text (Gutenberg removal)
?   ??? builder.py         # Dataset building, tokenise & pack
?   ??? sft.py             # SFT data loader, chat-template, collator
??? model/
?   ??? loader.py          # load_model, load_tokenizer, load_model_for_ddp
?   ??? lora.py            # apply_lora, load_adapter
?   ??? merge.py           # merge_and_unload
??? training/
?   ??? cpt.py             # CPT trainer (single-GPU + DDP)
?   ??? sft.py             # SFT trainer (response-only loss)
??? export/
?   ??? gguf.py            # llama.cpp conversion + quantization
?   ??? hub.py             # Push to HuggingFace Hub
??? utils/
    ??? logging.py         # Structured logging
    ??? memory.py          # GPU VRAM diagnostics
```

## Configuration

All hyperparameters live in YAML files. To retarget the entire pipeline at a **different person**, change just two things:

**`config/cpt.yaml`:**
```yaml
model_id: Qwen/Qwen2.5-3B         # your base model
corpus_dir: /data/nietzsche_txt    # directory of .txt files
```

**`config/sft.yaml`:**
```yaml
model_id: my/cpt-finished-model    # your CPT model (on HF or local)
dataset_path: /data/nietzsche_sft.jsonl
default_system_prompt: "You are Nietzsche."
```

That's it. The pipeline is corpus-agnostic.

## Hardware Notes

| GPU | VRAM | CPT | SFT | Notes |
|---|---|---|---|---|
| 1x T4 | 16 GB | Single GPU | Yes | SFT fits comfortably |
| 2x T4 | 32 GB | DDP (2x throughput) | Yes | Optimal for Kaggle |
| 1x A10 | 24 GB | Single GPU | Yes | Faster than T4 |
| CPU | -- | No | No | Quantization step only |

4-bit NF4 keeps the 3B model footprint at ~1.87 GB.

## Customising for a Different Figure

1. **Collect texts** -- Download works from Project Gutenberg as `.txt` files. Place them in `corpus_dir/`.
2. **Create SFT data** -- Build a JSONL where each line has `{"messages": [{"role": "system", "content": "You are X."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
3. **Update config/cpt.yaml** -- set `corpus_dir`.
4. **Update config/sft.yaml** -- set `dataset_path` and `default_system_prompt`.
5. **Run** -- `celestia cpt` then `celestia sft`.

The pipeline automatically handles Gutenberg boilerplate removal, text cleaning, sequence packing, chat-template tokenisation, and response masking.
