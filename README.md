# Celestia

**Turn any base LLM into any historical figure** via two-stage fine-tuning on their original writings.

Starting with a base model (Qwen2.5-3B, Llama-3-8B, etc.), Celestia runs:
1. **CPT** — Continual Pre-Training on raw text (learns their vocabulary, sentence rhythm, and worldview)
2. **SFT** — Supervised Fine-Tuning on chat-style data (teaches them to *answer as* that figure)

The corpus is just `.txt` files. Change `corpus_dir` and `model_id` in two YAML files and you're training on Aristotle, Kant, Nietzsche, Shakespeare, or anyone whose complete works you can collect.

---

## Pipeline

```
Raw .txt corpus (complete works)
       |
       v
+--------------------------+
|  Stage 1: CPT            |  Causal Language Modelling
|  (Continual Pre-Training)|  QLoRA (4-bit NF4) + LoRA rank 16
|                          |  The model absorbs their style and knowledge
+------------+-------------+
             | LoRA adapter
             v
+--------------------------+
|  Stage 2: SFT            |  Chat-style instruction tuning
|  Supervised Fine-Tuning  |  Response-only loss (prompt masked)
|                          |  Makes the model *answer as* them
+------------+-------------+
             | Merged weights
             v
+--------------------------+
|  Export                  |  HuggingFace Hub push
|                          |  GGUF conversion (f16 -> Q4_K_M)
|                          |  Local inference via gguf
+--------------------------+
```

---

## What You Need

- A base LLM (e.g. `Qwen/Qwen2.5-3B`, `meta-llama/Llama-3-8B`)
- A corpus of `.txt` files (Project Gutenberg works great)
- An SFT dataset in JSONL format (conversations where the assistant speaks as your figure)
- A GPU with >=12 GB VRAM (one T4 works; 2xT4 with DDP is better)

---

## Quickstart

```bash
# Install
pip install -e .

# 1. Continual Pre-Training -- learns the figure's style from raw text
celestia cpt --config config/cpt.yaml

# 2. Supervised Fine-Tuning -- teaches it to answer as the figure
celestia sft --config config/sft.yaml

# 3. Merge adapter into base (required before GGUF export)
celestia merge --adapter /path/to/sft_adapter --output /path/to/merged

# 4. Convert to GGUF + quantize for local inference
celestia quantize --model /path/to/merged --outdir /path/to/gguf

# 5. Quick inference test
celestia generate --model /path/to/merged --prompt "What is justice?"
```

---

## Project Structure

```
src/celestia/
  cli.py               CLI entry: 6 subcommands
  config.py            Pydantic models (CPTConfig, SFTConfig)
  data/
    loader.py          load_books + clean_text (Gutenberg removal)
    builder.py         Dataset building, tokenise & pack
    sft.py             SFT data loader, chat-template, collator
  model/
    loader.py          load_model, load_tokenizer, load_model_for_ddp
    lora.py            apply_lora, load_adapter
    merge.py           merge_and_unload (optional)
  training/
    cpt.py             CPT trainer (single-GPU + DDP)
    sft.py             SFT trainer (response-only loss)
  export/
    gguf.py            llama.cpp conversion + quantization
    hub.py             Push to HuggingFace Hub
  utils/
    logging.py         Structured logging
    memory.py          GPU VRAM diagnostics
```

---

## Configuration

All hyperparameters live in YAML files. To retarget the entire pipeline at a **different person**, change two things:

**`config/cpt.yaml`**
```yaml
model_id: Qwen/Qwen2.5-3B         # your base model
corpus_dir: /data/nietzsche_txt    # directory of .txt files
```

**`config/sft.yaml`**
```yaml
model_id: my/cpt-finished-model    # your CPT model (HF or local)
dataset_path: /data/nietzsche_sft.jsonl
default_system_prompt: "You are Nietzsche."
```

The pipeline is corpus-agnostic. That's all you change.

---

## Hardware Notes

| GPU          | VRAM   | CPT        | SFT        | Notes                         |
|--------------|--------|------------|------------|-------------------------------|
| 1x T4        | 16 GB  | Single GPU | Yes        | SFT fits comfortably          |
| 2x T4        | 32 GB  | DDP (2x)   | Yes        | Optimal for Kaggle            |
| 1x A10       | 24 GB  | Single GPU | Yes        | Faster than T4                |
| CPU          | -      | No         | No         | Quantization step only        |

4-bit NF4 keeps the 3B model footprint at ~1.87 GB.

---

## Customising for a Different Figure

1. **Collect texts** - Download works from Project Gutenberg as `.txt` files. Place them in `corpus_dir/`.
2. **Create SFT data** - Build a JSONL where each line has `{"messages": [{"role": "system", "content": "You are X."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
3. **Update config/cpt.yaml** - set `corpus_dir`.
4. **Update config/sft.yaml** - set `dataset_path` and `default_system_prompt`.
5. **Run** - `celestia cpt` then `celestia sft`.

The pipeline automatically handles Gutenberg boilerplate removal, text cleaning, sequence packing, chat-template tokenisation, and response masking.

---

## Origin

Originally built to turn Qwen2.5-3B into **Plato**. The name "Celestia" comes from that project. The pipeline was generalised so anyone can point it at any corpus.