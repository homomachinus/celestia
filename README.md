# Celestia-Plato

Turn Qwen2.5-3B into Plato. A two-stage fine-tuning pipeline:

1. **CPT** (Continual Pre-Training) — Causal language modelling on Plato's raw corpus with QLoRA + optional DDP
2. **SFT** (Supervised Fine-Tuning) — Chat-style instruction tuning with response-only loss, GGUF export

## Pipeline

```
Raw .txt corpus (Plato's dialogues)
    │
    ▼
┌─────────────────────┐
│  CPT                 │  QLoRA (4-bit NF4) + LoRA rank 16
│  next-token predict  │  Base: Qwen/Qwen2.5-3B
└─────────┬───────────┘
          │ LoRA adapter
          ▼
┌─────────────────────┐
│  SFT                 │  Chat-template tokenisation
│  response-only loss  │  Response masking (labels = -100 for prompt)
└─────────┬───────────┘
          │ Merged weights
          ▼
┌─────────────────────┐
│  Export              │  HF Hub push + GGUF (f16 → Q4_K_M)
└─────────────────────┘
```

## Quickstart

```bash
# Install
pip install -e .

# 1. Continual Pre-Training
celestia cpt --config config/cpt.yaml

# 2. Supervised Fine-Tuning
celestia sft --config config/sft.yaml

# 3. Merge adapter into base
celestia merge --adapter /path/to/sft_adapter --output /path/to/merged

# 4. Convert to GGUF + quantize
celestia quantize --model /path/to/merged --outdir /path/to/gguf
```

## Project Structure

```
src/celestia/
├── cli.py                 # CLI entry point (argparse subcommands)
├── config.py              # Pydantic models + YAML loader
├── data/
│   ├── loader.py          # load_books, clean_text (Gutenberg removal)
│   ├── builder.py         # Dataset building, tokenise & pack
│   └── sft.py             # SFT data loader, chat-template, collator
├── model/
│   ├── loader.py          # load_model, load_tokenizer, bnb config
│   ├── lora.py            # apply_lora, prepare_for_kbit
│   └── merge.py           # merge_and_unload
├── training/
│   ├── cpt.py             # CPT trainer (single-GPU + DDP)
│   └── sft.py             # SFT trainer
├── export/
│   ├── gguf.py            # llama.cpp conversion + quantization
│   └── hub.py             # push to HuggingFace Hub
└── utils/
    ├── logging.py         # structured logging setup
    └── memory.py          # GPU VRAM diagnostics
```

## Configuration

All hyperparameters live in YAML files:

```yaml
# config/cpt.yaml
model_id: Qwen/Qwen2.5-3B
corpus_dir: /path/to/plato/txt
output_dir: /output/Plato-CPT
max_seq_len: 2048
batch_size: 2
grad_accum: 4   # 2 GPUs × 2 batch × 4 = 16 effective
learning_rate: 2e-4
num_epochs: 3
lora_r: 16
lora_alpha: 32
use_ddp: true
num_gpus: 2
```

## Hardware Notes

- **CPT** was designed for Kaggle 2×T4 (16 GB each). DDP gives ~2× throughput vs single GPU.
- **SFT** runs on a single T4/Colab GPU. Low-rank LoRA keeps memory under 12 GB.
- 4-bit NF4 quantization keeps the model footprint at ~1.87 GB.
