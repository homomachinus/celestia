"""Celestia-Plato CLI.

Two-stage fine-tuning pipeline that turns any base LLM into any historical
figure by training on their original writings.

Usage::

    celestia cpt --config config/cpt.yaml
    celestia sft --config config/sft.yaml
    celestia merge --adapter PATH --base MODEL_ID --output DIR
    celestia quantize --model PATH --outdir DIR
    celestia hub --model PATH --repo USER/REPO
    celestia generate --model PATH --prompt "..."
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch

from celestia import __version__
from celestia.config import load_cpt_config, load_sft_config
from celestia.export.gguf import convert_to_gguf, quantize_gguf
from celestia.export.hub import hf_login, push_to_hub
from celestia.model.loader import load_model, load_tokenizer
from celestia.model.merge import merge_adapter
from celestia.training.cpt import run_cpt
from celestia.training.sft import run_sft
from celestia.utils.logging import setup_logging

log = logging.getLogger(__name__)


# ??????????????????????????????????????????????
#  Subcommand handlers
# ??????????????????????????????????????????????

def _cmd_cpt(args: argparse.Namespace) -> None:
    cfg = load_cpt_config(args.config)
    log.info("CPT config loaded from %s", args.config)
    log.info("Effective batch size: %d  (%d GPU%s x %d batch x %d accum)",
             cfg.effective_batch_size,
             cfg.num_gpus if cfg.use_ddp else 1,
             "s" if cfg.num_gpus > 1 else "",
             cfg.batch_size,
             cfg.grad_accum)
    run_cpt(cfg)


def _cmd_sft(args: argparse.Namespace) -> None:
    cfg = load_sft_config(args.config)
    log.info("SFT config loaded from %s", args.config)
    run_sft(cfg)


def _cmd_merge(args: argparse.Namespace) -> None:
    merge_adapter(
        base_model_id=args.base,
        adapter_path=args.adapter,
        output_dir=args.output,
        torch_dtype=torch.float16,
        device_map="cpu",
    )


def _cmd_quantize(args: argparse.Namespace) -> None:
    model_dir = args.model
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Step 1: HF -> f16 GGUF
    f16_path = os.path.join(outdir, f"{os.path.basename(model_dir)}-f16.gguf")
    convert_to_gguf(model_dir, f16_path, outtype="f16")

    # Step 2: f16 -> Q4_K_M
    q4_path = f16_path.replace("f16.gguf", f"{args.quant_type}.gguf")
    quantize_gguf(f16_path, q4_path, quant_type=args.quant_type)

    log.info("GGUF files ready:")
    log.info("  %s  (f16)", f16_path)
    log.info("  %s  (%s)", q4_path, args.quant_type)


def _cmd_hub(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        hf_login(token)
    else:
        hf_login()
    push_to_hub(
        model_dir=args.model,
        repo_id=args.repo,
        private=not args.public,
    )


def _cmd_generate(args: argparse.Namespace) -> None:
    """Quick inference test with a loaded model."""
    tokenizer = load_tokenizer(args.model)
    model = load_model(
        args.model,
        quant_cfg=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    messages = [{"role": "user", "content": args.prompt or "hello"}]
    if args.system:
        messages.insert(0, {"role": "system", "content": args.system})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    print()
    print("-" * 55)
    print(response)
    print("-" * 55)


# ??????????????????????????????????????????????
#  Argument parser
# ??????????????????????????????????????????????

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="celestia",
        description=f"Celestia v{__version__} -- Turn any LLM into any historical figure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  celestia cpt --config config/cpt.yaml
  celestia sft --config config/sft.yaml
  celestia merge --adapter output/sft_adapter --base Qwen/Qwen2.5-3B --output merged
  celestia quantize --model merged --outdir gguf
  celestia hub --model merged --repo myuser/my-model
  celestia generate --model merged --prompt "What is justice?" --system "You are Plato"
""",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- cpt --
    p_cpt = sub.add_parser("cpt", help="Run Continual Pre-Training")
    p_cpt.add_argument("--config", required=True, help="Path to CPT YAML config")
    p_cpt.set_defaults(func=_cmd_cpt)

    # -- sft --
    p_sft = sub.add_parser("sft", help="Run Supervised Fine-Tuning")
    p_sft.add_argument("--config", required=True, help="Path to SFT YAML config")
    p_sft.set_defaults(func=_cmd_sft)

    # -- merge --
    p_merge = sub.add_parser("merge", help="Merge LoRA adapter into base model")
    p_merge.add_argument("--adapter", required=True, help="Path to saved adapter dir")
    p_merge.add_argument("--base", required=True, help="Base model ID")
    p_merge.add_argument("--output", required=True, help="Output directory")
    p_merge.set_defaults(func=_cmd_merge)

    # -- quantize --
    p_q = sub.add_parser("quantize", help="Convert HF model to GGUF + quantize")
    p_q.add_argument("--model", required=True, help="Merged HF model directory")
    p_q.add_argument("--outdir", required=True, help="Output directory for .gguf files")
    p_q.add_argument("--quant-type", default="Q4_K_M",
                     choices=["Q4_K_M", "Q5_K_M", "Q8_0", "f16"],
                     help="Quantization format (default: Q4_K_M)")
    p_q.set_defaults(func=_cmd_quantize)

    # -- hub --
    p_hub = sub.add_parser("hub", help="Push model to HuggingFace Hub")
    p_hub.add_argument("--model", required=True, help="Local model directory")
    p_hub.add_argument("--repo", required=True, help="HF repo ID (e.g. user/model)")
    p_hub.add_argument("--token", default=None, help="HF token (or $HF_TOKEN)")
    p_hub.add_argument("--public", action="store_true", help="Make repo public")
    p_hub.set_defaults(func=_cmd_hub)

    # -- generate --
    p_gen = sub.add_parser("generate", help="Quick inference sanity check")
    p_gen.add_argument("--model", required=True, help="Model ID or local path")
    p_gen.add_argument("--prompt", default="hello", help="User prompt")
    p_gen.add_argument("--system", default=None, help="System prompt")
    p_gen.add_argument("--max-new-tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-p", type=float, default=0.9)
    p_gen.set_defaults(func=_cmd_generate)

    return parser


# ??????????????????????????????????????????????
#  Entry point
# ??????????????????????????????????????????????

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    log.info("Celestia v%s", __version__)

    if torch.cuda.is_available():
        log.info("CUDA available: %d device(s)", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            log.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))
    else:
        log.warning("CUDA not available -- training will be slow or impossible")

    args.func(args)


if __name__ == "__main__":
    main()
