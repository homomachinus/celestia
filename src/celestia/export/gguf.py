"""GGUF conversion and quantization via llama.cpp.

Two steps:
  1. ``convert_hf_to_gguf(merged_dir, output_path, outtype="f16")``
  2. ``quantize_gguf(f16_path, q4_path, quant_type="Q4_K_M")``
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

log = logging.getLogger(__name__)

# Path where llama.cpp is cloned / expected
LLAMA_CPP_DIR = "/content/llama.cpp"


def _ensure_llama_cpp():
    """Clone llama.cpp if not already present."""
    if os.path.isdir(LLAMA_CPP_DIR):
        return
    log.info("Cloning llama.cpp …")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-r",
        os.path.join(LLAMA_CPP_DIR, "requirements.txt"),
    ])


def convert_to_gguf(
    model_dir: str,
    output_path: str,
    outtype: str = "f16",
) -> str:
    """Convert a HuggingFace model directory to a GGUF file.

    Uses llama.cpp's ``convert_hf_to_gguf.py``.

    Parameters
    ----------
    model_dir:
        Directory containing ``config.json``, ``model.safetensors``, etc.
    output_path:
        Where to write the ``.gguf`` file.
    outtype:
        ``"f16"``, ``"q8_0"``, etc.  Default ``"f16"`` (full-precision
        intermediate before quantization).

    Returns
    -------
    The *output_path* that was written.
    """
    _ensure_llama_cpp()
    script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")

    log.info("Converting %s → %s (type=%s)", model_dir, output_path, outtype)
    subprocess.check_call([
        sys.executable, script,
        model_dir,
        "--outfile", output_path,
        "--outtype", outtype,
    ])
    log.info("GGUF conversion complete: %s", output_path)
    return output_path


def quantize_gguf(
    input_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
) -> str:
    """Quantize an f16 GGUF file to a smaller type (e.g. Q4_K_M).

    Two strategies are tried in order:
      1. ``llama-quantize`` (from compiled llama.cpp) — faster.
      2. ``llama_cpp.llama_model_quantize`` (Python binding) — fallback.

    Parameters
    ----------
    input_path:
        Path to the f16 GGUF file.
    output_path:
        Where to write the quantized GGUF.
    quant_type:
        Quantization format (``"Q4_K_M"``, ``"Q5_K_M"``, …).

    Returns
    -------
    The *output_path* that was written.
    """
    # Strategy 1: compiled llama-quantize
    quantize_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")
    if os.path.isfile(quantize_bin):
        log.info("Quantizing via llama-quantize: %s → %s", input_path, output_path)
        subprocess.check_call([quantize_bin, input_path, output_path, quant_type])
        log.info("Quantization complete: %s", output_path)
        return output_path

    # Strategy 2: Python binding fallback
    log.info("llama-quantize not found; trying llama-cpp-python binding …")
    try:
        from llama_cpp import (  # type: ignore[import-untyped]
            LLAMA_FTYPE_MOSTLY_Q4_K_M,
            llama_model_quantize,
            llama_model_quantize_params,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Cannot quantize: neither compiled llama-quantize nor "
            "llama-cpp-python is available. "
            "Install with: pip install 'celestia-plato[gguf]'"
        ) from exc

    ftype_map = {
        "Q4_K_M": LLAMA_FTYPE_MOSTLY_Q4_K_M,
    }
    ftype = ftype_map.get(quant_type, LLAMA_FTYPE_MOSTLY_Q4_K_M)

    params = llama_model_quantize_params()
    params.ftype = ftype

    result = llama_model_quantize(
        input_path.encode("utf-8"),
        output_path.encode("utf-8"),
        params,
    )
    if result != 0:
        raise RuntimeError(f"Quantization failed with code {result}")

    log.info("Quantization complete: %s", output_path)
    return output_path
