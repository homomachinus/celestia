"""Text loading and cleaning for Project Gutenberg corpora.

Handles:
  - Recursive .txt discovery
  - UTF-8 detection with graceful skipping of binary files
  - Gutenberg licence header/footer removal
  - Normalisation of unicode quotes, dashes, spacing
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def load_books(corpus_dir: str) -> dict[str, str]:
    """Scan *corpus_dir* recursively for every ``.txt`` file.

    Returns ``{filename_stem: raw_text}``.
    Raises ``FileNotFoundError`` if no text files are found.
    """
    pattern = os.path.join(corpus_dir, "**", "*.txt")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise FileNotFoundError(
            f"No .txt files found under {corpus_dir}. "
            "Check that the corpus directory is correct."
        )

    books: dict[str, str] = {}
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
            name = os.path.splitext(os.path.basename(path))[0]
            books[name] = text
            log.info("  Loaded %-30s %8s chars", name, f"{len(text):,}")
        except (UnicodeDecodeError, OSError) as exc:
            log.warning("  Skipped %s -- %s", path, exc)

    return books


# ?? Gutenberg regexes compiled once ??????????

_START_PATTERN = re.compile(
    r"\*\*\*\s*START\s+(OF\s+)?(THE\s+)?PROJECT\s+GUTENBERG",
    re.IGNORECASE,
)
_END_PATTERN = re.compile(
    r"\*\*\*\s*END\s+(OF\s+)?(THE\s+)?PROJECT\s+GUTENBERG",
    re.IGNORECASE,
)
_TOC_PATTERN = re.compile(r"^[IVXLCDM]+\.?\s*$")
_PAGE_NUM = re.compile(r"^\s*-?\s*\d+\s*-?\s*$")
_PAGE_BRACE = re.compile(r"^\s*\{\s*\d+\s*\}\s*$")
_SPECIAL_LINE = re.compile(r"^[\s\*_=#$%&~+]+$")


def clean_text(text: str, book_name: str = "") -> str:
    """Strip Gutenberg boilerplate, TOC, page numbers, OCR artefacts.

    Preserves dialogue formatting, speaker names, paragraph breaks,
    and all narrative prose.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    in_header = True
    in_licence = False

    # Unicode normalisation map
    _trans = str.maketrans({
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
    })

    for raw_line in lines:
        line = raw_line.strip()

        # --- Gutenberg header ---------------------------------
        if in_header:
            if _START_PATTERN.search(line):
                in_header = False
                continue
            if _END_PATTERN.search(line):
                in_licence = True
                continue
            continue

        # --- Gutenberg footer / licence -----------------------
        if _START_PATTERN.search(line):
            continue  # stray start marker in body -> skip
        if _END_PATTERN.search(line):
            in_licence = True
            continue
        if in_licence:
            continue

        # --- Table of contents --------------------------------
        if _TOC_PATTERN.match(line):
            continue

        # --- Page numbers -------------------------------------
        if _PAGE_NUM.match(line):
            continue
        if _PAGE_BRACE.match(line):
            continue

        # --- Normalise Unicode --------------------------------
        text_line = raw_line.translate(_trans)

        # Normalise multiple hyphens to em-dash
        text_line = re.sub(r"---?", "--", text_line)

        # --- Skip lines that are only decoration -------------
        if _SPECIAL_LINE.match(text_line.strip()):
            continue

        cleaned.append(text_line)

    # Post-process: collapse excessive blank lines
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()

    if book_name:
        orig = len(text.splitlines())
        final = len(result.splitlines())
        log.info("  Cleaned %s: %d -> %d lines (%s chars)",
                  book_name, orig, final, f"{len(result):,}")

    return result


# ?? GPU diagnostic (moved from utils for convenience) ??

def print_gpu_memory() -> None:
    """Log current GPU VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        log.info("GPU VRAM  allocated: %.2f GB  reserved: %.2f GB  peak: %.2f GB",
                 allocated, reserved, max_alloc)
    else:
        log.info("CUDA not available.")
