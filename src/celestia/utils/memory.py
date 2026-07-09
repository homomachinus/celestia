"""GPU / system utilities."""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def print_gpu_memory() -> None:
    """Log current GPU VRAM usage: allocated, reserved, peak."""
    if not torch.cuda.is_available():
        log.info("CUDA is not available.")
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_alloc = torch.cuda.max_memory_allocated(i) / 1024**3
        log.info(
            "GPU %d VRAM  allocated: %.2f GB  reserved: %.2f GB  peak: %.2f GB",
            i, allocated, reserved, max_alloc,
        )
