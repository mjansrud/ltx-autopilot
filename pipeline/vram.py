"""
GPU memory management — load/unload models cleanly between pipeline stages.
"""

import gc
import logging
from contextlib import contextmanager

import torch

log = logging.getLogger(__name__)


def get_vram_usage() -> dict:
    """Return current VRAM usage in MB for all visible GPUs."""
    if not torch.cuda.is_available():
        return {}
    info = {}
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        total = torch.cuda.get_device_properties(i).total_memory / 1024**2
        info[i] = {"allocated_mb": alloc, "reserved_mb": reserved, "total_mb": total}
    return info


def log_vram(label: str):
    info = get_vram_usage()
    for gpu_id, usage in info.items():
        log.info(
            "[VRAM] %s | GPU %d: %.0f MB allocated / %.0f MB reserved / %.0f MB total",
            label, gpu_id, usage["allocated_mb"], usage["reserved_mb"], usage["total_mb"],
        )


def flush_vram():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()


def unload_model(*models):
    """Delete model references and flush VRAM."""
    for model in models:
        if model is not None:
            del model
    flush_vram()


@contextmanager
def vram_stage(name: str, log_usage: bool = True):
    """Context manager that logs VRAM and flushes on exit."""
    if log_usage:
        log_vram(f"{name} — start")
    try:
        yield
    finally:
        flush_vram()
        if log_usage:
            log_vram(f"{name} — end (after flush)")
