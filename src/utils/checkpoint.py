"""Model checkpoint save & load helpers.
Saves can be either a bare ``state_dict`` (lightweight) or a full
dictionary with optimiser and scheduler state for resuming training.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
from src.utils.logging import get_logger
logger = get_logger(__name__)
def save_checkpoint(
    state: dict | nn.Module,
    save_dir: str,
    filename: str = "best.ckpt",
) -> str:
    """Save *state* (state_dict or full dict) to ``save_dir / filename``.
    Returns the full path of the saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    if isinstance(state, nn.Module):
        torch.save(state.state_dict(), ckpt_path)
    else:
        torch.save(state, ckpt_path)
    logger.info("Checkpoint saved to %s", ckpt_path)
    return ckpt_path
def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    """Load a checkpoint into *model* and optionally into *optimizer* / *scheduler*.
    Supports both:
    - bare ``state_dict`` (saved by ``torch.save(model.state_dict())``)
    - full dict with ``"model_state_dict"`` key
    Returns the raw checkpoint dict (or empty dict if bare state_dict).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        model.load_state_dict(raw["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in raw:
            optimizer.load_state_dict(raw["optimizer_state_dict"])
            logger.info("Optimizer state restored.")
        if scheduler is not None and "scheduler_state_dict" in raw:
            scheduler.load_state_dict(raw["scheduler_state_dict"])
            logger.info("Scheduler state restored.")
        logger.info("Checkpoint loaded (full dict) from %s", ckpt_path)
        return raw
    else:
        # bare state_dict
        model.load_state_dict(raw, strict=strict)
        logger.info("Checkpoint loaded (state_dict) from %s", ckpt_path)
        return {}
