"""Checkpoint save/load utilities for resumable multi-fold training.

Each checkpoint stores everything needed to resume exactly where training
left off: which fold and epoch it corresponds to, model + optimizer state,
the best validation QWK seen so far in this fold, and the full per-epoch
history (for logging/plotting after the fact).
"""

import os

import torch


def save_checkpoint(path: str, fold: int, epoch: int, model, optimizer, best_val_qwk: float, history: list) -> None:
    """Save a checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "fold": fold,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_qwk": best_val_qwk,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer, device: str):
    """Load a checkpoint from disk, restoring model and optimizer state in-place.

    Returns:
        (fold, epoch, best_val_qwk, history) from the checkpoint.
    """
    # weights_only=False: our checkpoints intentionally store non-tensor Python/numpy
    # objects (e.g. numpy.float64 values in `history` from sklearn's QWK). This is
    # safe because we only ever load checkpoints we wrote ourselves.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        checkpoint["fold"],
        checkpoint["epoch"],
        checkpoint["best_val_qwk"],
        checkpoint["history"],
    )
