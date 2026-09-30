"""Loss function for hemorrhage-subtype segmentation.

Combines Dice loss (handles severe class imbalance -- background is >95% of
pixels -- by measuring region overlap regardless of pixel count) with
CrossEntropyLoss (provides a stronger, better-calibrated per-pixel gradient
signal). This Dice+CE combination is a standard, well-validated choice in
medical image segmentation literature.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target_mask: torch.Tensor) -> dict:
        """
        Args:
            logits: (B, N_CLASSES, H, W) raw model output.
            target_mask: (B, H, W) integer class labels (0..N_CLASSES-1).

        Returns:
            Dict with "total", "dice", "ce".
        """
        dice = self.dice_loss(logits, target_mask)
        ce = self.ce_loss(logits, target_mask)
        total = self.dice_weight * dice + self.ce_weight * ce
        return {"total": total, "dice": dice.detach(), "ce": ce.detach()}
