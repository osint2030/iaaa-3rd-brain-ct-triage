"""Loss function for the Brain CT Triage Challenge MIL model.

Combines:
    - Smooth L1 (Huber) regression loss on log1p-transformed hemorrhage volumes,
      to prevent large volumes from dominating the gradient over small,
      clinically important ones near the triage decision thresholds.
    - Smooth L1 on MLS_mm directly (smaller dynamic range, no transform needed).
    - BCEWithLogitsLoss on the raw fracture logit vs. the binary fracture label.
"""

import torch
import torch.nn as nn

VOLUME_KEYS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH"]
MLS_KEY = "MLS_mm"
TARGET_COLUMNS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "fracture_prob", "MLS_mm"]


class BrainCTLoss(nn.Module):
    """Weighted combination of volume regression, MLS regression, and fracture classification losses."""

    def __init__(self, volume_weight: float = 1.0, mls_weight: float = 1.0, fracture_weight: float = 1.0):
        super().__init__()
        self.volume_weight = volume_weight
        self.mls_weight = mls_weight
        self.fracture_weight = fracture_weight
        self.regression_loss = nn.SmoothL1Loss(reduction="mean")
        self.fracture_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, output: dict, targets: torch.Tensor) -> dict:
        """
        Args:
            output: dict from BrainCTMILModel.forward(...) -- raw per-target predictions
                plus "fracture_logit".
            targets: (B, 7) ground-truth tensor, columns in TARGET_COLUMNS order
                (this is exactly what BrainCTSeriesDataset.__getitem__ returns as "targets").

        Returns:
            Dict with "total" (use this for backward()) and the three unweighted
            sub-losses (detached, for logging/diagnostics).
        """
        target_dict = {name: targets[:, i] for i, name in enumerate(TARGET_COLUMNS)}

        volume_losses = []
        for key in VOLUME_KEYS:
            pred_log = torch.log1p(output[key])
            true_log = torch.log1p(target_dict[key].clamp(min=0.0))
            volume_losses.append(self.regression_loss(pred_log, true_log))
        volume_loss = torch.stack(volume_losses).mean()

        mls_loss = self.regression_loss(output[MLS_KEY], target_dict[MLS_KEY].clamp(min=0.0))

        fracture_loss = self.fracture_loss(output["fracture_logit"], target_dict["fracture_prob"])

        total = (
            self.volume_weight * volume_loss
            + self.mls_weight * mls_loss
            + self.fracture_weight * fracture_loss
        )

        return {
            "total": total,
            "volume_loss": volume_loss.detach(),
            "mls_loss": mls_loss.detach(),
            "fracture_loss": fracture_loss.detach(),
        }
