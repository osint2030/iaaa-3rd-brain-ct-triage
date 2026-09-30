"""Loss for IntermediatesModel. v2: MLS loss uses PER-SAMPLE weights derived
from inverse-frequency binning of MLS_mm (mirrors segmentation v2's class-
weighting fix), so rare large-MLS slices contribute proportionally more
gradient instead of being drowned out by the ~99% of slices near the 0.1mm
noise floor. Fracture loss unchanged (already well-calibrated via
pos_weight + bias-init).
"""

import torch
import torch.nn as nn

from src.intermediates_model import IntermediatesModel

# Bin edges chosen around the triage rule's own decision thresholds
# (EPS_MLS=1.0, COMBO_MLS=3.0, MLS_CRITICAL=5.0), so weighting reflects what
# actually matters for the final triage decision, not an arbitrary binning.
MLS_BIN_EDGES = [1.0, 3.0, 5.0, 10.0]  # interior edges (bucketize convention)
# sqrt-inverse-frequency weights, computed from the real per-bin slice counts:
# near_zero[0,1), small[1,3), medium[3,5), large[5,10), very_large[10,inf)
MLS_BIN_WEIGHTS = [0.497374, 1.734710, 2.345286, 2.068348, 2.183342]


class IntermediatesLoss(nn.Module):
    def __init__(self, fracture_pos_weight: float,
                 mls_bin_edges: list = MLS_BIN_EDGES,
                 mls_bin_weights: list = MLS_BIN_WEIGHTS,
                 mls_weight: float = 1.0, fracture_weight: float = 1.0,
                 huber_beta: float = 1.0):
        super().__init__()
        self.mls_weight = mls_weight
        self.fracture_weight = fracture_weight
        self.huber_beta = huber_beta
        self.register_buffer("mls_bin_edges", torch.tensor(mls_bin_edges))
        self.register_buffer("mls_bin_weights", torch.tensor(mls_bin_weights))
        self.fracture_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(fracture_pos_weight)
        )

    def _sample_weight_for_mls(self, mls_target: torch.Tensor) -> torch.Tensor:
        bin_idx = torch.bucketize(mls_target, self.mls_bin_edges)
        return self.mls_bin_weights[bin_idx]

    def forward(self, mls_logit, mls_target, fracture_logit, fracture_target) -> dict:
        mls_pred = IntermediatesModel.mls_from_logit(mls_logit)

        # Manual per-sample Huber (nn.SmoothL1Loss has no per-sample weight
        # argument), weighted then reduced by weighted mean.
        diff = mls_pred - mls_target
        abs_diff = diff.abs()
        quadratic = 0.5 * diff ** 2 / self.huber_beta
        linear = abs_diff - 0.5 * self.huber_beta
        per_sample_huber = torch.where(abs_diff < self.huber_beta, quadratic, linear)

        weights = self._sample_weight_for_mls(mls_target)
        mls_loss = (per_sample_huber * weights).sum() / weights.sum()

        fracture_loss = self.fracture_loss_fn(fracture_logit, fracture_target)
        total = self.mls_weight * mls_loss + self.fracture_weight * fracture_loss
        return {
            "total": total,
            "mls_loss": mls_loss.detach(),
            "fracture_loss": fracture_loss.detach(),
        }
