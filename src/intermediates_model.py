"""2.5D CNN + physical-embedding architecture predicting MLS_mm and
fracture_prob directly from imaging, at the per-slice level. Series-level
aggregation uses physically-motivated pooling: hard-max (or LogSumExp
fallback) for MLS, complement-product (soft-OR) for fracture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class PhysicalEmbedding(nn.Module):
    """Encodes per-slice geometry (slice thickness, relative z-position)
    into a small embedding, concatenated onto the CNN feature vector."""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, slice_thickness: torch.Tensor, z_rel: torch.Tensor) -> torch.Tensor:
        x = torch.stack([slice_thickness, z_rel], dim=1)
        return self.mlp(x)


class IntermediatesModel(nn.Module):
    """Per-slice CNN + physical embedding -> per-slice MLS/fracture logits,
    aggregated at series level via hard-max (MLS) and complement-product
    (fracture) pooling.

    IMPORTANT: MLS non-negativity is enforced with Softplus, NOT ReLU.
    ReLU's derivative is exactly zero for negative inputs, so if a slice's
    raw mls_logit starts (or drifts) negative, gradients stop flowing to
    mls_head entirely -- a "dead neuron" that never recovers. Softplus is
    smooth and has a nonzero gradient everywhere, so it can't get stuck.
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True,
                 phys_embed_dim: int = 32, fracture_prior_prob: float = 0.02):
        super().__init__()

        backbone = getattr(tv_models, backbone_name)(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        cnn_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.physical_embedding = PhysicalEmbedding(out_dim=phys_embed_dim)

        combined_dim = cnn_feature_dim + phys_embed_dim
        self.mls_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)
        )
        self.fracture_head = nn.Sequential(
            nn.Linear(combined_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)
        )
        self._init_fracture_head_bias(fracture_prior_prob)

    def _init_fracture_head_bias(self, prior_prob: float) -> None:
        final_layer = self.fracture_head[-1]
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(final_layer.bias, bias_value.item())

    def forward_per_slice(self, images: torch.Tensor, slice_thickness: torch.Tensor,
                           z_rel: torch.Tensor) -> dict:
        cnn_features = self.backbone(images)
        phys_features = self.physical_embedding(slice_thickness, z_rel)
        combined = torch.cat([cnn_features, phys_features], dim=1)

        mls_logit = self.mls_head(combined).squeeze(-1)
        fracture_logit = self.fracture_head(combined).squeeze(-1)
        return {"mls_logit": mls_logit, "fracture_logit": fracture_logit}

    @staticmethod
    def mls_from_logit(mls_logit: torch.Tensor) -> torch.Tensor:
        return F.softplus(mls_logit)

    @staticmethod
    def aggregate_mls(mls_per_slice_nonneg: torch.Tensor, tau: float = 2.0,
                       mode: str = "hard_max") -> torch.Tensor:
        if mode == "hard_max":
            return mls_per_slice_nonneg.max()
        elif mode == "logsumexp":
            return torch.logsumexp(tau * mls_per_slice_nonneg, dim=0) / tau
        raise ValueError(f"Unknown aggregation mode: {mode}")

    @staticmethod
    def aggregate_fracture(fracture_prob_per_slice: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.prod(1.0 - fracture_prob_per_slice)
