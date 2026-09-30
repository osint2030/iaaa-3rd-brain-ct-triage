"""MIL (Multiple Instance Learning) model for the Brain CT Triage Challenge.

v3 architecture (builds on v2's Dropout + frozen-early-backbone regularization):
    per-slice CNN backbone (ResNet18, early layers frozen) -> Gated Attention
    pooling over slices (mask-aware, ignoring padded slices) -> two heads:
        - regression head: V_EDH, V_SDH, V_IPH, V_SAH, V_IVH, MLS_mm (Softplus, >= 0)
        - fracture head: raw logit for fracture_prob (apply sigmoid separately)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

TARGET_COLUMNS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "fracture_prob", "MLS_mm"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GatedAttentionMILPooling(nn.Module):
    """Gated attention pooling (Ilse et al., 2018 -- 'Attention-based Deep MIL').

    Standard attention: a = softmax(w^T tanh(V h))
    Gated attention:     a = softmax(w^T (tanh(V h) * sigmoid(U h)))

    The extra sigmoid gate lets the network learn to suppress irrelevant slices
    more sharply than plain tanh attention alone -- tanh can only shape the
    attention score smoothly, while the multiplicative sigmoid gate can drive
    the score much closer to zero for slices the network learns are irrelevant
    (e.g. slices with no hemorrhage), a documented improvement over vanilla
    attention MIL pooling.

    Padded slices (mask == 0) receive an attention weight of exactly 0 by
    setting their pre-softmax logit to -inf.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout_p: float = 0.3):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())
        self.attention_dropout = nn.Dropout(dropout_p)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            features: (B, T, D) per-slice feature vectors.
            mask: (B, T) with 1.0 for real slices, 0.0 for padding.

        Returns:
            pooled: (B, D) series-level embedding.
            attn_weights: (B, T) attention weight per slice (sums to 1 over real slices).
        """
        gate_v = self.attention_v(features)  # (B, T, hidden_dim)
        gate_u = self.attention_u(features)  # (B, T, hidden_dim)
        gated = self.attention_dropout(gate_v * gate_u)

        attn_logits = self.attention_w(gated).squeeze(-1)  # (B, T)
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(features * attn_weights.unsqueeze(-1), dim=1)
        return pooled, attn_weights


# Backbone submodules considered "early" (generic, low-level features) and
# frozen by default -- only layer3, layer4, and the heads are fine-tuned.
FROZEN_BACKBONE_SUBMODULES = ["conv1", "bn1", "layer1", "layer2"]


class BrainCTMILModel(nn.Module):
    """ResNet18 backbone + Gated Attention MIL pooling + regression/classification heads."""

    def __init__(
        self,
        pretrained: bool = True,
        attention_hidden: int = 128,
        dropout_p: float = 0.3,
        freeze_early_backbone: bool = True,
    ):
        super().__init__()

        backbone = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()  # use as a feature extractor
        self.backbone = backbone

        if freeze_early_backbone:
            for name in FROZEN_BACKBONE_SUBMODULES:
                for param in getattr(self.backbone, name).parameters():
                    param.requires_grad = False

        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.pooling = GatedAttentionMILPooling(self.feature_dim, attention_hidden, dropout_p)

        # 6 non-negative regression targets: V_EDH, V_SDH, V_IPH, V_SAH, V_IVH, MLS_mm
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, 6),
        )
        self.softplus = nn.Softplus()

        # 1 binary target: fracture_prob (raw logit; apply sigmoid outside, e.g. in the loss)
        self.fracture_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        Args:
            images: (B, T, 3, H, W) padded slice stack, values in [0, 1].
            mask: (B, T) with 1.0 for real slices, 0.0 for padding.

        Returns:
            Dict with per-target predictions and the attention weights (for inspection).
        """
        B, T, C, H, W = images.shape
        flat_images = images.view(B * T, C, H, W)
        flat_images = (flat_images - self.imagenet_mean) / self.imagenet_std

        flat_features = self.backbone(flat_images)          # (B*T, feature_dim)
        features = flat_features.view(B, T, self.feature_dim)

        pooled, attn_weights = self.pooling(features, mask)  # (B, feature_dim), (B, T)

        reg_out = self.softplus(self.regression_head(pooled))  # (B, 6), >= 0
        fracture_logit = self.fracture_head(pooled).squeeze(-1)  # (B,)

        return {
            "V_EDH": reg_out[:, 0],
            "V_SDH": reg_out[:, 1],
            "V_IPH": reg_out[:, 2],
            "V_SAH": reg_out[:, 3],
            "V_IVH": reg_out[:, 4],
            "MLS_mm": reg_out[:, 5],
            "fracture_logit": fracture_logit,
            "attention_weights": attn_weights,
        }

    def to_intermediate_tensor(self, output: dict) -> torch.Tensor:
        """Assemble model output into a (B, 7) tensor matching TARGET_COLUMNS order,
        i.e. the same column order as the "targets" tensor from BrainCTSeriesDataset.
        Applies sigmoid to the fracture logit.
        """
        fracture_prob = torch.sigmoid(output["fracture_logit"])
        return torch.stack(
            [
                output["V_EDH"], output["V_SDH"], output["V_IPH"],
                output["V_SAH"], output["V_IVH"], fracture_prob, output["MLS_mm"],
            ],
            dim=1,
        )
