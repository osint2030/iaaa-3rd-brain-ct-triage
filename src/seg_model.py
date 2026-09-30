"""U-Net segmentation model for hemorrhage-subtype segmentation.

Uses segmentation_models_pytorch (a well-tested, actively-maintained library)
rather than a from-scratch U-Net, for reliability and access to ImageNet-
pretrained encoders -- the same transfer-learning rationale used for the
MIL model's ResNet18 backbone.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.rle import N_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HemorrhageSegModel(nn.Module):
    """Thin wrapper around smp.Unet: adds ImageNet normalization (since our
    windowed multi-window images are in [0, 1], not ImageNet-normalized) and
    a consistent forward() interface.
    """

    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=N_CLASSES,
        )
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W), values in [0, 1].

        Returns:
            logits: (B, N_CLASSES, H, W)
        """
        normalized = (images - self.imagenet_mean) / self.imagenet_std
        return self.unet(normalized)
