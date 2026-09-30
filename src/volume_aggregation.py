"""Aggregate per-slice segmentation predictions into series-level hemorrhage
volumes, using the SAME formula the ground-truth *_Area columns were derived
from (validated earlier to ~1% agreement):

    Volume_mL = sum_over_slices(pixel_count_for_class * PixelSpacing0 * PixelSpacing1 * SliceThickness) / 1000
"""

import numpy as np
import torch

from src.rle import CLASS_TO_AREA_COLUMN  # {1: "..._Area", 2: ..., ...}

# Map RLE class value -> official intermediate volume key.
CLASS_VALUE_TO_TARGET_KEY = {
    1: "V_IVH",
    2: "V_IPH",
    3: "V_SDH",
    4: "V_EDH",
    5: "V_SAH",
}


@torch.no_grad()
def predict_series_volumes(net, series_slices: list, device: str, image_size: int = 224) -> dict:
    """Run the segmentation model on every slice of one series and aggregate
    predicted pixel counts into volumes (mL) for each hemorrhage subtype.

    Args:
        net: trained HemorrhageSegModel, in eval() mode.
        series_slices: list of dicts, each with keys "image" (preprocessed
            (3, image_size, image_size) tensor), "pixel_spacing_0",
            "pixel_spacing_1", "slice_thickness", "original_height",
            "original_width" (the DICOM's native Rows/Columns BEFORE resizing
            to image_size -- required to compute the correct effective pixel
            spacing on the resized prediction; using the raw DICOM spacing
            directly on a resized mask silently over/under-counts volume).
        device: "cuda" or "cpu".

    Returns:
        Dict with keys "V_IVH", "V_IPH", "V_SDH", "V_EDH", "V_SAH" (mL).
    """
    volumes = {key: 0.0 for key in CLASS_VALUE_TO_TARGET_KEY.values()}

    for slice_data in series_slices:
        image = slice_data["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
        logits = net(image)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (image_size, image_size)

        # Rescale spacing to account for resizing: each predicted-mask pixel
        # now covers (original_size / image_size) as many physical mm as a
        # native-resolution pixel did.
        h_scale = slice_data["original_height"] / image_size
        w_scale = slice_data["original_width"] / image_size
        effective_spacing_0 = slice_data["pixel_spacing_0"] * h_scale
        effective_spacing_1 = slice_data["pixel_spacing_1"] * w_scale
        slice_thickness = slice_data["slice_thickness"]

        for class_value, target_key in CLASS_VALUE_TO_TARGET_KEY.items():
            pixel_count = int((pred_mask == class_value).sum())
            volumes[target_key] += (
                pixel_count * effective_spacing_0 * effective_spacing_1 * slice_thickness / 1000.0
            )

    return volumes
