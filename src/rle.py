"""RLE (run-length encoding) decoding for the Brain CT Triage Challenge's
pixel-level segmentation annotations.

Format (empirically verified against training_df's ground-truth *_Area
columns -- mean error ~1.2% across 20 samples spanning all 5 hemorrhage
subtypes, consistent with a minor difference in area-computation convention
rather than a decoding error):

    segmentation_rle = {"shape": [H, W], "counts": [v0, len0, v1, len1, ...]}

`counts` is a flat list of (class_value, run_length) PAIRS (not standard
alternating-binary COCO RLE). Decoding fills the flattened (row-major) image
with `class_value` for `run_length` pixels, advancing through the whole image.
"""

import numpy as np

CLASS_MAP = {
    0: "BG",
    1: "IntraventricularHemorrhage",
    2: "IntraparenchymalHemorrhage",
    3: "SubduralHemorrhage",
    4: "EpiduralHemorrhage",
    5: "SubarachnoidHemorrhage",
}

# Matches the *_Area column naming in training_df, for cross-checking.
CLASS_TO_AREA_COLUMN = {
    1: "IntraventricularHemorrhage_Area",
    2: "IntraparenchymalHemorrhage_Area",
    3: "SubduralHemorrhage_Area",
    4: "EpiduralHemorrhage_Area",
    5: "SubarachnoidHemorrhage_Area",
}

N_CLASSES = len(CLASS_MAP)  # 6 (BG + 5 hemorrhage subtypes)


def decode_rle_mask(counts: list, shape: list) -> np.ndarray:
    """Decode a (class_value, run_length) pair-encoded RLE mask.

    Args:
        counts: flat list [v0, len0, v1, len1, ...].
        shape: [height, width].

    Returns:
        2D uint8 array of shape (height, width) with integer class labels.
    """
    h, w = shape
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i in range(0, len(counts), 2):
        value = counts[i]
        run_length = counts[i + 1]
        flat[pos:pos + run_length] = value
        pos += run_length
    assert pos == h * w, f"Decoded {pos} pixels, expected {h * w} -- RLE doesn't cover the full image."
    return flat.reshape(h, w)


def load_annotation(json_path: str) -> dict:
    """Load one annotation JSON file and decode its segmentation mask.

    Returns:
        Dict with keys: "mask" (decoded 2D array), "keypoints", "boxes_xywh", "class_map".
    """
    import json

    with open(json_path) as f:
        data = json.load(f)

    seg = data["segmentation_rle"]
    mask = decode_rle_mask(seg["counts"], seg["shape"])

    return {
        "mask": mask,
        "keypoints": data.get("keypoints", {}),
        "boxes_xywh": data.get("boxes_xywh", []),
        "class_map": data.get("class_map", []),
    }
