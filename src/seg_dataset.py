"""Slice-level Dataset for hemorrhage-subtype segmentation.

Unlike BrainCTSeriesDataset (series-level MIL bags), this operates at the
SLICE level: each item is one CT slice + its pixel-level, 6-class mask
(BG, IVH, IPH, SDH, EDH, SAH). This uses the full annotation density
available (thousands of annotated slices) rather than the 338 series-level
aggregate labels.
"""

import json
import os

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

from src.preprocessing import WINDOW_PRESETS, apply_window, to_hu
from src.rle import N_CLASSES, decode_rle_mask


class SegmentationSliceDataset(Dataset):
    """Loads one CT slice as a multi-window image + its decoded segmentation mask.

    Args:
        slice_df: DataFrame with one row per slice, must contain columns
            "dicom_series.id", "dicom_series.SOPInstanceUID", "RelativeAnnotationPath".
        dicom_root: Directory containing one subfolder per series_id with .dcm files,
            each named "{SOPInstanceUID}.dcm".
        annotations_root: Directory containing annotation JSONs, referenced by
            slice_df's "RelativeAnnotationPath" column.
        image_size: Output height/width after resizing.
    """

    def __init__(self, slice_df, dicom_root: str, annotations_root: str, image_size: int = 224):
        self.slice_df = slice_df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.annotations_root = annotations_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.slice_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.slice_df.iloc[idx]
        series_id = row["dicom_series.id"]
        sop_uid = row["dicom_series.SOPInstanceUID"]

        # --- Load and window the image ---
        dcm_path = os.path.join(self.dicom_root, str(series_id), f"{sop_uid}.dcm")
        dcm = pydicom.dcmread(dcm_path)
        hu_image = to_hu(dcm)
        channels = [
            apply_window(hu_image, center, width)
            for center, width in WINDOW_PRESETS.values()
        ]
        image = np.stack(channels, axis=0).astype(np.float32)  # (3, H, W)

        # --- Load and decode the mask ---
        json_path = os.path.join(self.annotations_root, row["RelativeAnnotationPath"])
        with open(json_path) as f:
            ann = json.load(f)
        seg = ann["segmentation_rle"]
        mask = decode_rle_mask(seg["counts"], seg["shape"])  # (H, W) uint8, values 0..5
        original_height, original_width = seg["shape"]

        # --- Resize: AREA for the image (smooth), NEAREST for the mask
        # (bilinear/area would blend discrete class labels into invalid values). ---
        image_resized = cv2.resize(
            image.transpose(1, 2, 0),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        ).transpose(2, 0, 1)
        mask_resized = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return {
            "image": torch.from_numpy(image_resized),
            "mask": torch.from_numpy(mask_resized.astype(np.int64)),
            "series_id": int(series_id),
            "pixel_spacing_0": float(row["dicom_series.PixelSpacing0"]),
            "pixel_spacing_1": float(row["dicom_series.PixelSpacing1"]),
            "slice_thickness": float(row["dicom_series.SliceThickness"]),
            "original_height": original_height,
            "original_width": original_width,
        }


def build_slice_df_with_folds(training_df, series_targets_df):
    """Merge slice-level training_df with series-level fold assignments.

    Args:
        training_df: slice-level DataFrame (7508 rows), from training_df.pkl.
        series_targets_df: series-level DataFrame with a "fold" column
            (built earlier via StratifiedGroupKFold).

    Returns:
        training_df with an added "fold" column, one row per slice.
    """
    fold_map = series_targets_df.set_index("series_id")["fold"]
    slice_df = training_df.copy()
    slice_df["fold"] = slice_df["dicom_series.id"].map(fold_map)
    n_unmatched = slice_df["fold"].isna().sum()
    if n_unmatched > 0:
        print(f"Warning: {n_unmatched} slices have no matching series in series_targets_df "
              f"(dropping them).")
        slice_df = slice_df.dropna(subset=["fold"])
    slice_df["fold"] = slice_df["fold"].astype(int)
    return slice_df


def filter_to_available_annotations(slice_df, annotations_root: str):
    """Restrict slice_df to series whose annotation folder actually exists on disk.

    Only ~198 of the 338 training series ship pixel-level annotations (the rest
    have only the series-level aggregate labels) -- verified against the raw
    competition zip, not just the local extraction, so this is a genuine
    dataset limitation, not a re-extractable local issue.

    Returns:
        Filtered slice_df, and prints the resulting per-fold slice/series counts.
    """
    import os

    unique_series = slice_df["dicom_series.id"].unique()
    available_series = {
        sid for sid in unique_series
        if os.path.isdir(os.path.join(annotations_root, str(sid)))
    }
    print(f"Series with annotations available: {len(available_series)} / {len(unique_series)}")

    filtered = slice_df[slice_df["dicom_series.id"].isin(available_series)].reset_index(drop=True)

    print(f"\nSlices per fold (after filtering):")
    print(filtered["fold"].value_counts().sort_index())
    print(f"\nSeries per fold (after filtering):")
    print(filtered.groupby("fold")["dicom_series.id"].nunique())

    return filtered
