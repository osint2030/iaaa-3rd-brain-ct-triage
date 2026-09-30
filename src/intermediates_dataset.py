"""Per-slice dataset for IntermediatesModel: loads a 2.5D multi-window CT
image plus physical metadata (slice thickness, relative z-position from
ImagePositionPatient) and the REAL per-slice MLS/fracture labels.
"""
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from src.preprocessing import to_hu, apply_window, WINDOW_PRESETS


class IntermediatesSliceDataset(Dataset):
    def __init__(self, slice_df, dicom_root: str, image_size: int = 224):
        self.slice_df = slice_df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.slice_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.slice_df.iloc[idx]
        series_id = row["dicom_series.id"]
        sop_uid = row["dicom_series.SOPInstanceUID"]

        dcm_path = os.path.join(self.dicom_root, str(series_id), f"{sop_uid}.dcm")
        dcm = pydicom.dcmread(dcm_path)
        hu_image = to_hu(dcm)
        channels = [apply_window(hu_image, center, width) for center, width in WINDOW_PRESETS.values()]
        image = np.stack(channels, axis=0).astype(np.float32)

        image_resized = cv2.resize(
            image.transpose(1, 2, 0), (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        ).transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(image_resized),
            "slice_thickness": float(row["dicom_series.SliceThickness"]),
            "z_rel": float(row["z_rel"]),
            "mls_target": float(row["MidlineShiftMM"]),
            "fracture_target": float(row["SkullFracture"]),
            "series_id": int(series_id),
        }


def build_slice_df_with_z_rel(metadata_df, series_targets_df, dicom_root: str) -> pd.DataFrame:
    slice_df = metadata_df.merge(
        series_targets_df[["series_id", "fold"]],
        left_on="dicom_series.id", right_on="series_id", how="inner",
    )

    z_positions = []
    for _, row in slice_df.iterrows():
        dcm_path = os.path.join(
            dicom_root, str(row["dicom_series.id"]),
            f"{row['dicom_series.SOPInstanceUID']}.dcm",
        )
        dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        z_positions.append(float(dcm.ImagePositionPatient[2]))
    slice_df["z_position"] = z_positions

    z_min = slice_df.groupby("dicom_series.id")["z_position"].transform("min")
    z_max = slice_df.groupby("dicom_series.id")["z_position"].transform("max")
    z_range = (z_max - z_min).replace(0, 1.0)
    slice_df["z_rel"] = (slice_df["z_position"] - z_min) / z_range

    return slice_df
