"""PyTorch Dataset for the Brain CT Triage Challenge.

Each item represents one full CT series (a "bag" of slices for MIL-style processing),
padded to a fixed maximum number of slices with an accompanying validity mask.

Augmentation, when enabled, is sampled ONCE per series (via SeriesAugmentor) and
applied identically to every slice in that series, preserving 3D anatomical
coherence (e.g. a horizontal flip must be the same across all slices of one scan).
"""

import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.augmentation import SeriesAugmentor
from src.preprocessing import load_dicom_series, make_multiwindow_image

TARGET_COLUMNS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "fracture_prob", "MLS_mm"]


class BrainCTSeriesDataset(Dataset):
    """Loads a full CT series as a padded stack of multi-window 2D images.

    Args:
        targets_df: DataFrame with one row per series, containing "series_id",
            the columns in TARGET_COLUMNS, and (for training) "fold".
        dicom_root: Directory containing one subfolder per series_id with .dcm files.
        image_size: Output height/width for each slice after resizing.
        max_slices: Fixed number of slices to pad/truncate each series to.
        is_train: Whether this dataset provides targets (True) or is inference-only (False).
        augment: Whether to apply series-consistent augmentation. Must be False for
            validation/inference. Ignored (treated as False) if augmentor is None.
        augmentor: A SeriesAugmentor instance controlling augmentation behavior.
            Required if augment=True. If augment=False, this is unused and may be None.
    """

    def __init__(
        self,
        targets_df: pd.DataFrame,
        dicom_root: str,
        image_size: int = 224,
        max_slices: int = 50,
        is_train: bool = True,
        augment: bool = False,
        augmentor: Optional[SeriesAugmentor] = None,
    ):
        if augment and augmentor is None:
            raise ValueError("augment=True requires a SeriesAugmentor instance (augmentor=...).")

        self.targets_df = targets_df.reset_index(drop=True)
        self.dicom_root = dicom_root
        self.image_size = image_size
        self.max_slices = max_slices
        self.is_train = is_train
        self.augment = augment
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.targets_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.targets_df.iloc[idx]
        series_id = row["series_id"]
        series_dir = os.path.join(self.dicom_root, str(series_id))

        dcm_slices = load_dicom_series(series_dir)
        n_slices = len(dcm_slices)

        images = np.zeros((self.max_slices, 3, self.image_size, self.image_size), dtype=np.float32)
        mask = np.zeros(self.max_slices, dtype=np.float32)

        n_to_load = min(n_slices, self.max_slices)

        # Sample augmentation parameters ONCE per series (not per slice), so that
        # flip/rotation/window-jitter stay consistent across the whole scan.
        series_params = self.augmentor.sample_series_params() if self.augment else None

        for i in range(n_to_load):
            if self.augment:
                resized = self.augmentor.apply_to_slice(
                    dcm_slices[i], series_params, self.image_size
                )
            else:
                multiwindow = make_multiwindow_image(dcm_slices[i])
                resized = cv2.resize(
                    multiwindow.transpose(1, 2, 0),
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA,
                ).transpose(2, 0, 1)
            images[i] = resized
            mask[i] = 1.0

        # Keep series_id as int when the folder name is purely numeric (our training
        # data), but fall back to the original string otherwise -- the hidden test
        # set's folder naming convention is not guaranteed to match (the official
        # docs show a "series_0"-style example, unlike our numeric training IDs).
        try:
            series_id_value = int(series_id)
        except (ValueError, TypeError):
            series_id_value = str(series_id)

        sample = {
            "images": torch.from_numpy(images),
            "mask": torch.from_numpy(mask),
            "series_id": series_id_value,
        }

        if self.is_train:
            targets = row[TARGET_COLUMNS].values.astype(np.float32)
            sample["targets"] = torch.from_numpy(targets)

        return sample
