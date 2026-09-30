"""Series-consistent augmentation utilities for the Brain CT dataset.

Key design principle: augmentation parameters are sampled ONCE per series
and applied identically to every slice, preserving 3D anatomical coherence.

Performance note: all spatial operations (flip, rotation, resize) are applied
to the full 3-channel image in a single call (channel-last layout), rather than
looping per channel. cv2 natively supports multi-channel arrays for warpAffine
and resize, so looping was pure overhead (3x the function calls and array
copies per slice for no benefit).
"""

import random

import cv2
import numpy as np

from src.preprocessing import to_hu, apply_window, WINDOW_PRESETS


class SeriesAugmentor:
    """Samples a consistent set of augmentation parameters for one series,
    then applies them to each slice individually.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        max_rotation_deg: float = 10.0,
        window_center_jitter: float = 5.0,
        window_width_jitter: float = 10.0,
        noise_std: float = 0.01,
    ):
        self.flip_prob = flip_prob
        self.max_rotation_deg = max_rotation_deg
        self.window_center_jitter = window_center_jitter
        self.window_width_jitter = window_width_jitter
        self.noise_std = noise_std

    def sample_series_params(self) -> dict:
        """Sample augmentation parameters once, to be reused across all slices in a series."""
        do_flip = random.random() < self.flip_prob
        rotation_deg = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)

        jittered_windows = {}
        for name, (center, width) in WINDOW_PRESETS.items():
            jittered_windows[name] = (
                center + random.uniform(-self.window_center_jitter, self.window_center_jitter),
                width + random.uniform(-self.window_width_jitter, self.window_width_jitter),
            )

        return {
            "do_flip": do_flip,
            "rotation_deg": rotation_deg,
            "windows": jittered_windows,
        }

    def apply_to_slice(self, dcm, params: dict, image_size: int) -> np.ndarray:
        """Build an augmented multi-window image for a single slice using series-level params."""
        hu_image = to_hu(dcm)

        channels = [
            apply_window(hu_image, center, width)
            for center, width in params["windows"].values()
        ]
        # Channel-last (H, W, 3) layout so every cv2 op below runs ONCE for all
        # 3 channels together, instead of once per channel.
        image = np.stack(channels, axis=-1)

        if params["do_flip"]:
            image = np.ascontiguousarray(image[:, ::-1, :])

        h, w = image.shape[0], image.shape[1]
        rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), params["rotation_deg"], scale=1.0)
        rotated = cv2.warpAffine(
            image, rot_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        resized = cv2.resize(
            rotated, (image_size, image_size), interpolation=cv2.INTER_AREA
        )

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, resized.shape).astype(np.float32)
            resized = np.clip(resized + noise, 0.0, 1.0)

        # Back to (3, H, W) channel-first, as expected by the rest of the pipeline.
        return resized.transpose(2, 0, 1).astype(np.float32)
