"""Reusable DICOM loading and windowing utilities shared between training and inference."""

import glob
import os

import numpy as np
import pydicom


def load_dicom_series(series_dir: str) -> list:
    """Load all DICOM files in a series directory, sorted by physical slice position.

    Args:
        series_dir: Path to a directory containing .dcm files for one series.

    Returns:
        List of pydicom Dataset objects, sorted by ImagePositionPatient z-coordinate.
    """
    dcm_paths = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    slices = [pydicom.dcmread(p) for p in dcm_paths]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    return slices


def to_hu(dcm: pydicom.Dataset) -> np.ndarray:
    """Convert raw DICOM pixel values to Hounsfield Units."""
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    return pixel_array * slope + intercept


def apply_window(hu_image: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply windowing: clip HU to [center-width/2, center+width/2], rescale to [0, 1]."""
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(hu_image, low, high)
    return (clipped - low) / (high - low)


# Standard clinical windows used to build the 3-channel "pseudo-RGB" input.
# Order matters: this defines the channel order (R, G, B) of the output image.
WINDOW_PRESETS = {
    "brain": (40, 80),
    "blood": (75, 215),
    "bone": (500, 2500),
}


def make_multiwindow_image(dcm: pydicom.Dataset) -> np.ndarray:
    """Build a 3-channel image from a single DICOM slice using brain/blood/bone windows.

    Returns:
        Array of shape (3, H, W), each channel in range [0, 1].
    """
    hu_image = to_hu(dcm)
    channels = [
        apply_window(hu_image, center, width)
        for center, width in WINDOW_PRESETS.values()
    ]
    return np.stack(channels, axis=0).astype(np.float32)
