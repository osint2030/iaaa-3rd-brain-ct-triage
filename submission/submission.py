#!/usr/bin/env python
"""Self-contained hybrid submission entry point for the IAAA Brain CT Triage
Challenge. Combines two independently-trained models:

  1. HemorrhageSegModel (U-Net, segmentation-models-pytorch) -- predicts
     per-slice pixel masks for 5 hemorrhage subtypes, aggregated into
     series-level volumes (mL) via the physical formula
     pixel_count x PixelSpacing0 x PixelSpacing1 x SliceThickness / 1000.
  2. IntermediatesModel (2.5D CNN + physical embedding) -- predicts
     per-slice MLS_mm and fracture probability, aggregated into series-level
     values via hard-max pooling (MLS) and complement-product pooling
     (fracture), matching each variable's physical/clinical meaning.

Both are 5-fold ensembles (probability/prediction averaging across folds,
NOT a single-fold prediction), trained independently on the same
patient-grouped, class-stratified cross-validation splits.

Deliberately self-contained (no dependency on the custom "src" package),
matching the same rationale as the original MIL submission.py: the
evaluation harness's file-packaging behavior for arbitrary directories is
unknown/unreliable, so this file only depends on the `models/` directory
sitting next to it.

IMPORTANT: architectures below must exactly match the checkpoints' state_dict
keys. Verified against:
  - HemorrhageSegModel: smp.Unet(encoder_name="resnet18", in_channels=3, classes=6)
  - IntermediatesModel: ResNet18 backbone + 32-dim physical embedding,
    Softplus (NOT ReLU) for MLS non-negativity, bias-initialized fracture head.

Official usage (per BCT.md):
    python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv

Expected directory layout alongside this script:
    submission.py
    models/
        seg_fold_0_best.pt ... seg_fold_4_best.pt
        intermediates_fold_0_best.pt ... intermediates_fold_4_best.pt
"""

import argparse
import glob
import os

import cv2
import numpy as np
from scipy import ndimage
import pandas as pd
import pydicom
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader, Dataset

WINDOW_PRESETS = {
    "brain": (40, 80),
    "blood": (75, 215),
    "bone": (500, 2500),
}
IMAGE_SIZE = 224

TARGET_COLUMNS = ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "fracture_prob", "MLS_mm"]

CLASS_VALUE_TO_TARGET_KEY = {
    1: "V_IVH",
    2: "V_IPH",
    3: "V_SDH",
    4: "V_EDH",
    5: "V_SAH",
}
N_CLASSES = 6


def load_dicom_series(series_dir: str) -> list:
    dcm_paths = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    slices = [pydicom.dcmread(p) for p in dcm_paths]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    return slices


def to_hu(dcm) -> np.ndarray:
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    return pixel_array * slope + intercept


def apply_window(hu_image: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2
    high = center + width / 2
    clipped = np.clip(hu_image, low, high)
    return (clipped - low) / (high - low)


def make_multiwindow_image(dcm) -> np.ndarray:
    hu_image = to_hu(dcm)
    channels = [apply_window(hu_image, center, width) for center, width in WINDOW_PRESETS.values()]
    return np.stack(channels, axis=0).astype(np.float32)


class InferenceSeriesDataset(Dataset):
    def __init__(self, series_ids: list, dicom_root: str, image_size: int = IMAGE_SIZE):
        self.series_ids = list(series_ids)
        self.dicom_root = dicom_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.series_ids)

    def __getitem__(self, idx: int) -> dict:
        series_id = self.series_ids[idx]
        series_dir = os.path.join(self.dicom_root, str(series_id))
        dcm_slices = load_dicom_series(series_dir)

        z_positions = [float(dcm.ImagePositionPatient[2]) for dcm in dcm_slices]
        z_min, z_max = min(z_positions), max(z_positions)
        z_range = (z_max - z_min) if (z_max - z_min) != 0 else 1.0

        slices = []
        for dcm, z_pos in zip(dcm_slices, z_positions):
            multiwindow = make_multiwindow_image(dcm)
            native_height, native_width = multiwindow.shape[1], multiwindow.shape[2]

            resized = cv2.resize(
                multiwindow.transpose(1, 2, 0), (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            ).transpose(2, 0, 1)

            pixel_spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
            slice_thickness = float(getattr(dcm, "SliceThickness", 1.0))

            slices.append({
                "image": torch.from_numpy(resized.astype(np.float32)),
                "slice_thickness": slice_thickness,
                "pixel_spacing_0": float(pixel_spacing[0]),
                "pixel_spacing_1": float(pixel_spacing[1]),
                "original_height": native_height,
                "original_width": native_width,
                "z_rel": (z_pos - z_min) / z_range,
            })

        try:
            series_id_value = int(series_id)
        except (ValueError, TypeError):
            series_id_value = str(series_id)

        return {"series_id": series_id_value, "slices": slices}


def collate_single_series(batch: list) -> dict:
    return batch[0]


SEG_IMAGENET_MEAN = [0.485, 0.456, 0.406]
SEG_IMAGENET_STD = [0.229, 0.224, 0.225]


class HemorrhageSegModel(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=N_CLASSES,
        )
        self.register_buffer("imagenet_mean", torch.tensor(SEG_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(SEG_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.imagenet_mean) / self.imagenet_std
        return self.unet(normalized)


class PhysicalEmbedding(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim))

    def forward(self, slice_thickness: torch.Tensor, z_rel: torch.Tensor) -> torch.Tensor:
        x = torch.stack([slice_thickness, z_rel], dim=1)
        return self.mlp(x)


class IntermediatesModel(nn.Module):
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
        self.mls_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.fracture_head = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
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
        return {
            "mls_logit": self.mls_head(combined).squeeze(-1),
            "fracture_logit": self.fracture_head(combined).squeeze(-1),
        }

    @staticmethod
    def mls_from_logit(mls_logit: torch.Tensor) -> torch.Tensor:
        return F.softplus(mls_logit)

    @staticmethod
    def aggregate_mls(mls_per_slice_nonneg: torch.Tensor) -> torch.Tensor:
        return mls_per_slice_nonneg.max()
    @staticmethod
    def aggregate_fracture(fracture_prob_per_slice: torch.Tensor) -> torch.Tensor:
        # Changed from noisy-OR (1 - prod(1-p_i)) to max pooling: the noisy-OR
        # formula compounds many individually-uncertain slices (e.g. 13 slices
        # at ~0.2 each) into a false near-certain aggregate. Validated against
        # 2 known false positives (271569, 658: dropped from 0.96/0.83 to
        # 0.25/0.26) and 6 genuine fracture-positive series (stayed at
        # 0.70-0.87) -- clean separation, no true positives lost.
        return fracture_prob_per_slice.max()


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")
N_FOLDS = 5


def load_ensembles(device: str):
    seg_models, intermediates_models = [], []
    for fold in range(N_FOLDS):
        seg_ckpt_path = os.path.join(MODELS_DIR, f"seg_fold_{fold}_best.pt")
        if not os.path.isfile(seg_ckpt_path):
            raise FileNotFoundError(f"Missing segmentation checkpoint: {seg_ckpt_path}")
        seg_net = HemorrhageSegModel(pretrained=False).to(device)
        seg_ckpt = torch.load(seg_ckpt_path, map_location=device, weights_only=False)
        seg_net.load_state_dict(seg_ckpt["model_state"])
        seg_net.eval()
        seg_models.append(seg_net)

        int_ckpt_path = os.path.join(MODELS_DIR, f"intermediates_fold_{fold}_best.pt")
        if not os.path.isfile(int_ckpt_path):
            raise FileNotFoundError(f"Missing intermediates checkpoint: {int_ckpt_path}")
        int_net = IntermediatesModel(pretrained=False).to(device)
        int_ckpt = torch.load(int_ckpt_path, map_location=device, weights_only=False)
        int_net.load_state_dict(int_ckpt["model_state"])
        int_net.eval()
        intermediates_models.append(int_net)

    return seg_models, intermediates_models


# --- v2: 3D connected-component filtering added on top of raw per-slice
# pixel counting. A hemorrhage-class component is only counted toward volume
# if it spans >=MIN_SLICES_SPANNED contiguous slices, OR is already large
# enough on its own (>=LARGE_BLOB_OVERRIDE_ML) to plausibly be real even in a
# single thick slice. This removes single-slice segmentation noise (partial
# volume effect, small artifacts) while leaving genuine multi-slice
# hemorrhages untouched. Empirically validated: fixes 16/17 known noise-scale
# false-Urgent cases locally with zero change to Critical-class performance.
MIN_SLICES_SPANNED = 2
LARGE_BLOB_OVERRIDE_ML = 2.0


@torch.no_grad()
def predict_series(slices: list, seg_models: list, intermediates_models: list, device: str) -> dict:
    pred_masks = []
    effective_spacings = []
    mls_per_slice, fracture_per_slice = [], []

    for s in slices:
        image = s["image"].unsqueeze(0).to(device)

        seg_probs_sum = None
        for seg_net in seg_models:
            logits = seg_net(image)
            probs = torch.softmax(logits, dim=1)
            seg_probs_sum = probs if seg_probs_sum is None else seg_probs_sum + probs
        seg_probs_mean = seg_probs_sum / len(seg_models)
        pred_mask = seg_probs_mean.argmax(dim=1).squeeze(0).cpu().numpy()
        pred_masks.append(pred_mask)

        h_scale = s["original_height"] / IMAGE_SIZE
        w_scale = s["original_width"] / IMAGE_SIZE
        effective_spacings.append((
            s["pixel_spacing_0"] * h_scale,
            s["pixel_spacing_1"] * w_scale,
            s["slice_thickness"],
        ))

        thickness_t = torch.tensor([s["slice_thickness"]]).float().to(device)
        z_rel_t = torch.tensor([s["z_rel"]]).float().to(device)

        mls_sum, fracture_sum = 0.0, 0.0
        for int_net in intermediates_models:
            out = int_net.forward_per_slice(image, thickness_t, z_rel_t)
            mls_sum += IntermediatesModel.mls_from_logit(out["mls_logit"]).item()
            fracture_sum += torch.sigmoid(out["fracture_logit"]).item()
        mls_per_slice.append(mls_sum / len(intermediates_models))
        fracture_per_slice.append(fracture_sum / len(intermediates_models))

    mask_stack = np.stack(pred_masks, axis=0)  # (n_slices, H, W)
    volumes = {key: 0.0 for key in CLASS_VALUE_TO_TARGET_KEY.values()}
    structure_6conn = ndimage.generate_binary_structure(3, 1)  # face connectivity only

    for class_value, target_key in CLASS_VALUE_TO_TARGET_KEY.items():
        binary_stack = (mask_stack == class_value)
        if not binary_stack.any():
            continue

        labeled, n_components = ndimage.label(binary_stack, structure=structure_6conn)

        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            slice_indices = np.where(comp_mask.any(axis=(1, 2)))[0]

            comp_volume_ml = 0.0
            for z in slice_indices:
                pixel_count = int(comp_mask[z].sum())
                sp0, sp1, thickness = effective_spacings[z]
                comp_volume_ml += pixel_count * sp0 * sp1 * thickness / 1000.0

            keep = (len(slice_indices) >= MIN_SLICES_SPANNED) or (comp_volume_ml >= LARGE_BLOB_OVERRIDE_ML)
            if keep:
                volumes[target_key] += comp_volume_ml

    pred_mls = IntermediatesModel.aggregate_mls(torch.tensor(mls_per_slice)).item()
    pred_fracture = IntermediatesModel.aggregate_fracture(torch.tensor(fracture_per_slice)).item()

    return {**volumes, "MLS_mm": pred_mls, "fracture_prob": pred_fracture}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with one subfolder per series.")
    parser.add_argument("--predictions-file-path", required=True, help="Output CSV path.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    series_ids = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )
    print(f"Found {len(series_ids)} series in {args.data_dir}")
    if len(series_ids) == 0:
        raise RuntimeError(f"No series subdirectories found under {args.data_dir}")

    dataset = InferenceSeriesDataset(series_ids, args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_single_series)

    seg_models, intermediates_models = load_ensembles(device)
    print(f"Loaded {len(seg_models)} segmentation folds and {len(intermediates_models)} intermediates folds.")

    rows = []
    for item in loader:
        series_id = item["series_id"]
        intermediates = predict_series(item["slices"], seg_models, intermediates_models, device)
        rows.append({"series_id": series_id, **intermediates})
        print(f"  series {series_id}: done ({len(item['slices'])} slices)")

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df[["series_id"] + TARGET_COLUMNS]
    pred_df.to_csv(args.predictions_file_path, index=False)
    print(f"Saved predictions for {len(pred_df)} series to {args.predictions_file_path}")


if __name__ == "__main__":
    main()
