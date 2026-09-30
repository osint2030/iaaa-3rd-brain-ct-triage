"""Training and validation loop building blocks for the Brain CT Triage Challenge.

Kept separate from the orchestration script (the notebook cell that loops over
folds/epochs and handles checkpointing) so these pieces are independently
testable and reusable later in the final inference/submission script.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


def build_dataloaders(
    fold, series_targets_df, dicom_dir, dataset_module, augmentation_module,
    batch_size=4, num_workers=2, use_weighted_sampling=True,
):
    """Build train/val DataLoaders for one fold.

    Train = all series NOT in this fold, with series-consistent augmentation.
    Val   = series IN this fold, no augmentation (deterministic evaluation).

    If use_weighted_sampling=True, the train loader uses a WeightedRandomSampler
    so each triage_class is sampled roughly equally often per epoch, rather than
    in proportion to its natural frequency (~49% / 18% / 33%). This directly
    targets the Urgent class (the minority, ~18%), which showed the weakest
    per-fold f1_macro in our diagnostics.
    """
    BrainCTSeriesDataset = dataset_module.BrainCTSeriesDataset
    SeriesAugmentor = augmentation_module.SeriesAugmentor

    train_df = series_targets_df[series_targets_df["fold"] != fold].reset_index(drop=True)
    val_df = series_targets_df[series_targets_df["fold"] == fold].reset_index(drop=True)

    augmentor = SeriesAugmentor()
    train_ds = BrainCTSeriesDataset(train_df, dicom_dir, is_train=True, augment=True, augmentor=augmentor)
    val_ds = BrainCTSeriesDataset(val_df, dicom_dir, is_train=True, augment=False)

    if use_weighted_sampling:
        class_counts = train_df["triage_class"].value_counts()
        class_weight = 1.0 / class_counts
        sample_weights = train_df["triage_class"].map(class_weight).values
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_df), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        print(f"Weighted sampling enabled. Class counts in this fold's train split: "
              f"{class_counts.to_dict()}")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, train_df, val_df


def build_model_and_optimizer(
    model_module, losses_module, device,
    backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-3,
):
    """Build model, loss, and optimizer.

    Uses differential learning rates: a LOW rate for the pretrained backbone
    (which already extracts decent features and shouldn't be perturbed too
    fast) and a HIGHER rate for the freshly-initialized attention/regression/
    fracture heads (which start from random weights and need to move faster
    to catch up).

    Only trainable backbone parameters are added to the optimizer -- early
    ResNet stages are frozen by default inside BrainCTMILModel (see
    freeze_early_backbone), so their .requires_grad is False and they're
    excluded here.
    """
    BrainCTMILModel = model_module.BrainCTMILModel
    BrainCTLoss = losses_module.BrainCTLoss

    net = BrainCTMILModel(pretrained=True).to(device)
    loss_fn = BrainCTLoss()

    head_params = (
        list(net.pooling.parameters())
        + list(net.regression_head.parameters())
        + list(net.fracture_head.parameters())
    )
    backbone_params = [p for p in net.backbone.parameters() if p.requires_grad]

    n_trainable_backbone = sum(p.numel() for p in backbone_params)
    n_frozen_backbone = sum(p.numel() for p in net.backbone.parameters() if not p.requires_grad)
    print(f"Backbone: {n_trainable_backbone:,} trainable params, {n_frozen_backbone:,} frozen params")

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    return net, loss_fn, optimizer


def train_one_epoch(net, loader, loss_fn, optimizer, device):
    """Run one training epoch. Returns the average total loss."""
    net.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        output = net(images, mask)
        loss_dict = loss_fn(output, targets)
        loss_dict["total"].backward()
        optimizer.step()

        total_loss += loss_dict["total"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_one_epoch(net, loader, loss_fn, val_df, device, metrics_module, dataset_module):
    """Run one validation epoch.

    Computes both the average loss AND the actual competition metrics, by
    assembling predicted intermediates for every series in val_df and
    comparing to ground truth via the official triage rule.

    Returns:
        (avg_loss, metrics_dict, pred_df, true_df) where metrics_dict has
        keys "qwk", "accuracy", "f1_macro".
    """
    TARGET_COLUMNS = dataset_module.TARGET_COLUMNS
    net.eval()

    total_loss = 0.0
    n_batches = 0
    all_series_ids = []
    all_preds = []

    for batch in loader:
        images = batch["images"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)
        series_ids = batch["series_id"]

        output = net(images, mask)
        loss_dict = loss_fn(output, targets)
        total_loss += loss_dict["total"].item()
        n_batches += 1

        pred_intermediates = net.to_intermediate_tensor(output)  # (B, 7)
        all_preds.append(pred_intermediates.cpu())

        if torch.is_tensor(series_ids):
            all_series_ids.extend(series_ids.tolist())
        else:
            all_series_ids.extend(list(series_ids))

    pred_tensor = torch.cat(all_preds, dim=0)
    pred_df = pd.DataFrame(pred_tensor.numpy(), columns=TARGET_COLUMNS)
    pred_df["series_id"] = all_series_ids

    # Align true_df to the exact same series_id order as pred_df, since the
    # DataLoader may not preserve val_df's original row order.
    true_df = val_df.set_index("series_id").loc[pred_df["series_id"]].reset_index()

    metrics_dict = metrics_module.compute_metrics(pred_df, true_df)
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, metrics_dict, pred_df, true_df
