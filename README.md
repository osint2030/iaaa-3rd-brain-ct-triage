# IAAA Brain CT Triage Challenge

A deep-learning pipeline that predicts triage urgency (Non-Urgent / Urgent /
Critical) from non-contrast head CT scans, by estimating seven clinically
meaningful intermediates — five hemorrhage-subtype volumes, skull-fracture
probability, and midline shift — and applying the competition's fixed,
rule-based triage function to them.

**Official score: `f1_macro = 0.8551`** (up from a `0.3427` multiple-instance-learning
baseline — see [Results](#results) and [Engineering Journey](#engineering-journey--key-findings)).

## Table of Contents
- [Problem](#problem)
- [Approach](#approach)
- [Results](#results)
- [Engineering Journey & Key Findings](#engineering-journey--key-findings)
- [Repository Structure](#repository-structure)
- [Reproducing This Work](#reproducing-this-work)
- [Tech Stack](#tech-stack)

## Problem

Given a non-contrast head CT series (a stack of 2D DICOM slices), predict one
of three triage classes:

| Class | Meaning |
|---|---|
| 0 | Non-Urgent |
| 1 | Urgent |
| 2 | Critical |

The competition scores submissions by **`f1_macro`** against a hidden test
set. Critically, the final triage decision is **not learned** — it's a fixed
rule (`triage_from_intermediates`, ported verbatim from the official docs)
applied to seven predicted intermediates:

- Five hemorrhage-subtype volumes in mL: epidural (EDH), subdural (SDH),
  intraparenchymal (IPH), subarachnoid (SAH), intraventricular (IVH)
- Skull fracture probability
- Midline shift (MLS) in mm

This reframes the problem: instead of training an end-to-end classifier, the
whole system is built to estimate these seven imaging biomarkers as
accurately as possible.

## Approach

**Pipeline evolution:**

1. **MIL baseline** — an attention-based Multiple Instance Learning classifier
   trained directly on triage labels. Ceiling: `f1_macro=0.34` (confirmed via
   an internal-CV-vs-hidden-test diagnostic to be a real architecture
   ceiling, not overfitting — see notebook Section 12).
2. **Pivot to a hybrid, intermediates-first pipeline** (the approach that
   shipped):
   - **Segmentation model** (U-Net, ResNet18 encoder, 5-fold ensemble) — predicts
     a 6-class (background + 5 hemorrhage types) pixel mask per slice;
     volumes are aggregated from pixel counts × physical voxel spacing.
   - **IntermediatesModel** (ResNet18 + physical-embedding MLP, 5-fold
     ensemble) — predicts fracture probability and midline shift directly
     from each slice.
   - **Official triage rule** — deterministic, applied to the aggregated
     predictions.

Both model families are combined in a single self-contained `submission.py`
with no external dependencies beyond standard libraries, verified end-to-end
before every official submission (see `notebooks/`, Sections 11 & 23).

## Results

| Version | f1_macro | Notes |
|---|---|---|
| MIL baseline (v1) | 0.343 | Direct end-to-end classifier; architecture ceiling |
| Hybrid pipeline (v1) | 0.830 | Segmentation + IntermediatesModel + triage rule |
| **Hybrid pipeline (fracture fix)** | **0.855** | Fixed a noisy-OR aggregation bug (see below) |

## Engineering Journey & Key Findings

This project's real value is in the debugging and validation discipline
behind these numbers, not just the final architecture. A few highlights
(full details in `notebooks/`, each linked to its section):

- **A confounded experiment, caught before it mattered.** A weighted-loss
  fix for midline-shift regression looked like a clear win — until a
  paired significance test revealed the comparison had silently changed
  *two* variables at once (loss weighting **and** checkpoint-selection
  criterion). A deconfounded re-test showed no real effect. *(Section 21.7)*
- **A canary test to rule out evaluator-side caching.** Two resubmissions
  produced byte-identical scores despite believed code changes. Rather than
  assume a caching bug, a diagnostic submission (all-zero predictions,
  cheap and fully predictable) confirmed the evaluator *does* re-execute
  code — the real explanation was a genuine null effect of one specific
  filter on that hidden test set. *(Section 25.5)*
- **A physics-based hypothesis, tested and rejected.** Suspected
  false-positive hemorrhage detections were hypothesized to be calcification
  (distinguishable by Hounsfield Unit). Direct native-resolution HU
  measurement showed every case sat in the blood-HU range, not
  calcification — hypothesis rejected by data, not assumption. *(Section 26)*
- **The single highest-leverage fix: a math bug in aggregation, not the
  model.** Two false-Urgent series traced to `aggregate_fracture` using
  noisy-OR (`1 - ∏(1-p_i)`) across slices — a formula that compounds many
  individually-uncertain per-slice predictions into a falsely confident
  aggregate. Switching to max-pooling (matching the pattern already used for
  midline shift) fixed both known false positives with zero true positives
  lost, and delivered the largest single score improvement in the project
  (`+0.025` on the official leaderboard). *(Section 27)*

The throughline: every fix that shipped was one that had been **measured**
against real data first — hypotheses that didn't survive that test (the MLS
reweighting, the HU calcification filter) were documented and abandoned
rather than force-fit.

## Repository Structure

```
.
├── README.md
├── notebooks/
│   └── IAAA_Brain_CT_Triage_MASTER.ipynb   # full pipeline, 25+ documented sections
├── src/
│   ├── preprocessing.py      # DICOM loading, HU conversion, multi-window imaging
│   ├── dataset.py            # BrainCTSeriesDataset (MIL / IntermediatesModel)
│   ├── seg_dataset.py        # SegmentationSliceDataset
│   ├── model.py               # model architectures
│   └── augmentation.py       # series-consistent augmentation
├── submission/
│   └── submission.py         # self-contained, competition-ready inference script
├── checkpoints/               # trained model weights (Git LFS)
│   ├── segmentation_v2/       # 5-fold U-Net ensemble
│   └── intermediates_v1/      # 5-fold IntermediatesModel ensemble
├── requirements.txt
└── .gitattributes             # Git LFS config for checkpoints/
```

## Reproducing This Work

```bash
git clone <this-repo>
cd iaaa-brain-ct-triage
git lfs pull                  # fetch checkpoint weights
pip install -r requirements.txt
```

- **Full pipeline walkthrough:** open `notebooks/IAAA_Brain_CT_Triage_MASTER.ipynb`
  (designed for Google Colab; expects the competition data mounted from
  Google Drive — see Section 1 for setup).
- **Just want predictions:** run the standalone script directly —
  ```bash
  python submission/submission.py \
      --data-dir /path/to/series-folders \
      --predictions-file-path /path/to/output.csv
  ```

## Tech Stack

PyTorch · segmentation-models-pytorch (U-Net/ResNet18) · pydicom ·
scikit-learn · pandas / NumPy · OpenCV · SciPy (`ndimage` for 3D connected
components) · Google Colab (GPU training) · WSL2/Ubuntu + Miniconda (local dev)
