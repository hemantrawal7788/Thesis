# MSA-CNN: What Was Added

**Date**: 2026-04-16

---

## Overview

A custom CNN architecture — **MSA-CNN (Multi-Scale Attention 1D CNN)** — was added to the repository as a self-contained Python package (`src/`) and an evaluation notebook (`notebooks/03_Custom_CNN_MSA.ipynb`). It is designed to be trained on UNSW-NB15 and transfer-evaluated on the synthetic IPv6 dataset, following the exact same protocol as the benchmark CNN notebook.

---

## Files Added

### `src/` — Python Package (5 files)

| File | Purpose | Key Exports |
|------|---------|-------------|
| `__init__.py` | Package entry point and exports | — |
| `models.py` | MSA-CNN architecture definition | `MSACNN`, `SEBlock`, `MultiScaleConvBlock`, `ResConvBlock` |
| `losses.py` | Focal loss for class imbalance | `FocalLoss` |
| `dataset.py` | Dataset and DataLoader utilities | `TabularDataset`, `make_weighted_loader`, `make_eval_loader` |
| `training.py` | Training loop and inference helpers | `train_model`, `predict`, `train_one_epoch` |

### `notebooks/03_Custom_CNN_MSA.ipynb` — Evaluation Notebook (15 cells)

End-to-end notebook that:
1. Loads UNSW-NB15 (via HuggingFace) and synthetic IPv6 data
2. Trains MSA-CNN for both binary and multiclass tasks
3. Evaluates on the UNSW-NB15 test split (in-domain)
4. Transfer-evaluates on the synthetic IPv6 test split (out-of-domain)
5. Repeats for 5 seeds and aggregates mean/std
6. Saves all artifacts (models, classification reports, confusion matrices, training curves, drop analysis)

The notebook reuses the same data loading, vectorizer, and evaluation functions as the benchmark CNN notebook so that results are directly comparable.

---

## Architecture

```
Input (B, D) — vectorised tabular features, any dimension D
  │
  ▼ Unsqueeze → (B, 1, D)
  │
  ├── Conv1d(1, 64, k=3, pad=1) → BN → GELU ─┐
  ├── Conv1d(1, 64, k=5, pad=2) → BN → GELU ─┼── Concat → (B, 192, D)
  └── Conv1d(1, 64, k=7, pad=3) → BN → GELU ─┘
  │
  ▼ SE Channel Attention (192 channels)
  ▼ MaxPool1d(2) → (B, 192, D/2)
  │
  ▼ ResConv(192 → 128, k=3) + residual skip
  ▼ SE Channel Attention (128 channels)
  ▼ MaxPool1d(2) → (B, 128, D/4)
  │
  ▼ ResConv(128 → 64, k=3) + residual skip
  ▼ AdaptiveAvgPool1d(1) → (B, 64)
  │
  ▼ Dropout(0.4) → Linear(64, n_classes)
```

**Total parameters**: ~161,000

---

## Design Rationale

Each architectural choice addresses a specific weakness identified in the benchmark CNN models:

| Component | Benchmark Weakness Addressed | How |
|-----------|------------------------------|-----|
| **Multi-scale parallel Conv1d** (k=3, 5, 7) | Benchmark 1D CNNs use a single fixed kernel size | Captures feature interactions at short-range (k=3) and wide-range (k=7) simultaneously |
| **Squeeze-and-Excitation (SE) attention** | Benchmarks treat all convolutional channels equally | Learns per-channel importance — acts as implicit feature selection |
| **Residual connections** | Deep networks degrade on small datasets | Stabilises gradient flow; prevents performance degradation |
| **AdaptiveAvgPool1d** | Benchmark models require a fixed input dimension | Produces fixed-size output for any D — same model works on UNSW-NB15 (~186 features) and synthetic IPv6 (~40 features) |
| **Focal loss** (gamma=2) | `class_weight="balanced"` only reweights uniformly | Downweights well-classified (easy) Benign samples so the model focuses on hard, misclassified attack samples |

### Model Size Comparison

| Model | Parameters |
|-------|-----------|
| Systems2024 Arch1 (benchmark) | ~50-100K |
| Systems2024 Arch2 (benchmark) | ~60-110K |
| **MSA-CNN (ours)** | **~161K** |
| Noever2021 MobileNetV2 (benchmark) | ~3,400K |

MSA-CNN adds modest complexity over the simple 1D CNNs while being 20x smaller than MobileNetV2.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base channels (C) | 64 | Multi-scale block outputs 3C=192 |
| Kernel sizes | (3, 5, 7) | Parallel branches |
| SE reduction ratio | 4 | Channel attention bottleneck |
| Dropout | 0.4 | Before classifier head |
| Focal loss gamma | 2.0 | Paper default; gamma=0 recovers standard CE |
| Learning rate | 1e-3 | AdamW optimiser |
| Weight decay | 1e-4 | L2 regularisation |
| Epochs | 25 | Maximum |
| Early stopping patience | 6 | On validation macro-F1 |
| Seeds | [11, 22, 33, 44, 55] | Same as benchmark |
| Validation split | 15% | Stratified, from training pool |

---

## How to Use

### Import the model in Python

```python
from src.models import MSACNN
from src.losses import FocalLoss

model = MSACNN(n_classes=10, base_channels=64, dropout=0.4)
criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
```

### Run the full evaluation

Open and run `notebooks/03_Custom_CNN_MSA.ipynb`. It will:
- Train on UNSW-NB15 for both binary and multiclass tasks
- Evaluate in-domain (UNSW test) and out-of-domain (synthetic IPv6 test)
- Save all artifacts to `msa_cnn_artifacts/`

### Adjust for Kaggle

If running on Kaggle, copy the `src/` directory into the working directory and update `SYN_ROOT` to the correct Kaggle input path.

---

## Output Artifacts

The notebook produces the following directory structure:

```
msa_cnn_artifacts/
├── unsw/
│   ├── binary/msa_cnn/
│   │   ├── models/          # .pt checkpoints per seed
│   │   ├── tables/          # all_runs.csv, summary.json, support flags
│   │   ├── figures/         # training curves, confusion matrices
│   │   └── reports/         # classification reports
│   └── multiclass/msa_cnn/
│       └── (same structure)
├── synthetic_transfer/
│   ├── binary/msa_cnn/
│   │   └── (same structure)
│   └── multiclass/msa_cnn/
│       └── (same structure)
└── combined/
    ├── msa_cnn_drop_analysis.csv
    └── artefact_dashboard.json
```

---

## Suggested Ablations

The notebook's closing section lists four ablation experiments to strengthen the thesis:

1. **Focal loss gamma sweep** — compare gamma in {0, 1, 2, 3} to quantify the effect of focus weighting vs standard cross-entropy
2. **Multi-scale ablation** — run with single kernel sizes (k=3 only, k=5 only, k=7 only) to verify the benefit of parallel branches
3. **SE attention ablation** — remove SE blocks to isolate the contribution of channel attention
4. **Channel width sweep** — compare base_channels in {32, 64, 128} to find the capacity sweet spot for this dataset size
