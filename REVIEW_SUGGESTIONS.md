# Repository Review & Improvement Suggestions

**Repo**: Hemant — Network Intrusion Detection (UNSW-NB15 + Synthetic IPv6)
**Date**: 2026-04-15

---

## Table of Contents

1. [Overview](#overview)
2. [What's Done Well](#whats-done-well)
3. [Critical Issues (P0)](#critical-issues-p0)
4. [High-Priority Issues (P1)](#high-priority-issues-p1)
5. [Medium-Priority Issues (P2)](#medium-priority-issues-p2)
6. [Low-Priority Issues (P3)](#low-priority-issues-p3)
7. [Methodology / ML-Specific Feedback](#methodology--ml-specific-feedback)
8. [Summary Table](#summary-table)

---

## Overview

This repo contains work on **network intrusion detection** using the UNSW-NB15 dataset and a synthetic IPv6 variant. It has two main tracks:

- **Logistic Regression baselines** (`notebooks/01_LR_Multiclass_5Seed_Eval.ipynb`, `notebooks/02_LR_Binary_PR_Threshold.ipynb`) — multiclass 5-seed evaluation + binary PR-threshold selection.
- **CNN benchmarks** (`cnn-on-unsw-nb15-benchmarks-annotated.ipynb`) — reproducing 3 literature CNN architectures on UNSW-NB15, then testing transfer to synthetic IPv6 data.

---

## What's Done Well

- **Structured evaluation protocol**: The 5-seed loop with macro-F1 mean +/- std and the unstable-support flagging (classes with <5 test samples) shows good experimental rigor.
- **Binary threshold selection on validation**: Choosing the threshold via max-F1 on the PR curve *on the validation set*, then reporting on a held-out test set, is methodologically correct and avoids data leakage.
- **CNN benchmark notebook is well-organized**: Clean section numbering (0-10), artifact dashboarding, paper-vs-reproduced comparison tables, and seed-averaged summaries.
- **Artifact saving discipline**: JSON summaries, CSVs, classification reports, confusion matrices, and PR curve plots are all persisted — making results auditable.
- **Proper use of `class_weight="balanced"`**: Appropriate for this level of class imbalance.
- **Transfer evaluation design**: Training on UNSW-NB15 and evaluating on synthetic IPv6 is a meaningful out-of-distribution test.

---

## Critical Issues (P0)

### 1. Git LFS is configured but broken — result CSVs are empty stubs

**Problem**: `.gitattributes` tracks `*.csv` via Git LFS, but LFS is not installed or not properly configured. All CSV files in `data/UNSW-NB15/` and `results/` are **LFS pointer files** (~129 bytes each), not actual data.

Affected files include:
- `results/lr_multiclass_confusion_matrix.csv`
- `results/lr_binary_confusion_matrix_test.csv`
- `results/test_support_unstable_flags.csv`
- `results/lr_multiclass_5seed_runs.csv`
- All CSVs under `data/UNSW-NB15/`

**Impact**: Anyone cloning this repo gets empty pointer files instead of actual data. Results cannot be inspected.

**Fix**:
- **Option A**: Install Git LFS (`brew install git-lfs && git lfs install`), then `git lfs pull` to materialize the files. Ensure the LFS remote is configured.
- **Option B** (recommended for small CSVs): Remove LFS tracking for result CSVs (they are <1KB — LFS adds overhead for no benefit). Keep LFS only for large data files.

```bash
# Remove LFS tracking for small result CSVs
git lfs untrack "*.csv"
git lfs track "data/UNSW-NB15/*.csv"   # keep LFS only for large datasets
git add .gitattributes
```

### 2. Typo in `.gitattributes` — `*.cvs` instead of `*.csv`

**Problem**: Line 2 of `.gitattributes` reads:
```
*.cvs filter=lfs diff=lfs merge=lfs -text
```

This tracks `*.cvs` files (which don't exist) instead of being a useful rule. This is a silent misconfiguration.

**Fix**: Either remove the line (line 1 already covers `*.csv`) or correct the typo.

---

## High-Priority Issues (P1)

### 3. The 5-seed multiclass experiment shows zero variance (std = 0.0)

**Problem**: All 5 seeds produce **identical results**:

| seed | accuracy | macro_f1 |
|------|----------|----------|
| 11   | 0.9244   | 0.3980   |
| 22   | 0.9244   | 0.3980   |
| 33   | 0.9244   | 0.3980   |
| 44   | 0.9244   | 0.3980   |
| 55   | 0.9244   | 0.3980   |

The `random_state` parameter on `LogisticRegression(solver="saga")` only affects the optimization initialization. With clean, separable data, SAGA converges to the same solution regardless of seed.

**Impact**: The 5-seed evaluation is **not actually testing stability**. A std of 0.0 doesn't mean the model is perfectly stable — it means the experiment design doesn't introduce enough variation to measure instability.

**Fix**: Use one or more of:
- **Bootstrap resampling** of the training set across seeds
- **Stratified k-fold cross-validation** (e.g., 5-fold)
- **Subsampling** different fractions of the training data

Example with bootstrap:
```python
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    X_boot, y_boot = X_train.iloc[idx], y_train[idx]
    # train on X_boot, y_boot ...
```

### 4. `src/` directory referenced in README but does not exist

**Problem**: The README documents:
```
src/ (optional scripts)
  eval_lr_multiclass_5seed.py
  eval_lr_binary_pr.py
```

But the `src/` directory was never created. The repo has no `.py` files at all.

**Fix**: Either create the `src/` directory with the promised scripts, or update the README to remove the reference.

### 5. `requirements.txt` is incomplete and unpinned

**Problem**: Current `requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
```

Issues:
- **No version pins** — different versions may produce different results or break entirely.
- **Missing dependencies** for the CNN notebook: `torch`, `torchvision`, `huggingface_hub`.
- No Python version requirement specified.

**Fix**:
```
# requirements.txt (LR baselines)
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
matplotlib==3.9.3

# requirements-cnn.txt (CNN benchmarks — additional)
torch==2.5.1
torchvision==0.20.1
huggingface_hub==0.27.1
```

Or use a single file with sections, and specify `python_requires >= 3.10` in a `pyproject.toml`.

---

## Medium-Priority Issues (P2)

### 6. Hardcoded Kaggle paths break local execution

**Problem**: Both LR notebooks and the CNN notebook use:
```python
ROOT = "/kaggle/input/datasets/kashyap1264/synthetic-data/synthetic_ipv6_grounded_v3_32x32"
```

This is a **Kaggle-specific absolute path** that fails on any local machine. The README says to "place the zip in the repo root" but the code doesn't reflect that.

**Fix**: Use a relative path or environment variable with a fallback:
```python
import os
ROOT = os.environ.get("DATA_ROOT", "data/synthetic_ipv6_grounded_v3_32x32")
```

### 7. Duplicate code between notebooks

**Problem**: The two LR notebooks share ~80% of their setup code:
- Imports
- Data loading and merge logic
- `split_xy()` function
- Preprocessing pipeline (`ColumnTransformer` with `OneHotEncoder` + `StandardScaler`)

Any change (e.g., adding a feature column, fixing a preprocessing bug) needs to be applied in both places.

**Fix**: Extract shared logic into a Python module:
```
src/
  data_loader.py      # load_splits(), split_xy()
  preprocessing.py    # build_preprocessor()
  evaluation.py       # support_flag_table(), save artifacts
```

Both notebooks then become thin wrappers:
```python
from src.data_loader import load_splits, split_xy
from src.preprocessing import build_preprocessor
```

### 8. The CNN notebook (3.8MB) has embedded outputs

**Problem**: `cnn-on-unsw-nb15-benchmarks-annotated.ipynb` is 3.8MB with embedded cell outputs (plots, tables, training logs). This makes:
- Git diffs nearly impossible to review
- Every re-run changes cell output metadata, creating noisy commits
- The repo larger than necessary

**Fix**:
- Install `nbstripout` and add it as a pre-commit hook:
  ```bash
  pip install nbstripout
  nbstripout --install
  ```
- Alternatively, use `jupyter nbconvert --ClearOutputPreprocessor.enabled=True` before committing.
- Consider modularizing the CNN code into `.py` files (model definitions, training loop, evaluation) with the notebook as an orchestration layer.

### 9. 3,000+ PNG images committed directly to Git

**Problem**: All 32x32 thumbnail PNGs under `data/synthetic_ipv6_grounded_v3_32x32/images/` are committed to Git. While individually small, they permanently bloat the repo history and make clones slow.

**Fix**:
- **Option A**: Track images with Git LFS (they are binary files — good LFS candidates).
- **Option B**: Host the image dataset externally (e.g., HuggingFace Datasets, S3) and provide a download script.
- **Option C**: Add `data/synthetic_ipv6_grounded_v3_32x32/images/` to `.gitignore` and document the download process.

---

## Low-Priority Issues (P3)

### 10. Vague commit messages

**Problem**: Three of five commits say the same thing:
```
03bef26 update modified files
8d56318 Update modified files
25cd674 Update modified files
```

These messages provide zero information about what changed or why.

**Fix**: Follow a conventional commit format:
```
feat: add 5-seed multiclass LR evaluation (Deliverable 1)
feat: add binary LR with PR-threshold selection (Deliverable 2)
fix: track large CSVs with Git LFS
```

### 11. No `.env.example` or local setup guide

**Problem**: There is no guidance on:
- Required Python version
- Virtual environment setup
- How to obtain the UNSW-NB15 data locally (the CNN notebook uses HuggingFace download, but this isn't documented)
- How to run the notebooks end-to-end

**Fix**: Add a `SETUP.md` or expand the README with:
```markdown
## Local Setup
1. Python >= 3.10
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Download data: `python scripts/download_data.py` (or manual instructions)
5. Update `ROOT` path in notebooks if needed
```

### 12. No `.pre-commit-config.yaml` or linting

**Problem**: No automated code quality checks. Notebook code has inconsistent formatting, and there's no guard against committing broken notebooks or outputs.

**Fix**: Add a minimal pre-commit config:
```yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
```

---

## Methodology / ML-Specific Feedback

### 13. Massive class imbalance is underdiagnosed

**Problem**: The dataset has extreme imbalance:

| Class | Total Samples | Test Samples |
|-------|--------------|--------------|
| Benign | 2829 | 425 |
| Exploits | 50 | 7 |
| Fuzzers | 44 | 6 |
| Generic | 33 | 5 |
| Reconnaissance | 16 | 3 |
| DoS | 10 | 1 |
| Shellcode | 8 | 1 |
| Backdoor | 5 | 0 |
| Analysis | 3 | 1 |
| Worms | 2 | 1 |

The notebook flags classes with <5 test samples as "unstable" but doesn't discuss the implications:
- **6 out of 10 classes** are flagged unstable
- Macro-F1 of **0.398** is low and unreliable
- The high weighted-F1 (**0.94**) masks that the model fails on most attack types
- **Backdoor has 0 test samples** — it cannot be evaluated at all

**Fix**: Add explicit analysis stating:
- These metric values are unreliable for minority classes
- The weighted-F1 vs macro-F1 gap (0.94 vs 0.40) quantifies how much Benign dominance masks poor minority performance
- Whether the project goals require minority-class detection (if yes, this baseline is insufficient)

### 14. Validation set is unused in Notebook 01 (multiclass)

**Problem**: The multiclass LR trains on `train` and evaluates directly on `test`. The `val` split is loaded but never used. There is no hyperparameter tuning or model selection step.

**Fix**: Either:
- Use the validation set for hyperparameter tuning (e.g., regularization strength `C`)
- Use it for early stopping or model selection
- Remove the `val` loading to avoid confusion

### 15. Binary threshold logic is fragile

**Problem**: In Notebook 02:
```python
y_val_true = (y_val_b=="Attack").astype(int)   # Attack = 1
scores_val = pipe.predict_proba(X_val)[:, 0]   # Column 0 = Attack
```

This works because `classes_` order happens to be `['Attack', 'Benign']`, making column 0 the Attack probability. But this is implicit and breaks silently if the class order changes (e.g., different sklearn version, different data).

**Fix**: Explicitly look up the class index:
```python
attack_idx = list(pipe.named_steps["clf"].classes_).index("Attack")
scores_val = pipe.predict_proba(X_val)[:, attack_idx]
```

### 16. No discussion of `class_weight="balanced"` effect

**Problem**: Both LR models use `class_weight="balanced"`, which reweights the loss to compensate for class imbalance. This is a reasonable default, but there is no comparison with unweighted performance or discussion of whether it helps for this specific distribution.

**Fix**: Add a brief comparison or at minimum a note explaining:
- Why `balanced` was chosen
- Whether it improved macro-F1 vs the unweighted default
- The trade-off (it may sacrifice Benign accuracy to improve minority recall)

---

## Summary Table

| Priority | # | Issue | Impact |
|----------|---|-------|--------|
| **P0** | 1 | Git LFS broken — result CSVs are empty stubs | Anyone cloning gets no data |
| **P0** | 2 | Typo `*.cvs` in `.gitattributes` | Silent misconfiguration |
| **P1** | 3 | Zero-variance 5-seed results (std = 0.0) | Undermines stability analysis credibility |
| **P1** | 4 | Missing `src/` directory referenced in README | README is misleading |
| **P1** | 5 | Unpinned and incomplete `requirements.txt` | Repo is not reproducible |
| **P2** | 6 | Hardcoded Kaggle paths | Blocks local execution |
| **P2** | 7 | Duplicate code between notebooks | Maintenance burden |
| **P2** | 8 | CNN notebook has embedded outputs (3.8MB) | Diffs are unreviewable |
| **P2** | 9 | 3,000+ PNGs committed to Git | Bloated repo history |
| **P3** | 10 | Vague commit messages | Poor audit trail |
| **P3** | 11 | No local setup guide | Onboarding friction |
| **P3** | 12 | No pre-commit hooks or linting | No quality guardrails |
| ML | 13 | Class imbalance underdiagnosed | Evaluation narrative is incomplete |
| ML | 14 | Validation set unused in Notebook 01 | Missed opportunity for tuning |
| ML | 15 | Fragile binary threshold logic | Silent breakage risk |
| ML | 16 | No `class_weight="balanced"` ablation | Missing justification |
