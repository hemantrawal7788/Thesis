# Weekly Progress — Evaluation Credibility (Deliverables 1 & 2)

This repo snapshot contains the work requested in the weekly feedback:
1) **Deliverable 1:** Multiclass Logistic Regression evaluation wrapped in a **5-seed loop** with **Macro-F1 mean ± std** and a **per-class support check** that flags classes with <5 test samples as *unstable*.
2) **Deliverable 2:** Binary track (**Benign vs Attack**) with Logistic Regression, **PR curve**, **threshold selection on validation**, and test-set precision/recall at the chosen threshold.

## Structure
- `notebooks/`
  - `01_LR_Multiclass_5Seed_Eval.ipynb`
  - `02_LR_Binary_PR_Threshold.ipynb`
- `src/` (optional scripts)
  - `eval_lr_multiclass_5seed.py`
  - `eval_lr_binary_pr.py`
- `results/` (generated outputs)
  - 5-seed run table + summary JSON
  - support/unstable flags CSV
  - PR curve plots + threshold metrics JSON
  - confusion matrices + classification reports

## How to run
1. Place `synthetic_ipv6_grounded_v3_32x32.zip` in the repo root (or update the path in the notebooks).
2. Run Notebook 1, then Notebook 2.
3. Commit `notebooks/`, `src/`, and `results/`.

> Note: The dataset is small enough for standard Git. If you later use larger datasets, use Git LFS or store data externally.
