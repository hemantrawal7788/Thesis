# Thesis IDS Repository - UNSW-NB15 and Synthetic IPv6

This repository contains the implementation notebooks and supporting Python modules for intrusion detection experiments on UNSW-NB15 and a grounded synthetic IPv6 dataset. The work covers tabular baselines, benchmark CNN models, a custom MSA-CNN model, final heat-map computer-vision models, and transfer diagnostics between UNSW-NB15 and the synthetic dataset.

The repository is code-focused. Large datasets and generated experiment artifacts are expected to be supplied locally or through Kaggle input paths.

## Repository Layout

```text
.
|-- README.md
|-- notebooks/
|   |-- 01_LR_Multiclass_5Seed_Eval.ipynb
|   |-- 02_LR_Binary_PR_Threshold.ipynb
|   |-- 03_Benchmark_Models_UNSW_and_Synthetic_Fixed_Pipeline.ipynb
|   |-- 04_Paper_Based_Benchmark_CNNs_UNSW_then_Synthetic.ipynb
|   |-- 05_Custom_CNN_MSA.ipynb
|   |-- 06_Final_Models_Heatmap_CV_UNSW_to_Synthetic.ipynb
|   `-- 07_Transfer_Diagnostics_and_Unified_Comparison.ipynb
|-- scripts/
|   |-- build_synthetic_ipv6_v3.py
|   |-- eval_lr_binary_pr.py
|   |-- eval_lr_multiclass_5seed.py
|   |-- run_tabular_baselines.py
|   `-- run_transfer_diagnostics.py
`-- src/
    |-- __init__.py
    |-- data_loading.py
    |-- generate_synthetic_ipv6_grounded_v2.py
    |-- generate_synthetic_ipv6_grounded_v3_32x32.py
    |-- io_utils.py
    |-- losses.py
    |-- metrics.py
    |-- models.py
    |-- paths.py
    |-- preprocessing.py
    |-- tabular_baselines.py
    |-- training.py
    `-- transfer.py
```

## Data Inputs

Expected UNSW-NB15 files:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Expected synthetic IPv6 bundle:

- `synthetic_ipv6_grounded_v3_32x32.zip`

The path resolver in `src/paths.py` searches:

- environment variables: `UNSW_TRAIN_CSV`, `UNSW_TEST_CSV`, `SYN_ZIP_PATH`, `DATA_ROOT`
- `/kaggle/input/**`
- the current working directory
- `/mnt/data`

## Notebook Workflow

Run the notebooks in this order for the full thesis experiment sequence:

1. `01_LR_Multiclass_5Seed_Eval.ipynb`
2. `02_LR_Binary_PR_Threshold.ipynb`
3. `03_Benchmark_Models_UNSW_and_Synthetic_Fixed_Pipeline.ipynb`
4. `04_Paper_Based_Benchmark_CNNs_UNSW_then_Synthetic.ipynb`
5. `05_Custom_CNN_MSA.ipynb`
6. `06_Final_Models_Heatmap_CV_UNSW_to_Synthetic.ipynb`
7. `07_Transfer_Diagnostics_and_Unified_Comparison.ipynb`

## Script Entry Points

```bash
python scripts/eval_lr_multiclass_5seed.py --dataset synthetic --outdir artifacts/lr_multiclass
python scripts/eval_lr_binary_pr.py --dataset synthetic --outdir artifacts/lr_binary
python scripts/run_tabular_baselines.py --dataset unsw --task multiclass --outdir artifacts/unsw_multiclass
python scripts/run_tabular_baselines.py --dataset synthetic --task multiclass --outdir artifacts/synthetic_multiclass
python scripts/run_transfer_diagnostics.py --task multiclass --outdir artifacts/transfer_multiclass
```

The source files `src/generate_synthetic_ipv6_grounded_v2.py` and `src/generate_synthetic_ipv6_grounded_v3_32x32.py` contain standalone synthetic dataset generation logic. The wrapper `scripts/build_synthetic_ipv6_v3.py` is present in this repository, but it currently imports `src.synthetic_ipv6_generator`, which is not present in this folder.

## Python Packages

The notebooks and scripts use the following main packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `torch`
- `torchvision`
- `Pillow`
- `huggingface_hub`
- `xgboost`
- `shap`
- `umap-learn`

Install these in the active notebook or script environment before running the full pipeline.

## Current Saved Execution Status



| Notebook | Saved status |
| --- | --- |
| `01_LR_Multiclass_5Seed_Eval.ipynb` | Executed cells with saved outputs; no error outputs. |
| `02_LR_Binary_PR_Threshold.ipynb` | Executed cells with saved outputs; no error outputs. |
| `03_Benchmark_Models_UNSW_and_Synthetic_Fixed_Pipeline.ipynb` | Executed cells with saved outputs. One earlier split error output is still embedded, followed by executed recovery cells for the synthetic benchmark section. |
| `04_Paper_Based_Benchmark_CNNs_UNSW_then_Synthetic.ipynb` | Contains saved benchmark outputs. Several later cells have saved source but no execution count in the notebook. |
| `05_Custom_CNN_MSA.ipynb` | Executed cells with saved outputs; no error outputs. |
| `06_Final_Models_Heatmap_CV_UNSW_to_Synthetic.ipynb` | Not executed in the saved notebook; no saved outputs. |
| `07_Transfer_Diagnostics_and_Unified_Comparison.ipynb` | Not executed in the saved notebook; no saved outputs. |

## Output Locations

Most notebooks and scripts write generated files under local artifact directories such as:

- `artifacts/`
- `benchmark_artifacts/`
- `cnn_benchmark_artifacts/`
- `msa_cnn_artifacts/`
- `final_model_artifacts/`
- `transfer_diagnostics/`

