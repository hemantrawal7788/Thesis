"""Path discovery helpers for local runs and Kaggle.

These helpers remove hard-coded absolute paths and let the same code run:
- on Kaggle (`/kaggle/input/...`)
- locally from a cloned repository
- inside a notebook or script with environment-variable overrides
"""

from __future__ import annotations
import glob
import os
from pathlib import Path

KAGGLE_INPUT = Path('/kaggle/input')

def find_file(patterns: list[str]) -> str | None:
    roots = []
    data_root = os.environ.get('DATA_ROOT')
    if data_root:
        roots.append(Path(data_root))
    if KAGGLE_INPUT.exists():
        roots.append(KAGGLE_INPUT)
    roots.extend([Path.cwd(), Path('/mnt/data')])

    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for pat in patterns:
            hits = glob.glob(str(root / '**' / pat), recursive=True)
            if hits:
                return hits[0]
    return None

def resolve_unsw_paths() -> tuple[str, str]:
    train_csv = os.environ.get('UNSW_TRAIN_CSV') or find_file(['UNSW_NB15_training-set.csv'])
    test_csv = os.environ.get('UNSW_TEST_CSV') or find_file(['UNSW_NB15_testing-set.csv'])
    if train_csv is None or test_csv is None:
        raise FileNotFoundError('UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv not found. Set UNSW_TRAIN_CSV and UNSW_TEST_CSV or place them in Kaggle input / data root.')
    return train_csv, test_csv

def resolve_synthetic_zip() -> str:
    syn = os.environ.get('SYN_ZIP_PATH') or find_file(['synthetic_ipv6_grounded_v3_32x32.zip'])
    if syn is None:
        raise FileNotFoundError('synthetic_ipv6_grounded_v3_32x32.zip not found. Set SYN_ZIP_PATH or place it in Kaggle input / data root.')
    return syn

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
