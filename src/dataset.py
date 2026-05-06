"""Dataset and DataLoader utilities for vectorised tabular data."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class TabularDataset(Dataset):
    """Simple PyTorch dataset wrapping NumPy arrays of features and labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_weighted_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 256
) -> tuple[DataLoader, np.ndarray]:
    """DataLoader with inverse-frequency weighted sampling for training.

    Returns:
        loader:        DataLoader with WeightedRandomSampler.
        class_weights: 1-D array of per-class inverse-frequency weights.
    """
    n_classes = int(y.max()) + 1
    class_counts = np.bincount(y, minlength=n_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    sample_weights = class_weights[y]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(
        TabularDataset(X, y),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )
    return loader, class_weights


def make_eval_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 512
) -> DataLoader:
    """DataLoader for evaluation — no sampling, no shuffle."""
    return DataLoader(
        TabularDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
