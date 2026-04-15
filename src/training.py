"""Training loop, prediction, and metric helpers for MSA-CNN."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a DataLoader.

    Returns:
        y_true:  (N,) ground-truth integer labels.
        y_pred:  (N,) predicted integer labels (argmax).
        y_probs: (N, C) softmax probability matrix.
    """
    model.eval()
    all_true, all_pred, all_probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_true.append(y.numpy())
        all_pred.append(probs.argmax(axis=1))
        all_probs.append(probs)
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_probs),
    )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module, loader: DataLoader,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.  Returns mean batch loss."""
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(precision_recall_fscore_support(
        y_true, y_pred, average="macro",
        labels=list(range(n_classes)), zero_division=0,
    )[2])


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_classes: int,
    epochs: int = 20,
    patience: int = 5,
    save_path=None,
    prefix: str = "",
) -> tuple[dict, dict]:
    """Full training loop with early stopping on validation macro-F1.

    Args:
        model:        PyTorch model.
        train_loader: Training DataLoader (with weighted sampling).
        val_loader:   Validation DataLoader.
        criterion:    Loss function (FocalLoss or CrossEntropyLoss).
        optimizer:    Optimiser instance.
        device:       torch.device.
        n_classes:    Number of classes (for macro-F1 computation).
        epochs:       Maximum training epochs.
        patience:     Early-stopping patience (epochs without improvement).
        save_path:    Optional path to save the best checkpoint.
        prefix:       Logging prefix string.

    Returns:
        history:    dict with lists ``train_loss``, ``val_acc``, ``val_macro_f1``.
        best_state: ``state_dict`` of the model at the best validation epoch.
    """
    history = {"train_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, _ = predict(model, val_loader, device)
        val_acc = float(accuracy_score(y_true, y_pred))
        val_f1 = _macro_f1(y_true, y_pred, n_classes)

        history["train_loss"].append(loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        tag = f"{prefix} | " if prefix else ""
        print(f"  {tag}epoch {epoch:02d} | loss={loss:.4f}"
              f" | val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            if save_path is not None:
                torch.save(best_state, save_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    return history, best_state
