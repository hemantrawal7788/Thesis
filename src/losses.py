"""
Focal Loss for class-imbalanced intrusion detection.

Standard cross-entropy treats every sample equally, so the loss is
dominated by the abundant Benign class.  Focal loss (Lin et al., ICCV
2017) adds a modulating factor (1 - p_t)^gamma that downweights
well-classified (easy) examples and focuses training on hard,
misclassified samples.

Combined with per-class alpha weights this gives:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Typical settings:
    gamma=0  →  reduces to weighted cross-entropy
    gamma=2  →  recommended default from the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with optional per-class weighting.

    Args:
        alpha:     1-D tensor of per-class weights, or ``None`` for uniform.
                   Typically set to inverse class frequency.
        gamma:     Focusing parameter.  Higher values downweight easy examples
                   more aggressively.  ``gamma=0`` recovers standard CE.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(self, alpha: torch.Tensor | None = None,
                 gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw model outputs.
            targets: (B,) integer class labels.

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or (B,) per-sample loss.
        """
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)                          # p_t for true class
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            focal_weight = self.alpha[targets] * focal_weight

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
