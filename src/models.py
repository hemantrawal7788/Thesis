"""
Multi-Scale Attention 1D CNN (MSA-CNN) for Network Intrusion Detection.

Architecture overview
---------------------
    Input  (B, D)  — vectorised tabular features
      |
      v
    Unsqueeze → (B, 1, D)
      |
      +--► Conv1d(1, C, k=3, pad=1) → BN → GELU
      +--► Conv1d(1, C, k=5, pad=2) → BN → GELU
      +--► Conv1d(1, C, k=7, pad=3) → BN → GELU
      |
      v
    Concat → (B, 3C, D)
      |
      v
    SE channel-attention → MaxPool(2) → (B, 3C, D/2)
      |
      v
    ResConv(3C → 2C) → SE → MaxPool(2) → (B, 2C, D/4)
      |
      v
    ResConv(2C → C) → GlobalAvgPool → (B, C)
      |
      v
    Dropout → Linear(C, n_classes)

Design rationale
----------------
* Multi-scale convolutions capture feature interactions at different
  window sizes — short-range (k=3) for individual-feature effects and
  wider (k=7) for cross-feature correlations in the vectorised row.
* SE attention learns which conv filters (feature detectors) are
  informative, acting as implicit feature selection.  This addresses
  the benchmark 1D CNNs treating all channels equally.
* Residual connections allow deeper stacking without degradation,
  important given the small training sets in intrusion-detection
  benchmarks.
* Global average pooling produces a fixed-length representation
  regardless of input dimension D, enabling the same architecture
  to run on UNSW-NB15 (~186 features) and the synthetic IPv6 dataset
  (~40 features) without modification.
* ~160K parameters — modest overhead vs the simple benchmark CNNs
  (~50-100K) while being far smaller than MobileNetV2 (~3.4M).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Learns per-channel scaling factors via global-pool → FC → ReLU → FC → Sigmoid.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        w = x.mean(dim=2)            # (B, C)   — squeeze
        w = self.fc(w).unsqueeze(2)   # (B, C, 1) — excitation
        return x * w                  # (B, C, L) — scale


class MultiScaleConvBlock(nn.Module):
    """Parallel Conv1D branches with different kernel sizes, concatenated."""

    def __init__(self, in_channels: int, out_per_branch: int,
                 kernel_sizes: tuple = (3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, out_per_branch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_per_branch),
                nn.GELU(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class ResConvBlock(nn.Module):
    """Conv1D → BN → GELU with a residual skip (1x1 projection if needed)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class MSACNN(nn.Module):
    """Multi-Scale Attention 1D CNN for tabular intrusion detection.

    Accepts any input dimension D (number of vectorised features).
    Works identically on UNSW-NB15 and synthetic IPv6 data.

    Args:
        n_classes:     Number of output classes (2 for binary, 10 for multiclass).
        base_channels: Channel width *C*.  Multi-scale block outputs 3C.
        kernel_sizes:  Kernel sizes for the multi-scale conv block.
        se_reduction:  Reduction ratio in SE attention blocks.
        dropout:       Dropout probability before the classifier head.
    """

    def __init__(self, n_classes: int, base_channels: int = 64,
                 kernel_sizes: tuple = (3, 5, 7), se_reduction: int = 4,
                 dropout: float = 0.4):
        super().__init__()
        n_branches = len(kernel_sizes)
        ms_out = base_channels * n_branches       # 192 default

        self.ms_block = MultiScaleConvBlock(1, base_channels, kernel_sizes)
        self.se1 = SEBlock(ms_out, se_reduction)
        self.pool1 = nn.MaxPool1d(2)

        mid = base_channels * 2                    # 128 default
        self.res1 = ResConvBlock(ms_out, mid)
        self.se2 = SEBlock(mid, se_reduction)
        self.pool2 = nn.MaxPool1d(2)

        self.res2 = ResConvBlock(mid, base_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, D) float tensor of vectorised tabular features.

        Returns:
            (B, n_classes) logits.
        """
        x = x.unsqueeze(1)           # (B, 1, D)
        x = self.ms_block(x)         # (B, 3C, D)
        x = self.se1(x)              # (B, 3C, D)
        x = self.pool1(x)            # (B, 3C, D/2)
        x = self.res1(x)             # (B, 2C, D/2)
        x = self.se2(x)              # (B, 2C, D/2)
        x = self.pool2(x)            # (B, 2C, D/4)
        x = self.res2(x)             # (B, C, D/4)
        x = self.gap(x).squeeze(2)   # (B, C)
        return self.head(x)          # (B, n_classes)
