"""
MSA-CNN: Multi-Scale Attention 1D CNN for Network Intrusion Detection.

Custom CNN architecture designed for the thesis, to be compared against
literature benchmark CNNs (Systems2024 Arch1/2, Noever2021 MobileNetV2)
on both UNSW-NB15 and synthetic IPv6 transfer evaluation.
"""

from .models import MSACNN, SEBlock, MultiScaleConvBlock, ResConvBlock
from .losses import FocalLoss
from .dataset import TabularDataset, make_weighted_loader, make_eval_loader
from .training import predict, train_one_epoch, train_model

__all__ = [
    "MSACNN", "SEBlock", "MultiScaleConvBlock", "ResConvBlock",
    "FocalLoss",
    "TabularDataset", "make_weighted_loader", "make_eval_loader",
    "predict", "train_one_epoch", "train_model",
]
