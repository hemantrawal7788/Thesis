from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score

def support_flag_table(labels, threshold: int = 5) -> pd.DataFrame:
    vc = pd.Series(labels).value_counts().sort_values(ascending=True)
    out = vc.rename_axis('label').reset_index(name='support')
    out['unstable'] = out['support'] < threshold
    return out

def multiclass_metrics(y_true, y_pred, labels, y_prob=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    out = {
        'accuracy': float(acc),
        'macro_precision': float(p_macro),
        'macro_recall': float(r_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(p_weighted),
        'weighted_recall': float(r_weighted),
        'weighted_f1': float(f1_weighted),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'report': classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0),
    }
    if y_prob is not None:
        try:
            y_idx = pd.Series(y_true).map({l:i for i,l in enumerate(labels)}).values
            out['ovr_auc'] = float(roc_auc_score(y_idx, y_prob, multi_class='ovr', average='macro'))
        except Exception:
            out['ovr_auc'] = None
    return out

def binary_metrics(y_true, y_pred, y_score=None, pos_label='Attack') -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    out = {
        'accuracy': float(acc),
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=['Benign','Attack']).tolist(),
        'report': classification_report(y_true, y_pred, labels=['Benign','Attack'], digits=4, zero_division=0),
    }
    if y_score is not None:
        try:
            out['auc'] = float(roc_auc_score((pd.Series(y_true)==pos_label).astype(int).values, y_score))
        except Exception:
            out['auc'] = None
    return out
