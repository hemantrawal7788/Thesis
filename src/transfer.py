from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def feature_overlap(unsw_cols, synthetic_cols):
    unsw_cols = list(unsw_cols)
    synthetic_cols = list(synthetic_cols)
    return {
        'unsw_feature_count': len(unsw_cols),
        'synthetic_feature_count': len(synthetic_cols),
        'shared_feature_count': len(set(unsw_cols).intersection(set(synthetic_cols))),
        'shared_features': sorted(set(unsw_cols).intersection(set(synthetic_cols))),
        'unsw_only': sorted(set(unsw_cols) - set(synthetic_cols)),
        'synthetic_only': sorted(set(synthetic_cols) - set(unsw_cols)),
    }

def pca_projection(X_unsw, X_syn, n_components=2):
    common = [c for c in X_unsw.columns if c in X_syn.columns]
    if not common:
        raise ValueError('No shared columns between UNSW and synthetic frames.')
    Xu = X_unsw[common].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    Xs = X_syn[common].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    pca = PCA(n_components=n_components, random_state=42)
    merged = np.vstack([Xu, Xs])
    coords = pca.fit_transform(merged)
    domain = np.array(['UNSW'] * len(Xu) + ['Synthetic'] * len(Xs))
    return coords, domain, pca.explained_variance_ratio_.tolist(), common
