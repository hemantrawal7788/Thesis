from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessors(X_pool):
    cat_cols = [c for c in X_pool.columns if X_pool[c].dtype == 'object']
    num_cols = [c for c in X_pool.columns if c not in cat_cols]

    preprocess_linear = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler(with_mean=False))]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
        ],
        remainder='drop',
    )
    preprocess_dense = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler(with_mean=True))]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))]), cat_cols),
        ],
        remainder='drop',
    )
    preprocess_tree = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num_cols),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols),
        ],
        remainder='drop',
    )
    return preprocess_linear, preprocess_dense, preprocess_tree, num_cols, cat_cols
