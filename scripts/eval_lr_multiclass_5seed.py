#!/usr/bin/env python
"""Synthetic multiclass Logistic Regression with bootstrap-style 5-seed evaluation.

This script addresses the review note that the earlier 5-seed setup had zero variance
because it only changed solver initialization. Here, each seed bootstraps the UNSW or
synthetic training pool before fitting.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.data_loading import prepare_synthetic_task, prepare_unsw_task
from src.preprocessing import build_preprocessors
from src.metrics import multiclass_metrics, support_flag_table
from src.io_utils import save_json, save_text
from src.paths import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['synthetic','unsw'], default='synthetic')
    ap.add_argument('--outdir', default='artifacts/lr_multiclass')
    ap.add_argument('--seeds', nargs='*', type=int, default=[11,22,33,44,55])
    args = ap.parse_args()

    X_pool, y_pool, X_test, y_test, labels = (prepare_synthetic_task('multiclass') if args.dataset == 'synthetic' else prepare_unsw_task('multiclass'))
    preprocess_linear, _, _, _, _ = build_preprocessors(X_pool)
    outdir = ensure_dir(args.outdir)

    rows = []
    for i, seed in enumerate(args.seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_pool), size=len(X_pool), replace=True)
        X_boot = X_pool.iloc[idx].reset_index(drop=True)
        y_boot = pd.Series(y_pool)[idx].reset_index(drop=True).values

        pipe = Pipeline([
            ('prep', preprocess_linear),
            ('clf', LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', multi_class='auto', random_state=seed))
        ])
        pipe.fit(X_boot, y_boot)
        y_pred = pipe.predict(X_test)
        m = multiclass_metrics(y_test, y_pred, labels)
        rows.append({'seed': seed, 'accuracy': m['accuracy'], 'macro_precision': m['macro_precision'], 'macro_recall': m['macro_recall'], 'macro_f1': m['macro_f1'], 'weighted_f1': m['weighted_f1']})
        if i == 0:
            pd.DataFrame(m['confusion_matrix'], index=labels, columns=labels).to_csv(outdir / 'confusion_matrix_seed11.csv')
            save_text(m['report'], outdir / 'classification_report_seed11.txt')

    runs = pd.DataFrame(rows)
    summary = runs.agg({'accuracy':['mean','std'], 'macro_precision':['mean','std'], 'macro_recall':['mean','std'], 'macro_f1':['mean','std'], 'weighted_f1':['mean','std']}).to_dict()
    runs.to_csv(outdir / 'all_runs.csv', index=False)
    save_json(summary, outdir / 'summary.json')
    support_flag_table(y_test, threshold=5).to_csv(outdir / 'test_support_flags.csv', index=False)
    print(runs)
    print(summary)

if __name__ == '__main__':
    main()
