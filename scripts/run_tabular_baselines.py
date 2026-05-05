#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loading import prepare_unsw_task, prepare_synthetic_task
from src.preprocessing import build_preprocessors
from src.tabular_baselines import build_tabular_models
from src.metrics import multiclass_metrics, binary_metrics, support_flag_table
from src.io_utils import save_json, save_text
from src.paths import ensure_dir

def run_block(dataset: str, task: str, outdir: str, seeds: list[int]):
    X_pool, y_pool, X_test, y_test, labels = (prepare_unsw_task(task) if dataset == 'unsw' else prepare_synthetic_task(task))
    preprocess_linear, _, preprocess_tree, _, _ = build_preprocessors(X_pool)
    outdir = ensure_dir(outdir)
    support_flag_table(y_test, threshold=5).to_csv(outdir / 'test_support_flags.csv', index=False)

    rows = []
    for seed in seeds:
        X_train, X_val, y_train, y_val = train_test_split(X_pool, y_pool, test_size=0.2, stratify=y_pool, random_state=seed)
        models = build_tabular_models(seed, preprocess_linear, preprocess_tree, task)
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            if task == 'binary':
                score = None
                clf = pipe.named_steps['clf']
                if hasattr(clf, 'predict_proba'):
                    attack_idx = list(clf.classes_).index('Attack') if hasattr(clf, 'classes_') and 'Attack' in list(clf.classes_) else 1
                    score = pipe.predict_proba(X_test)[:, attack_idx]
                m = binary_metrics(y_test, y_pred, score)
                rows.append({'seed': seed, 'model': name, 'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'], 'auc': m['auc']})
            else:
                prob = None
                clf = pipe.named_steps['clf']
                if hasattr(clf, 'predict_proba'):
                    prob = pipe.predict_proba(X_test)
                m = multiclass_metrics(y_test, y_pred, labels, prob)
                rows.append({'seed': seed, 'model': name, 'accuracy': m['accuracy'], 'macro_precision': m['macro_precision'], 'macro_recall': m['macro_recall'], 'macro_f1': m['macro_f1'], 'weighted_f1': m['weighted_f1'], 'ovr_auc': m['ovr_auc']})
    runs = pd.DataFrame(rows)
    runs.to_csv(outdir / 'all_runs.csv', index=False)
    if task == 'binary':
        summary = runs.groupby('model').agg(accuracy_mean=('accuracy','mean'), accuracy_std=('accuracy','std'), precision_mean=('precision','mean'), precision_std=('precision','std'), recall_mean=('recall','mean'), recall_std=('recall','std'), f1_mean=('f1','mean'), f1_std=('f1','std'), auc_mean=('auc','mean')).reset_index().sort_values('f1_mean', ascending=False)
    else:
        summary = runs.groupby('model').agg(accuracy_mean=('accuracy','mean'), accuracy_std=('accuracy','std'), macro_f1_mean=('macro_f1','mean'), macro_f1_std=('macro_f1','std'), macro_recall_mean=('macro_recall','mean'), macro_recall_std=('macro_recall','std'), weighted_f1_mean=('weighted_f1','mean'), weighted_f1_std=('weighted_f1','std')).reset_index().sort_values('macro_f1_mean', ascending=False)
    summary.to_csv(outdir / 'summary_mean_std.csv', index=False)
    print(summary)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['unsw','synthetic'], required=True)
    ap.add_argument('--task', choices=['binary','multiclass'], required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seeds', nargs='*', type=int, default=[11,22,33,44,55])
    args = ap.parse_args()
    run_block(args.dataset, args.task, args.outdir, args.seeds)

if __name__ == '__main__':
    main()
