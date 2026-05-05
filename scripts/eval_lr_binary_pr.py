#!/usr/bin/env python
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from src.data_loading import prepare_synthetic_task, prepare_unsw_task
from src.preprocessing import build_preprocessors
from src.metrics import binary_metrics
from src.io_utils import save_json, save_text
from src.paths import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['synthetic','unsw'], default='synthetic')
    ap.add_argument('--outdir', default='artifacts/lr_binary')
    args = ap.parse_args()

    X_pool, y_pool, X_test, y_test, labels = (prepare_synthetic_task('binary') if args.dataset == 'synthetic' else prepare_unsw_task('binary'))
    X_train, X_val, y_train, y_val = train_test_split(X_pool, y_pool, test_size=0.2, stratify=y_pool, random_state=42)
    preprocess_linear, _, _, _, _ = build_preprocessors(X_pool)

    pipe = Pipeline([
        ('prep', preprocess_linear),
        ('clf', LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)

    attack_idx = list(pipe.named_steps['clf'].classes_).index('Attack')
    val_scores = pipe.predict_proba(X_val)[:, attack_idx]
    prec, rec, thr = precision_recall_curve((pd.Series(y_val)=='Attack').astype(int), val_scores)
    f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-8, None)
    best_idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    threshold = float(thr[best_idx]) if len(thr) else 0.5

    test_scores = pipe.predict_proba(X_test)[:, attack_idx]
    y_pred = np.where(test_scores >= threshold, 'Attack', 'Benign')
    m = binary_metrics(y_test, y_pred, test_scores)

    outdir = ensure_dir(args.outdir)
    save_json({'threshold': threshold, **{k:v for k,v in m.items() if k not in ['confusion_matrix','report']}}, outdir / 'threshold_metrics.json')
    save_text(m['report'], outdir / 'classification_report.txt')
    pd.DataFrame(m['confusion_matrix'], index=['Benign','Attack'], columns=['Benign','Attack']).to_csv(outdir / 'confusion_matrix.csv')

    plt.figure(figsize=(7,4))
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Validation PR curve')
    plt.tight_layout()
    plt.savefig(outdir / 'pr_curve_val.png', dpi=180)
    plt.close()
    print(m)

if __name__ == '__main__':
    main()
