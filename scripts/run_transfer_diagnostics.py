#!/usr/bin/env python
from __future__ import annotations
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loading import prepare_unsw_task, prepare_synthetic_task
from src.transfer import feature_overlap, pca_projection
from src.io_utils import save_json
from src.paths import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['binary','multiclass'], default='multiclass')
    ap.add_argument('--outdir', default='artifacts/transfer_diagnostics')
    args = ap.parse_args()

    X_unsw, y_unsw, _, _, _ = prepare_unsw_task(args.task)
    _, _, X_syn, y_syn, _ = prepare_synthetic_task(args.task)
    outdir = ensure_dir(args.outdir)

    overlap = feature_overlap(X_unsw.columns, X_syn.columns)
    save_json(overlap, outdir / 'feature_overlap.json')

    coords, domain, var_ratio, common = pca_projection(X_unsw, X_syn)
    df = pd.DataFrame({'pc1': coords[:,0], 'pc2': coords[:,1], 'domain': domain})
    df.to_csv(outdir / 'pca_projection.csv', index=False)

    plt.figure(figsize=(6,5))
    for d in sorted(df['domain'].unique()):
        sub = df[df['domain']==d]
        plt.scatter(sub['pc1'], sub['pc2'], s=6, alpha=0.5, label=d)
    plt.xlabel(f'PC1 ({var_ratio[0]:.2%})')
    plt.ylabel(f'PC2 ({var_ratio[1]:.2%})')
    plt.title('UNSW vs Synthetic shared-feature PCA')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'pca_projection.png', dpi=180)
    print('Saved diagnostics to', outdir)

if __name__ == '__main__':
    main()
