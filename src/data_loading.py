from __future__ import annotations
import zipfile
import pandas as pd
from pathlib import Path
from .paths import resolve_unsw_paths, resolve_synthetic_zip

def normalize_attack_name(x):
    if pd.isna(x):
        return 'Unknown'
    s = str(x).strip()
    if s.lower() == 'normal':
        return 'Benign'
    return s

def load_unsw(use_hf: bool = True):
    train_csv, test_csv = resolve_unsw_paths()
    unsw_train = pd.read_csv(train_csv)
    unsw_test = pd.read_csv(test_csv)
    return unsw_train, unsw_test

def load_synthetic():
    syn_zip = resolve_synthetic_zip()
    with zipfile.ZipFile(syn_zip, 'r') as zf:
        names = zf.namelist()
        syn_root = sorted(set([n.split('/')[0] for n in names if '/' in n]))[0]
        syn_flows = pd.read_csv(zf.open(f'{syn_root}/flows.csv'))
        syn_train_split = pd.read_csv(zf.open(f'{syn_root}/train.csv'))
        syn_val_split = pd.read_csv(zf.open(f'{syn_root}/val.csv'))
        syn_test_split = pd.read_csv(zf.open(f'{syn_root}/test.csv'))
    syn_train = syn_flows.merge(syn_train_split[['record_id']], on='record_id', how='inner')
    syn_val = syn_flows.merge(syn_val_split[['record_id']], on='record_id', how='inner')
    syn_test = syn_flows.merge(syn_test_split[['record_id']], on='record_id', how='inner')
    return syn_flows, syn_train, syn_val, syn_test

def prepare_unsw_task(task: str):
    tr, te = load_unsw()
    if task == 'binary':
        target = 'label'
        tr[target] = tr[target].astype(str).map(lambda x: 'Attack' if str(x) == '1' else 'Benign')
        te[target] = te[target].astype(str).map(lambda x: 'Attack' if str(x) == '1' else 'Benign')
        labels = ['Benign','Attack']
        drop_cols = ['id','attack_cat']
    else:
        target = 'attack_cat'
        tr[target] = tr[target].apply(normalize_attack_name)
        te[target] = te[target].apply(normalize_attack_name)
        labels = sorted(pd.Series(tr[target]).unique().tolist())
        drop_cols = ['id','label']
    X_pool = tr.drop(columns=[target] + drop_cols, errors='ignore').reset_index(drop=True)
    y_pool = tr[target].astype(str).values
    X_test = te.drop(columns=[target] + drop_cols, errors='ignore').reset_index(drop=True)
    y_test = te[target].astype(str).values
    return X_pool, y_pool, X_test, y_test, labels

def prepare_synthetic_task(task: str):
    _, syn_train, syn_val, syn_test = load_synthetic()
    df_pool = pd.concat([syn_train, syn_val], axis=0).reset_index(drop=True)
    df_test = syn_test.copy()
    target = 'label'
    if task == 'binary':
        df_pool[target] = df_pool[target].astype(str).map(lambda x: 'Benign' if x == 'Benign' else 'Attack')
        df_test[target] = df_test[target].astype(str).map(lambda x: 'Benign' if x == 'Benign' else 'Attack')
        labels = ['Benign','Attack']
    else:
        df_pool[target] = df_pool[target].astype(str).apply(lambda s: 'Benign' if str(s).lower() == 'normal' else str(s))
        df_test[target] = df_test[target].astype(str).apply(lambda s: 'Benign' if str(s).lower() == 'normal' else str(s))
        labels = sorted(pd.Series(df_pool[target]).unique().tolist())
    drop_cols = ['record_id','window_start_utc','window_end_utc','src_ip','dst_ip']
    X_pool = df_pool.drop(columns=[target] + drop_cols, errors='ignore').reset_index(drop=True)
    y_pool = df_pool[target].astype(str).values
    X_test = df_test.drop(columns=[target] + drop_cols, errors='ignore').reset_index(drop=True)
    y_test = df_test[target].astype(str).values
    return X_pool, y_pool, X_test, y_test, labels
