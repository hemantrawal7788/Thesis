from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def build_tabular_models(seed: int, preprocess_linear, preprocess_tree, task: str):
    models = {
        'lr': Pipeline([
            ('prep', preprocess_linear),
            ('clf', LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', multi_class='auto', random_state=seed))
        ]),
        'linearsvc': Pipeline([
            ('prep', preprocess_linear),
            ('clf', LinearSVC(class_weight='balanced', random_state=seed))
        ]),
        'rf': Pipeline([
            ('prep', preprocess_tree),
            ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', n_jobs=-1, random_state=seed))
        ]),
        'histgb': Pipeline([
            ('prep', preprocess_tree),
            ('clf', HistGradientBoostingClassifier(random_state=seed))
        ]),
        'mlp': Pipeline([
            ('prep', preprocess_linear),
            ('clf', MLPClassifier(hidden_layer_sizes=(256,128), activation='relu', early_stopping=True, max_iter=150, random_state=seed))
        ]),
    }
    if HAS_XGB:
        if task == 'binary':
            clf = XGBClassifier(
                objective='binary:logistic', eval_metric='logloss',
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, tree_method='hist'
            )
        else:
            clf = XGBClassifier(
                objective='multi:softprob', eval_metric='mlogloss',
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, tree_method='hist'
            )
        models['xgboost'] = Pipeline([('prep', preprocess_tree), ('clf', clf)])
    return models
