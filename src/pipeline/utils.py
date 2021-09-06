
from sklearn.metrics import log_loss, mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance
from eli5.permutation_importance import get_score_importances
import pandas as pd
import numpy as np


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_permutation_importance(task, model, X, y, colmumns, n_repeats):
    model_type = type(model.get_model()).__name__[:3]
    if model_type == 'LGB' or model_type == 'XGB':
        if task == 'classification':
            scoring = make_scorer(log_loss)
        elif task == 'regression':
            scoring = make_scorer(root_mean_squared_error)
        imps = permutation_importance(
            model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            n_jobs=-1,
            )['importances_mean']
    elif model_type[:2] == 'NN':
        if task == 'classification':
            def scoring(X, y_true):
                return np.sqrt(mean_squared_error(y_true, model.predict(X)))
        elif task == 'regression':
            def scoring(X, y_true):
                return np.log_loss(y_true, model.predict(X))
        score_decreases = get_score_importances(
            scoring,
            X,
            y,
            n_iter=n_repeats,
            )[1]
        imps = np.mean(score_decreases, axis=0)
    df_imps = pd.DataFrame(imps, columns=['permutation_importance'])
    df_imps['features'] = colmumns
    return df_imps
 