
from sklearn.inspection import permutation_importance
from eli5.permutation_importance import get_score_importances
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer
)
import pandas as pd
import numpy as np
import configparser
import pickle


def roc_cutoff(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    cutoff = thresholds[np.argmax(tpr - fpr)]
    return cutoff


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_clf(y_true, y_pred, cutoff):
    pred_label = (y_pred >= cutoff) * 1
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label).ravel()
    accuracy = accuracy_score(y_true, pred_label)
    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
    mcc = matthews_corrcoef(y_true, pred_label)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_pred)
    metrics = {
        'auc': [auc],
        'acc': [accuracy],
        'sen': [sensitivity],
        'spe': [specificity],
        'bac': [balanced_accuracy],
        'mcc': [mcc],
        'cutoff': [cutoff]
    }
    return metrics


def evaluate_reg(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics = {'r2': [r2], 'mae': [mae], 'rmse': [rmse], 'mse': [mse]}
    return metrics


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


def set_params(trainer, config_path, best_params_path):
    model_name = type(trainer.get_model()).__name__
    config = configparser.ConfigParser()
    config.read(config_path)
    params = {}
    if 'LGB' in model_name:
        section = 'lgb_params'
        params['learning_rate'] = float(config.get(section, 'learning_rate'))
        params['n_estimators'] = int(config.get(section, 'n_estimators'))
        params['max_depth'] = int(config.get(section, 'max_depth'))
        params['num_leaves'] = int(config.get(section, 'num_leaves'))
        params['subsample'] = float(config.get(section, 'subsample'))
        params['colsample_bytree'] = float(config.get(section, 'colsample_bytree'))
        params['bagging_freq'] = int(config.get(section, 'bagging_freq'))
        params['min_child_weight'] = float(config.get(section, 'min_child_weight'))
        params['min_child_samples'] = int(config.get(section, 'min_child_samples'))
        params['min_split_gain'] = float(config.get(section, 'min_split_gain'))
        params['n_jobs'] = int(config.get(section, 'n_jobs'))
    elif 'XGB' in model_name:
        section = 'xgb_params'
        params['learning_rate'] = float(config.get(section, 'learning_rate'))
        params['n_estimators'] = int(config.get(section, 'n_estimators'))
        params['max_depth'] = int(config.get(section, 'max_depth'))
        params['subsample'] = float(config.get(section, 'subsample'))
        params['colsample_bytree'] = float(config.get(section, 'colsample_bytree'))
        params['gamma'] = float(config.get(section, 'gamma'))
        params['min_child_weight'] = float(config.get(section, 'min_child_weight'))
        params['n_jobs'] = int(config.get(section, 'n_jobs'))
    elif 'NN' in model_name:
        section = 'nn_params'
        params['standardization'] = config.getboolean(section, 'standardization')
        params['learning_rate'] = float(config.get(section, 'learning_rate'))
        params['epochs'] = int(config.get(section, 'epochs'))
        params['hidden_units'] = int(config.get(section, 'hidden_units'))
        params['batch_size'] = int(config.get(section, 'batch_size'))
        params['input_dropout'] = float(config.get(section, 'input_dropout'))
        params['hidden_dropout'] = float(config.get(section, 'hidden_dropout'))
        params['hidden_layers'] = int(config.get(section, 'hidden_layers'))
        params['batch_norm'] = config.get(section, 'batch_norm')
    if best_params_path is not None:
        model_name_from_bestparams = best_params_path.split('/')[-1].split('_')[0]
        with open(best_params_path, 'rb') as f:
            best_params = pickle.load(f)
        if 'LGB' in model_name:
            if model_name == model_name_from_bestparams:
                params.update(best_params)
                params['min_child_samples'] = int(best_params['min_child_samples'])
            else:
                return 'Erorr!! Model name and best parameters do not match.'
        elif 'XGB' in model_name:
            if model_name == model_name_from_bestparams:
                params.update(best_params)
            else:
                return 'Erorr!! Model name and best parameters do not match.'
        elif 'NN' in model_name:
            if model_name == model_name_from_bestparams:
                params.update(best_params)
                params['hidden_units'] = int(best_params['hidden_units'])
                params['hidden_units'] = int(best_params['hidden_units'])
            else:
                return 'Erorr!! Model name and best parameters do not match.'
    return params
