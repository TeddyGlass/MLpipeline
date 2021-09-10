import pandas as pd
import numpy as np
import pickle
import os
import configparser
import argparse
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from keras.models import model_from_json
from utils import evaluate_clf, evaluate_reg, roc_cutoff


def load_lgb(path):
    if 'LGB' in path:
        with open(path, 'rb') as f:
            model = pickle.load(f)
    return model


def load_xgb(path):
    if 'XGB' in path:
        with open(path, 'rb') as f:
            model = pickle.load(f)
    return model


def load_nn(path_json, path_h5, path_transformer):
    if 'NN' in path_json:
        with open(path_json, 'r') as f:
            json_string = f.read()
        model = model_from_json(json_string)
        model.load_weights(path_h5)
        if os.path.exists(path_transformer):
            with open(path_transformer, 'rb') as f:
                transformer = pickle.load(f)
        else:
            transformer = None
        return model, transformer


def sort_preds(PREDS, IDXES):
    va_idxes = np.concatenate(IDXES)
    order = np.argsort(va_idxes)
    va_preds = np.concatenate(PREDS)
    return va_preds[order]


def gen_stk_features(models_dict, models_dir, task, X_train, y_train,
                     X_test, n_splits, random_state):
    # settings of cross validation
    if task == 'clf':
        cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    elif task == 'reg':
        cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    # generate prediction features
    DF_PRED_VA = []
    DF_PRED_TE = []
    for model_name, use_this_model in models_dict.items():
        if use_this_model:
            VA_IDXES = []
            VA_PREDS = []
            TE_PREDS = []
            # cross validation
            for i, (_, va_idx) in enumerate(cv.split(X_train, y_train)):
                if model_name == 'LGB':
                    if task == 'clf':
                        path_model = os.path.join(models_dir, f'LGBMClassifier_{i}_trainedmodel.pkl')
                        model = load_lgb(path_model)
                        y_pred_va = model.predict_proba(
                            X_train[va_idx],
                            num_iterations=model.best_iteration_
                            )[:, 1]
                        y_pred_te = model.predict_proba(
                            X_test,
                            num_iterations=model.best_iteration_
                            )[:, 1]
                    elif task == 'reg':
                        path_model = os.path.join(models_dir, f'LGBMRegressor_{i}_trainedmodel.pkl')
                        model = load_lgb(path_model)
                        y_pred_va = model.predict(
                            X_train[va_idx],
                            num_iterations=model.best_iteration_
                            )
                        y_pred_te = model.predict(
                            X_test,
                            num_iterations=model.best_iteration_
                            )
                elif model_name == 'XGB':
                    if task == 'clf':
                        path_model = os.path.join(models_dir, f'XGBClassifier_{i}_trainedmodel.pkl')
                        model = load_xgb(path_model)
                        y_pred_va = model.predict_proba(
                            X_train[va_idx],
                            ntree_limit=model.best_iteration
                            )[:, 1]
                        y_pred_te = model.predict(
                            X_test,
                            ntree_limit=model.best_iteration
                            )[:, 1]
                    elif task == 'reg':
                        path_model = os.path.join(models_dir, f'XGBRegressor_{i}_trainedmodel.pkl')
                        model = load_xgb(path_model)
                        y_pred_va = model.predict(
                            X_train[va_idx],
                            ntree_limit=model.best_iteration
                            )
                        y_pred_te = model.predict(
                            X_test,
                            ntree_limit=model.best_iteration
                            )
                elif model_name == 'NN':
                    if task == 'clf':
                        path_json = os.path.join(models_dir, 'NNClassifier_architecture.json')
                        path_h5 = os.path.join(models_dir, f'NNClassifier_{i}_trainedweight.h5')
                        path_transformer = os.path.join(models_dir, f'NNClassifier_{i}_transformer.pkl')
                    elif task == 'reg':
                        path_json = os.path.join(models_dir, 'NNRegressor_architecture.json')
                        path_h5 = os.path.join(models_dir, f'NNRegressor_{i}_trainedweight.h5')
                        path_transformer = os.path.join(models_dir, f'NNRegressor_{i}_transformer.pkl')
                    model, transformer = load_nn(path_json, path_h5, path_transformer)
                    if transformer:
                        x_va = transformer.transform(np.nan_to_num(X_train[va_idx]))
                        x_te = transformer.transform(np.nan_to_num(X_test))
                    else:
                        x_va = np.nan_to_num(X_train[va_idx])
                        x_te = np.nan_to_num(X_test)
                    y_pred_va = model.predict(x_va).astype("float64")
                    y_pred_va = y_pred_va.flatten()
                    y_pred_te = model.predict(x_te).astype("float64")
                    y_pred_te = y_pred_te.flatten()
                # stack prediction features
                VA_IDXES.append(va_idx)
                VA_PREDS.append(y_pred_va)
                TE_PREDS.append(y_pred_te)
            # sort prediction features
            va_preds = sort_preds(VA_PREDS, VA_IDXES)
            te_preds = np.mean(TE_PREDS, axis=0)
        # convert prediction features to dataframe
        df_pred_va = pd.DataFrame(va_preds, columns=[model_name])
        df_pred_te = pd.DataFrame(te_preds, columns=[model_name])
        DF_PRED_VA.append(df_pred_va)
        DF_PRED_TE.append(df_pred_te)
    return pd.concat(DF_PRED_VA, axis=1), pd.concat(DF_PRED_TE, axis=1)


if __name__ == '__main__':

    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument('conf')
    args = parser.parse_args()

    # load config
    config = configparser.ConfigParser()
    config.read(f'{args.conf}')
    use_lgb = config.getboolean('stacking', 'LGB')
    use_xgb = config.getboolean('stacking', 'XGB')
    use_nn = config.getboolean('stacking', 'NN')
    models_dir = config.get('stacking', 'models_dir')
    train_data = config.get('general', 'train_data')
    test_data = config.get('stacking', 'test_data')

    # load logging
    config = configparser.ConfigParser()
    config.read('../results/logging/logging.ini')
    n_splits = int(config.get('logging_trainer', 'n_splits'))
    random_state = int(config.get('logging_trainer', 'random_state'))

    # load datasets
    df_train, df_test = pd.read_csv(train_data), pd.read_csv(test_data)
    X_train, y_train = np.array(df_train.iloc[:, 2:]), np.array(df_train.iloc[:, 1])
    X_test = np.array(df_test.iloc[:, 2:])

    # task declaration
    if len(set(y_train)) <= 2:
        task = 'clf'
    elif len(set(y_train)) > 2:
        task = 'reg'

    # generate prediction feaures
    models_dict = {'LGB': use_lgb, 'XGB': use_xgb, "NN": use_nn}
    df_pred_va, df_pred_te = gen_stk_features(
        models_dict,
        models_dir,
        task,
        X_train,
        y_train,
        X_test,
        n_splits,
        random_state
        )
    df_pred_va.to_csv('../results/stk_feature/prediction_features_valid.csv')
    df_pred_te.to_csv('../results/stk_feature/prediction_features_test.csv')

    # load features to create stacking model
    X_train = np.array(pd.read_csv('../results/stk_feature/prediction_features_valid.csv').iloc[:, 1:])
    X_test = np.array(pd.read_csv('../results/stk_feature/prediction_features_test.csv').iloc[:, 1:])

    # cv setting for stacking model
    if task == 'clf':
        cv = StratifiedKFold(n_splits=10, random_state=1029, shuffle=True)
    elif task == 'reg':
        cv = KFold(n_splits=10, random_state=1029, shuffle=True)

    # cross validation to build staking model
    VA_PREDS = []
    VA_IDXES = []
    TE_PREDS = []
    METRICS = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        # feature preprocessing
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        x_tr = transformer.fit_transform(X_train[tr_idx])
        x_va = transformer.transform(X_train[va_idx])
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]
        # model training
        if task == 'clf':
            model = LogisticRegression(solver='sag', n_jobs=-1)
        elif task == 'reg':
            model = LinearRegression(n_jobs=-1)
        model.fit(x_tr, y_tr)
        # prediction
        if task == 'clf':
            va_pred = model.predict_proba(x_va)[:, 1]
            te_pred = model.predict_proba(X_test)[:, 1]
        elif task == 'reg':
            va_pred = model.predict(x_va)
            te_pred = model.predict(X_test)
        VA_PREDS.append(va_pred)
        VA_IDXES.append(va_idx)
        TE_PREDS.append(te_pred)
        # evaluation
        if task == 'clf':
            cutoff = roc_cutoff(y_va, va_pred)
            metrics = evaluate_clf(y_va, va_pred, cutoff)
        elif task == 'reg':
            metrics = evaluate_reg(y_va, va_pred)
        METRICS.append(pd.DataFrame(metrics))
    # saving metrics
    model_names = []
    for k, v in models_dict.items():
        if v:
            model_names.append(k)
    stk_comb = '--'.join(model_names)
    out_root = '../results/valid_metrics'
    out_neme = os.path.join(out_root, f'STK_{stk_comb}_metrics.csv')
    df_METRICS = pd.concat(METRICS)
    df_METRICS.to_csv(out_neme)

    va_preds = sort_preds(VA_PREDS, VA_IDXES)
    te_preds = np.mean(TE_PREDS, axis=0)
    pd.DataFrame(va_preds).to_csv(f'../results/prediction_results/STK_{stk_comb}_prediction_valid.csv')
    pd.DataFrame(te_preds).to_csv(f'../results/prediction_results/STK_{stk_comb}_prediction._test.csv')
