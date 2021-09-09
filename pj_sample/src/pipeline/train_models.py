from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from neuralnetwork import NNClassifier, NNRegressor
from sklearn.model_selection import StratifiedKFold, KFold

from trainer import Trainer
from utils import set_params, evaluate_clf, evaluate_reg, roc_cutoff

import numpy as np
import pandas as pd
import pickle
import os
import datetime
import configparser
import argparse


if __name__ == "__main__":
    # argments settings
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_conf')
    parser.add_argument('--LGB', action='store_true')
    parser.add_argument('--XGB', action='store_true')
    parser.add_argument('--NN', action='store_true')
    parser.add_argument('--use_best_params', action='store_true')
    # parser.add_argument('--gen_stack_features', action='store_true')
    args = parser.parse_args()

    # load config
    config = configparser.ConfigParser()
    config.read(f'{args.path_to_conf}')

    # general settings
    pj_name = config.get('general', 'pj_name')
    train_dataset = config.get('general', 'train_data')
    n_splits = int(config.get('general', 'n_splits_train'))
    early_stopping_rounds = int(config.get('general', 'early_stopping_rounds'))
    random_state = int(config.get('general', 'random_state'))

    # out directory existance
    out_dirs = []
    for dir in ['trained_models', 'valid_metrics']:
        out_dir = os.path.join(f'../results/{pj_name}', dir)
        if os.path.exists(out_dir) == False:
            os.makedirs(out_dir)
        out_dirs.append(out_dir)

    # dataset
    df = pd.read_csv(train_dataset)
    if args.LGB or args.XGB:
        X = np.array(df.iloc[:, 2:])
    elif args.NN:
        X = np.array(df.iloc[:, 2:].fillna(0))
    y = np.array(df.iloc[:, 1])

    # task declaration
    if len(set(y)) <= 2:
        task = 'clf'
    elif len(set(y)) > 2:
        task = 'reg'

    # cv settings
    if task == 'clf':
        cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    elif task == 'reg':
        cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # cross validation
    now = datetime.datetime.now()
    METRICS = []
    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        if args.LGB:
            # lgb parameter settings
            if args.use_best_params:
                best_params_path = config.get('lgb_params', 'best_prams_path')
            else:
                best_params_path = None
            # lgb model instantiation
            if task == 'clf':
                params = set_params(Trainer(LGBMClassifier()), args.path_to_conf, best_params_path)
                model = Trainer(LGBMClassifier(**params))
            elif task == 'reg':
                params = set_params(Trainer(LGBMRegressor()), args.path_to_conf, best_params_path)
                model = Trainer(LGBMRegressor(**params))
        elif args.XGB:
            # xgb parameter settings
            if args.use_best_params:
                best_params_path = config.get('xgb_params', 'best_prams_path')
            else:
                best_params_path = None
            # xgb model instantiation
            if task == 'clf':
                params = set_params(Trainer(XGBClassifier()), args.path_to_conf, best_params_path)
                model = Trainer(XGBClassifier(**params))
            elif task == 'reg':
                params = set_params(Trainer(XGBRegressor()), args.path_to_conf, best_params_path)
                model = Trainer(XGBRegressor(**params))
        elif args.NN:
            # nn parameter settings
            if args.use_best_params:
                best_params_path = config.get('nn_params', 'best_prams_path')
            else:
                best_params_path = None
            # nn model instantiation
            if task == 'clf':
                params = set_params(Trainer(NNClassifier()), args.path_to_conf, best_params_path)
                model = Trainer(NNClassifier(**params))
            elif task == 'reg':
                params = set_params(Trainer(NNRegressor()), args.path_to_conf, best_params_path)
                model = Trainer(NNRegressor(**params))
        # training
        model_name = type(model.get_model()).__name__
        print('-'*100)
        print(f'Training of {model_name} (fold {i+1}/{n_splits}) has begun with following parameters')
        print(params)
        model.fit(
                X[tr_idx],
                y[tr_idx],
                X[va_idx],
                y[va_idx],
                early_stopping_rounds
            )
        print(f'Training (fold {i+1}) has been been completed.')
        print('-'*100)
        # evaluation
        print('-'*100)
        print(f'Evaluation metrics (fold {i+1}/{n_splits})')
        y_pred = model.predict(X[va_idx])
        if task == 'clf':
            cutoff = roc_cutoff(y[va_idx], y_pred)
            metrics = evaluate_clf(y[va_idx], y_pred, cutoff)
        elif task == 'reg':
            metrics = evaluate_reg(y[va_idx], y_pred)
        METRICS.append(pd.DataFrame(metrics))
        print(metrics)
        print('-'*100)
        # saving learning curve
        model.get_learning_curve(
            os.path.join(out_dirs[1], f'{model_name}_{i}_learning_curve.png')
            )
        # saving model
        model = model.get_model()
        if args.LGB or args.XGB: # LGB model or XGB model
            out_name = os.path.join(
                out_dirs[0],
                f'{model_name}_{i}_trainedmodel_{now}.pkl'
                )
            with open(out_name, 'wb') as f:
                pickle.dump(model, f)
            del model
        elif args.NN: # Keras model
            # saving transfomer for feature standardization
            if params['standardization']:
                transformer = model.get_transformer()
                out_file = os.path.join(
                    out_dirs[0],
                    f'{model_name}_{i}_transformer_{now}.pkl'
                    )
                with open(out_file, 'wb') as f:
                    pickle.dump(transformer, f)
            # saving weight
            weight_name = os.path.join(
                out_dirs[0],
                f'{model_name}_{i}_trainedweight_{now}.h5'
                )
            model.model.save(weight_name)
            # saving model architecture
            archit_name = os.path.join(
                out_dirs[0],
                f'{model_name}_architecture_{now}.json'
                )
            json_string = model.model.to_json()
            with open(archit_name, 'w') as f:
                f.write(json_string)
            del model

    # saving metrics
    out_neme = os.path.join(out_dirs[1], f'{model_name}_metrics.csv')
    df_METRICS = pd.concat(METRICS)
    df_METRICS.to_csv(out_neme)