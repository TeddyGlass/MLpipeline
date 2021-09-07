import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from neuralnetwork import NNClassifier, NNRegressor
from optimizer import optuna_search, Objective
from trainer import Trainer
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
    args = parser.parse_args()

    # load config
    config = configparser.ConfigParser()
    config.read(f'{args.path_to_conf}')
    # general settings
    pj_name = config.get('general', 'pj_name')
    train_dataset = config.get('general', 'train_data')
    # optimizer settings
    early_stopping_rounds = int(config.get('optimizer', 'early_stopping_rounds'))
    n_splits = int(config.get('optimizer', 'n_splits'))
    n_trials = int(config.get('optimizer', 'n_trials'))
    n_jobs = int(config.get('optimizer', 'n_jobs'))
    random_state = int(config.get('optimizer', 'random_state'))

    # directory existance
    out_dir = os.path.join(f'../results/{pj_name}', 'best_params')
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    
    # dataset
    df = pd.read_csv(train_dataset)
    if args.LGB or args.XGB:
        X = np.array(df.iloc[:,2:])
    elif args.NN:
        X = np.array(df.iloc[:,2:].fillna(0))
    y = np.array(df.iloc[:,1])

    # model instantiation
    if len(list(set(y))) <= 2:
        task = 'clf'
        if args.LGB:
            model = Trainer(LGBMClassifier())
        elif args.XGB:
            model = Trainer(XGBClassifier())
        elif args.NN:
            model = Trainer(NNClassifier())   
    elif len(list(set(y))) > 2:
        task = 'reg'
        if args.LGB:
            model = Trainer(LGBMRegressor())
        elif args.XGB:
            model = Trainer(XGBRegressor())
        elif args.NN:
            model = Trainer(NNRegressor())
    
    model_name = type(model.get_model()).__name__
    print('-'*100)
    print(f'Start {model_name} parameters optimization')
    obj = Objective(
        model,
        X,
        y, 
        n_splits,
        early_stopping_rounds, 
        random_state
        )
    
    best_params = optuna_search(obj, n_trials, n_jobs, random_state)
    print(f'Best_params {type(model.get_model()).__name__}: ', best_params)
    print('-'*100)

    dt_now = datetime.datetime.now()
    out_file = os.path.join(out_dir, f'{model_name}_bestparams_{dt_now}.pkl')
    with open(out_file,'wb') as f:
        pickle.dump(best_params ,f)
