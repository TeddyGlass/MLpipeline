import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier, XGBRegressor
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
    
    df = pd.read_csv(train_dataset)
    X = np.array(df.iloc[:,2:])
    y = np.array(df.iloc[:,1])

    if len(list(set(y))) <= 2:
        task = 'clf'
        obj = Objective(
            Trainer(XGBClassifier()),
            X,
            y, 
            n_splits,
            early_stopping_rounds, 
            random_state
            )
    elif len(list(set(y))) > 3:
        task = 'reg'
        obj = Objective(
            Trainer(XGBRegressor()),
            X,
            y, 
            n_splits,
            early_stopping_rounds, 
            random_state
            )
    
    best_params = optuna_search(obj, n_trials, n_jobs, random_state)
    print('Completed. Best_params: ', best_params)

    dt_now = datetime.datetime.now()
    out_file = os.path.join(out_dir, f'xgb{task}_best_params_{dt_now}.pkl')
    with open(out_file,'wb') as f:
        pickle.dump(best_params ,f)
