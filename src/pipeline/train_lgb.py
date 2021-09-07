import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier, LGBMRegressor
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

    if len(list(set(y))) <= 2:
        task = 'clf'
        if args.LGB:
            obj = Objective(
                Trainer(LGBMClassifier()),
                X,
                y, 
                n_splits,
                early_stopping_rounds, 
                random_state
                )
        elif args.XGB:
            
    elif len(list(set(y))) > 3:
        task = 'reg'
        obj = Objective(
            Trainer(LGBMRegressor()),
            X,
            y, 
            n_splits,
            early_stopping_rounds, 
            random_state
            )