import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from neuralnetwork import NNClassifier, NNRegressor
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
    n_spilits = config.get('general', 'n_spilits_train')

    # directory existance
    out_dir = os.path.join(f'../results/{pj_name}', 'models')
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
    