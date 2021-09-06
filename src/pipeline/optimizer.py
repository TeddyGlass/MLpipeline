from trainer import Trainer
from sklearn.metrics import log_loss
from utils import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold, Kfold
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from neuralnetwork import NNClassifier, NNRlassifier
import optuna


class Objective:
     
    '''
    # Usage
    obj = Objective(LGBMRegressor(), X, y)
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=123))
    study.optimize(obj, n_trials=10, n_jobs=-1)
    '''

    def __init__(self, model, x, y, task):
        self.model = model
        self.model_type = type(self.model).__name__
        self.x = x
        self.y = y
        self.n_splits = 3
        self.random_state = 1214
        self.early_stopping_rounds = 20
        self.task = task
    
    def __call__(self, trial):
        if self.model_type[:3] == 'LGB':
            self.SPACE = {
                'num_leaves': trial.suggest_int(
                'num_leaves', 8, 31),
                'subsample': trial.suggest_uniform('subsample', 0.60, 0.80),
                'colsample_bytree': trial.suggest_uniform(
                    'colsample_bytree', 0.60, 0.80),
                'bagging_freq': trial.suggest_int(
                    'bagging_freq', 1, 51, 5),
                'min_child_weight': trial.suggest_loguniform(
                    'min_child_weight', 1, 32),
                'min_child_samples': int(trial.suggest_discrete_uniform(
                    'min_child_samples', 5, 50, 5)),
                'min_split_gain': trial.suggest_loguniform(
                    'min_split_gain', 1e-5, 1e-1),
                'learning_rate': 0.05,
                'n_estimators': 1000000,
                'random_state': 1112
            }
        elif self.model_type[:3] == 'XGB':
            self.SPACE = {
                'subsample': trial.suggest_uniform(
                    'subsample', 0.65, 0.85),
                'colsample_bytree': trial.suggest_uniform(
                    'colsample_bytree', 0.65, 0.80),
                'gamma': trial.suggest_loguniform(
                    'gamma', 1e-8, 1.0),
                'min_child_weight': trial.suggest_loguniform(
                    'min_child_weight', 1, 32),
                'learning_rate': 0.05,
                'n_estimators': 1000000,
                'random_state': 1112
            }
        elif self.model_type[:2] == 'NN':
            self.SPACE = {
                "input_dropout": trial.suggest_uniform(
                    "input_dropout", 0.01, 0.4),
                "hidden_layers": trial.suggest_int(
                    "hidden_layers", 1, 2),
                'hidden_units': int(trial.suggest_discrete_uniform(
                    'hidden_units', 64, 256, 64)),
                'hidden_dropout': trial.suggest_uniform(
                    'hidden_dropout', 0.01, 0.4),
                'batch_norm': trial.suggest_categorical(
                'batch_norm', ['before_act', 'non']),
                'batch_size': int(trial.suggest_discrete_uniform(
                    'batch_size', 32, 128, 16)),
                'input_shape':self.x.shape[1]
            }
        # cross validation
        if self.task == 'classification':
            cv = StratifiedKFold(n_splits=self.n_splits,
            random_state=self.random_state, shuffle=True)
        elif self.task == 'regression':
            cv = KFold(n_splits=self.n_splits,
            random_state=self.random_state, shuffle=True)
        LOSS = []
        for tr_idx, va_idx in cv.split(self.x, self.y):
            clf = Trainer(self.model(**self.SPACE))
            clf.fit(
                self.x[tr_idx],
                self.y[tr_idx],
                self.x[va_idx],
                self.y[va_idx],
                self.early_stopping_rounds
            )
            y_pred = clf.predict(self.x[va_idx])  # best_iteration
            if self.task == 'classification':
                loss = log_loss(self.y[va_idx], y_pred)
            elif self.task == 'regression':
                loss = root_mean_squared_error(self.y[va_idx], y_pred)
            LOSS.append(loss)

        #     # cross validation
        #     skf = StratifiedKFold(n_splits=self.n_splits,
        #     random_state=self.random_state, shuffle=True)
        #     LOGLOSS = []
        #     for tr_idx, va_idx in skf.split(self.x, self.y):
        #         clf = Trainer(LGBMClassifier(**self.PARAMS))
        #         clf.fit(
        #             self.x[tr_idx],
        #             self.y[tr_idx],
        #             self.x[va_idx],
        #             self.y[va_idx],
        #             self.early_stopping_rounds
        #         )
        #         y_pred = clf.predict_proba(self.x[va_idx])  # best_iteration
        #         logloss = log_loss(self.y[va_idx], y_pred)
        #         LOGLOSS.append(logloss)
        #     return np.mean(LOGLOSS)
        # elif self.model_type == 'XGBClassifier':
        #     SPACE = {
        #         'subsample': trial.suggest_uniform(
        #             'subsample', 0.65, 0.85),
        #         'colsample_bytree': trial.suggest_uniform(
        #             'colsample_bytree', 0.65, 0.80),
        #         'gamma': trial.suggest_loguniform(
        #             'gamma', 1e-8, 1.0),
        #         'min_child_weight': trial.suggest_loguniform(
        #             'min_child_weight', 1, 32)
        #     }
        #     self.PARAMS.update(SPACE)
        #     # cross validation
        #     skf = StratifiedKFold(n_splits=self.n_splits,
        #     random_state=self.random_state, shuffle=True)
        #     LOGLOSS = []
        #     for tr_idx, va_idx in skf.split(self.x, self.y):
        #         clf = Trainer(XGBClassifier(**self.PARAMS))
        #         clf.fit(
        #             self.x[tr_idx],
        #             self.y[tr_idx],
        #             self.x[va_idx],
        #             self.y[va_idx],
        #             self.early_stopping_rounds
        #         )
        #         y_pred = clf.predict_proba(self.x[va_idx])  # best_iteration
        #         logloss = log_loss(self.y[va_idx], y_pred)
        #         LOGLOSS.append(logloss)
        #     return np.mean(LOGLOSS)
        # elif self.model_type == 'NNClassifier':
        #     self.PARAMS['input_shape'] = self.x.shape[1]
        #     SPACE = {
        #         "input_dropout": trial.suggest_uniform(
        #             "input_dropout", 0.2, 1.0),
        #         "hidden_layers": trial.suggest_int(
        #             "hidden_layers", 1, 2),
        #         'hidden_units': int(trial.suggest_discrete_uniform(
        #             'hidden_units', 8, 64, 8)),
        #         'hidden_dropout': trial.suggest_uniform(
        #             'hidden_dropout', 0.2, 1.0),
        #         'batch_norm': trial.suggest_categorical(
        #         'batch_norm', ['before_act', 'non']),
        #         'batch_size': int(trial.suggest_discrete_uniform(
        #             'batch_size', 16, 64, 16))
        #     }
        #     self.PARAMS.update(SPACE)
        #     # cross validation
        #     skf = StratifiedKFold(n_splits=self.n_splits,
        #     random_state=self.random_state, shuffle=True)
        #     LOGLOSS = []
        #     for tr_idx, va_idx in skf.split(self.x, self.y):
        #         clf = Trainer(NNClassifier(**self.PARAMS))
        #         clf.fit(
        #             self.x[tr_idx],
        #             self.y[tr_idx],
        #             self.x[va_idx],
        #             self.y[va_idx],
        #             self.early_stopping_rounds
        #         )
        #         y_pred = clf.predict_proba(self.x[va_idx])
        #         logloss = log_loss(self.y[va_idx], y_pred) # best weghts
        #         LOGLOSS.append(logloss)
        #     return np.mean(LOGLOSS)
            
def optuna_search(obj, n_trials, n_jobs, random_state):
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=random_state))
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params


if __name__ == "__main__":
    pass
