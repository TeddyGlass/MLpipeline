import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:

    '''
    # Usage
    n_splits = 3
    random_state = 0
    early_stopping_rounds=10
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for tr_idx, va_idx in kf.split(X, y):
        model = Trainer(XGBRegressor(**XGB_PARAMS), "classification")
        model.fit(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            y[va_idx],
            early_stopping_rounds
        )
        model.get_learning_curve()
    '''

    def __init__(self, model, task):
        self.model = model
        self.model_type = type(model).__name__
        self.task = task
        self.best_iteration = 100
        self.train_rmse = []
        self.valid_rmse = []
        self.importance = []

    
    def fit(self,
            X_train, y_train, X_valid, y_valid,
            early_stopping_rounds):

        eval_set = [(X_train, y_train), (X_valid, y_valid)]

        if self.model_type[:3] == "LGB":
            if self.task == 'classification':
                eval_metric = 'logloss'
                monitoring_loss = 'binary_logloss'
            elif self.task == 'regression':
                eval_metric = 'rmse'
                monitoring_loss = 'rmse'
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=True
            )
            self.best_iteration = self.model.best_iteration_
            self.train_logloss = np.array(
                self.model.evals_result_['training'][monitoring_loss])
            self.valid_logloss = np.array(
                self.model.evals_result_['valid_1'][monitoring_loss])
        
        elif self.model_type[:3] == 'XGB':
            if self.task == 'classification':
                eval_metric = 'logloss'
            elif self.task == 'regression':
                eval_metric = 'rmse'
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=1
            )
            self.best_iteration = self.model.best_iteration
            self.train_logloss = np.array(
                self.model.evals_result_['validation_0'][eval_metric])
            self.valid_logloss = np.array(
                self.model.evals_result_['validation_1'][eval_metric])
            
        elif self.model_type[:2] == 'NN':
            if self.task == 'classification':
                eval_metric = 'binary_crossentropy'
            elif self.task == 'regression':
                eval_metric = 'mean_squared_error'
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=1
            )
            history = self.model.get_model().history
            self.train_logloss = np.array(history.history['loss'])
            self.valid_logloss = np.array(history.history['val_loss'])
            

    def predict(self, X):
        if self.model_type[:3] == "LGB":
            if self.task == 'classification':
                return self.model.predict_proba(X, num_iterations=self.best_iteration)[:,1]
            elif self.task == 'regression':
                return self.model.predict(X, num_iterations=self.best_iteration)
        elif self.model_type[:3] == "XGB":
            if self.task == 'classification':
                return self.model.predict_proba(X, ntree_limit=self.best_iteration)[:,1]
            elif self.task == 'regression':
                return self.model.predict(X, ntree_limit=self.best_iteration)
        elif self.model_type[:2] == 'NN':
            return self.model.predict(X)
        
        
    def get_model(self):
        if self.model_type[:3] == "LGB":
            return self.model
        elif self.model_type[:3] == "XGB":
            return self.model
        elif self.model_type[:2] == 'NN':
            return self.model.get_model()

    
    def get_best_iteration(self):
        print(print(f"model type is {self.model_type}"))
        return self.best_iteration


        
    def get_learning_curve(self):
        palette = sns.diverging_palette(220, 20, n=2)
        width = np.arange(self.train_logloss.shape[0])
        plt.figure(figsize=(10, 7.32))
        plt.title(
            'Learning_Curve ({})'.format(self.model_type), fontsize=15)
        plt.xlabel('Iterations', fontsize=15)
        if self.task == 'classification':
            plt.ylabel('LogLoss', fontsize=15)
            plt.plot(width, self.train_logloss, label='train_logloss', color=palette[0])
            plt.plot(width, self.valid_logloss, label='valid_logloss', color=palette[1])
        elif self.task == 'regression':
            if self.model_type[:3] == "LGB" or self.model_type[:3] == "XGB":
                plt.ylabel('RMSE', fontsize=15)
                plt.plot(width, self.train_logloss, label='train_rmse', color=palette[0])
                plt.plot(width, self.valid_logloss, label='valid_rmse', color=palette[1])
            elif self.model_type[:2] == "NN":
                plt.ylabel('MSE', fontsize=15)
                plt.plot(width, self.train_logloss, label='train_mse', color=palette[0])
                plt.plot(width, self.valid_logloss, label='valid_mse', color=palette[1])
        plt.legend(loc='upper right', fontsize=13)
        plt.show()
