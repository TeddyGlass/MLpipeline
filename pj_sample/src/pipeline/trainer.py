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
        model = Trainer(XGBRegressor(**XGB_PARAMS))
        model.fit(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            y[va_idx],
            early_stopping_rounds
        )
        model.get_learning_curve()
    '''
    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__
        self.best_iteration = 100
        self.train_rmse = []
        self.valid_rmse = []


    def fit(self,
            X_train, y_train, X_valid, y_valid,
            early_stopping_rounds):

        eval_set = [(X_train, y_train), (X_valid, y_valid)]

        if 'LGBM' in self.model_type:
            if 'Classifier' in self.model_type:
                eval_metric = 'logloss'
                monitoring_loss = 'binary_logloss'
            elif 'Regressor' in self.model_type:
                eval_metric = 'rmse'
                monitoring_loss = 'rmse'
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=False
            )
            self.best_iteration = self.model.best_iteration_
            self.train_logloss = np.array(
                self.model.evals_result_['training'][monitoring_loss])
            self.valid_logloss = np.array(
                self.model.evals_result_['valid_1'][monitoring_loss])

        elif 'XGB' in self.model_type:
            if 'Classifier' in self.model_type:
                eval_metric = 'logloss'
            elif 'Regressor' in self.model_type:
                eval_metric = 'rmse'
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=0
            )
            self.best_iteration = self.model.best_iteration
            self.train_logloss = np.array(
                self.model.evals_result_['validation_0'][eval_metric])
            self.valid_logloss = np.array(
                self.model.evals_result_['validation_1'][eval_metric])

        elif 'NN' in self.model_type:
            if 'Classifier' in self.model_type:
                eval_metric = 'binary_crossentropy'
            elif 'Regressor' in self.model_type:
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
        if 'LGBM' in self.model_type:
            if 'Classifier' in self.model_type:
                return self.model.predict_proba(X, num_iterations=self.best_iteration)[:,1]
            elif 'Regressor' in self.model_type:
                return self.model.predict(X, num_iterations=self.best_iteration)
        if 'XGB' in self.model_type:
            if 'Classifier' in self.model_type:
                return self.model.predict_proba(X, ntree_limit=self.best_iteration)[:,1]
            elif 'Regressor' in self.model_type:
                return self.model.predict(X, ntree_limit=self.best_iteration)
        if 'NN' in self.model_type:
            return self.model.predict(X)


    def get_model(self):
            return self.model


    def get_best_iteration(self):
        print(print(f"model type is {self.model_type}"))
        return self.best_iteration


    def get_learning_curve(self, out_file):
        palette = sns.diverging_palette(220, 20, n=2)
        width = np.arange(self.train_logloss.shape[0])
        plt.figure(figsize=(10, 7.32))
        plt.title(
            'Learning_Curve ({})'.format(self.model_type), fontsize=15)
        plt.xlabel('Iterations', fontsize=15)
        if 'Classifier' in self.model_type:
            plt.ylabel('LogLoss', fontsize=15)
            plt.plot(width, self.train_logloss, label='train_logloss', color=palette[0])
            plt.plot(width, self.valid_logloss, label='valid_logloss', color=palette[1])
        elif 'Regressor' in self.model_type:
            if 'LGBM' in self.model_type or 'XGB' in self.model_type:
                plt.ylabel('RMSE', fontsize=15)
                plt.plot(width, self.train_logloss, label='train_rmse', color=palette[0])
                plt.plot(width, self.valid_logloss, label='valid_rmse', color=palette[1])
            elif 'NN' in self.model_type:
                plt.ylabel('MSE', fontsize=15)
                plt.plot(width, self.train_logloss, label='train_mse', color=palette[0])
                plt.plot(width, self.valid_logloss, label='valid_mse', color=palette[1])
        plt.legend(loc='upper right', fontsize=13)
        if out_file is not None:
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
        elif out_file is None:
            plt.show()
