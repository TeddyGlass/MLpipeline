from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import QuantileTransformer
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor


class NNClassifier:
    '''
    Usage:
    clf = NNClassifier(**params)
    history = clf.fit(
    X_train,
    y_train,
    X_valid,
    y_valid,
    early_stopping_rounds
    )
    '''
    
    def __init__(self, input_dropout=0.01, hidden_layers=2, hidden_units=256, hidden_dropout=0.01,
                 batch_norm="before_act", learning_rate=1e-5, batch_size=64, epochs=10000,
                 standardization=True
                 ):
        self.input_dropout = input_dropout  # layer param
        self.hidden_layers = int(hidden_layers)  # layer param
        self.hidden_units = int(hidden_units)  # layer param
        self.hidden_dropout = hidden_dropout  # layer param
        self.batch_norm = batch_norm  # layer param
        self.learning_rate = learning_rate  # optimizer param
        self.batch_size = int(batch_size)  # fit param
        self.epochs = int(epochs)  # fit param
        self.standardization = standardization

    def fit(self, X_train, y_train, early_stopping_rounds, eval_set, eval_metric, verbose=1):
        # Data standardization
        if self.standardization:
            self.transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
            X_train = self.transformer.fit_transform(X_train)
            X_valid = self.transformer.transform(eval_set[1][0])
        elif self.standardization is False:
            X_valid = eval_set[1][0]
        
        # Keras Wrapper for sklearn AIP
        def create_model():
            self.model = Sequential()
            self.model.add(Dropout(self.input_dropout, input_shape=(X_train.shape[1],)))
            for i in range(self.hidden_layers):
                self.model.add(Dense(self.hidden_units))
                if self.batch_norm == 'before_act':
                    self. model.add(BatchNormalization())
                self.model.add(ReLU())
                self.model.add(Dropout(self.hidden_dropout))
            self.model.add(Dense(1, activation='sigmoid'))
            # Optimazer
            optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, decay=0.)
            # Compile
            self.model.compile(
                loss=eval_metric,
                optimizer=optimizer,
                metrics=['accuracy']
            )
            return self.model
        early_stopping = EarlyStopping(patience=early_stopping_rounds, restore_best_weights=True)
        self.model = KerasClassifier(
            build_fn=create_model,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_data=(X_valid, eval_set[1][1]),
            callbacks=[early_stopping]
            )
        self.model.fit(X_train, y_train)
    
    def predict(self, x):
        if self.standardization:
            x = self.transformer.transform(x)
        y_pred = self.model.predict(x).astype("float64")
        y_pred = y_pred.flatten()
        return y_pred

    def get_model(self):
        return self.model

    def get_transformer(self):
        if self.standardization:
            return self.transformer


class NNRegressor(NNClassifier):

    def fit(self, X_train, y_train, early_stopping_rounds, eval_set, eval_metric, verbose=1):
        # Data standardization
        if self.standardization:
            self.transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
            X_train = self.transformer.fit_transform(X_train)
            X_valid = self.transformer.transform(eval_set[1][0])
        elif self.standardization is False:
            X_valid = eval_set[1][0]
        
        # Keras Wrapper for sklearn AIP
        def create_model():
            self.model = Sequential()
            self.model.add(Dropout(self.input_dropout, input_shape=(X_train.shape[1],)))
            for i in range(self.hidden_layers):
                self.model.add(Dense(self.hidden_units))
                if self.batch_norm == 'before_act':
                    self. model.add(BatchNormalization())
                self.model.add(ReLU())
                self.model.add(Dropout(self.hidden_dropout))
            self.model.add(Dense(1))
            # Optimazer
            optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, decay=0.)
            # Compile
            self.model.compile(
                loss=eval_metric,
                optimizer=optimizer,
                metrics=['mae']
            )
            return self.model
        early_stopping = EarlyStopping(patience=early_stopping_rounds, restore_best_weights=True)
        self.model = KerasRegressor(
            build_fn=create_model,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_data=(X_valid, eval_set[1][1]),
            callbacks=[early_stopping]
            )
        self.model.fit(X_train, y_train)
