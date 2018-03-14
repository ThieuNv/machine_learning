from keras.initializations import glorot_uniform,uniform, lecun_uniform,glorot_normal
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adagrad,adam,sgd, rmsprop, adamax
from keras.activations import sigmoid, hard_sigmoid, tanh, relu
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from utils.initializer import *

"""
    Base Class Estimator for Keras
"""
optimizer_mapping = {
    "adagrad": adagrad,
    "sgd": sgd,
    "rmsprop": rmsprop,
    "adam": adam,
    "adamax": adamax
}
random_initialization = {
    "glorot_uniform":glorot_uniform,
    "uniform":uniform,
    "lecun_uniform":lecun_uniform,
    "glorot_normal":glorot_normal
}

class KerasRegressor(BaseEstimator):
    def get_params(self, deep=True):
        return {
            "uniform_init": self.uniform_init,
            "learning_rate": self.learning_rate,
            "activation": self.activation_name,
            "optimize": self.optimize,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "weights_matrix": self.weights_matrix,
        }

    def set_params(self, **params):
        for param, value in params.items():
            self.__setattr__(param, value)
        return self

    def __init__(self, hidden_nodes=None, uniform_init=True, learning_rate=1E-01, activation="relu", optimize="adam", steps=1000,
                 batch_size=100, weights_matrix=None, verbose=0,cv=False, callbacks=[]):
        print "Initialization"
        self.activation_name = activation
        optimizer = {
            "adagrad": adagrad,
            "sgd": sgd,
            "rmsprop": rmsprop,
            "adam": adam,
            "adamax": adamax
        }
        activation_fn = {
            "sigmoid":sigmoid,
            "hard_sigmoid":hard_sigmoid,
            "tanh":tanh,
            "relu":relu
        }
        self.optimization = optimizer[optimize](lr=learning_rate)
        self.activation = activation_fn[activation]
        # Initialize neural network shape
        self.learning_rate = learning_rate
        self.steps = steps
        self.optimize = optimizer[optimize]
        self.batch_size = batch_size
        self.weights_matrix = weights_matrix
        self.uniform_init = uniform_init
        self.weights_matrix = None
        self.network = None
        self.verbose = verbose
        self.callbacks = callbacks
        self.cross_validation = cv
        if type(hidden_nodes) is list:
            self.hidden_nodes = np.array(hidden_nodes)
        else:
            self.hidden_nodes = hidden_nodes
    def model_regression(self, X, y):
        model = Sequential()
        weights = self.weight_init(), self.bias_init()
        model.add(Dense(self.n_hidden[0], input_dim=X.shape[1], activation=self.activation, weights=weights))
        for hidden in self.n_hidden[1:]:
            weights = self.weight_init(), self.bias_init()
            model.add(Dense(hidden,activation=self.activation,weights=weights))
        model.add(Dense(len(y[0])))
        model.compile(optimizer=self.optimization,loss="mean_squared_error")
        return model

    def score(self, X, y):
        return mean_squared_error(y_true=y, y_pred=self.network.predict(X))

    def predict(self, X):
        return self.network.predict(X)

    def fit(self, X, y, **param):
        self.neural_shape = []
        if(param.has_key('neural_shape')):
            self.neural_shape = param.get("neural_shape")
            self.n_output = self.neural_shape[-1]
            self.n_hidden = self.neural_shape[1:-1]
            self.number_of_layers = len(self.neural_shape)
        else:
            self.n_input = len(X[0])
            if type(y[0]) is list:
                self.n_output = len(y[0])
            else:
                self.n_output = 1
            self.neural_shape = self.hidden_nodes.tolist()
            self.neural_shape.insert(0,self.n_input)
            self.neural_shape.append(self.n_output)
            self.n_hidden = self.hidden_nodes
        self.kFold = KFold(X.shape[0], n_folds=5, shuffle=True)
        self.weights_matrix = param.get('weights_matrix')
        self.weight_layers = [(self.neural_shape[t - 1], self.neural_shape[t]) for t in
                              range(1, len(self.neural_shape))]
        self.bias_layers = [self.neural_shape[t] for t in range(1, len(self.neural_shape))]
        self.total_nodes_per_layer = zip(self.weight_layers, self.bias_layers)
        self.total_nodes = 0
        for layer in self.total_nodes_per_layer:
            self.total_nodes += (layer[0][0] + 1) * layer[0][1]
        # If weights are None then initialize randomly
        if self.weights_matrix == None:
            self.W, self.b = initialize_param(self.weight_layers, self.bias_layers, self.uniform_init)
        else:
            self.W, self.b = self.set_weights(self.weights_matrix)
        # Iterator for weights and bias
        self.W_iter = iter(self.W)
        self.b_iter = iter(self.b)
        # Initialize neural network layers
        self.network = self.model_regression(X,y)
        print self.network.summary()
        early_stopper = EarlyStopping(monitor='val_loss',patience=20)
        self.summary = self.network.fit(X, y, batch_size=self.batch_size, nb_epoch=self.steps,
                                        verbose=self.verbose, validation_split=0.1, callbacks=self.callbacks)
        return self

    def weight_init(self):
        W = self.W_iter.next()
        return W

    def save(self, filename):
        return self.network.save(filename)

    def bias_init(self):
        b = self.b_iter.next()
        return b

    def plot(self, y_actual, y_pred, label=["predict", "actual"]):
        ax = plt.subplot()
        ax.plot(y_actual, label=label[0])
        ax.plot(y_pred, label=label[1])
        plt.show()

    def set_weights(self, weights_matrix):
        if (len(weights_matrix) != self.total_nodes):
            print "Check again weights shape, must be equal with total nodes"
            return
        self.total_nodes_per_layer = zip(self.weight_layers, self.bias_layers)
        current_pos = 0
        W = []
        b = []
        for layer in self.total_nodes_per_layer:
            total_nodes = (layer[0][0] + 1) * layer[0][1]
            weights = weights_matrix[current_pos:total_nodes + current_pos]
            current_pos = total_nodes
            b.append(weights[-layer[1]:])
            W.append(np.array(weights[:-layer[1]].reshape(layer[0])))
        self.W = W
        self.b = b
        return W, b
