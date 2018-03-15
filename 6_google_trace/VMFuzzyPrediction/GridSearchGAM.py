from sklearn.grid_search import GridSearchCV
from io_utils.NumLoad import *
from sklearn.cross_validation import train_test_split
from estimators.FuzzyFlow import FuzzyFlow
from utils.TrainingTestMaker import TrainingTestMaker
from scaling.ProactiveSLA import ProactiveSLA
import pandas as pd
import numpy as np

import math
from utils.SlidingWindowUtil import SlidingWindow
from sklearn.preprocessing import MinMaxScaler
from estimators.GAEstimator import GAEstimator
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from estimators.NeuralFlow import *
from utils.GraphUtil import *
from sklearn import datasets, metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.GraphUtil import *
from estimators.GAEstimator import GAEstimator
from io_utils.NumLoad import *
from estimators.OptimizerNNEstimator import OptimizerNNEstimator
from scaling.ProactiveSLA import ProactiveSLA
from io_utils.GFeeder import GFeeder




dataset_holder = []
trainee_holder = {}
metrics = ["cpu_rate","mem_usage"]
arr_desk = ['X_train','y_train','X_test']
sliding_number = 3
data = pd.read_csv('sampling_617685_metric_10min_datetime.csv',parse_dates=True,index_col=0)[:3000]
def experiment(sliding_number=3):
    for metric in metrics:
        dat = pd.Series(data[metric].round(3))
        fuzzy_engine = FuzzyFlow()
        data_maker = TrainingTestMaker()
        fuzzy_engine.fit_transform(dat)
        sliding = np.array(list(SlidingWindow(fuzzy_engine.u_class_transform, sliding_number)))
        X_train, y_train, X_test, y_test = data_maker.make_fuzzy_test(sliding, fuzzy_engine.u_class_transform, dat)
        dataset_holder.append(fuzzy_engine)
        trainee_holder[metric] = {
            'X_train': X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
    y_train = np.asarray(zip(trainee_holder['cpu_rate']['y_train'], trainee_holder['mem_usage']['y_train']))
    # X_test = zip(trainee_holder['cpu_rate']['X_test'],trainee_holder['mem_usage']['X_test'])
    X_train = []
    X_test = []
    # y_train = []
    for i in np.arange(len(trainee_holder['cpu_rate']['X_train'])):
        tmp = zip(trainee_holder['cpu_rate']['X_train'][i], trainee_holder['mem_usage']['X_train'][i])
        X_train.append(np.ravel(tmp))
    for i in np.arange(len(trainee_holder['cpu_rate']['X_test'])):
        tmp = zip(trainee_holder['cpu_rate']['X_test'][i], trainee_holder['mem_usage']['X_test'][i])
        X_test.append(np.ravel(tmp))
    X_train = np.array(X_train)
    n_hidden = 10
    X_test = np.array(X_test)
    fit_params = {
        'neural_shape': [len(X_train[0]), n_hidden, 2]
    }
    param_dicts = {
    	"cross_rate":np.arange(0.3,0.5,step=0.01),
    	"pop_size":[45,60],
	"mutation_rate":np.arange(0.02,0.05,step=0.01),
    	"gen_size": [100,150,200]
    }
    estimator = GAEstimator(cross_rate=0.5, mutation_rate=0.06, gen_size=100, pop_size=30)
    gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=-1,fit_params=fit_params,scoring='mean_squared_error')
    gs.fit(X_train,y_train)   
result = [experiment(sliding_number=i) for i in np.arange(2, 6)]
result = pd.DataFrame(result, columns=["sliding_number", "MAE CPU", "MAE RAM"])
result.to_csv('fuzzy_gabpnn_experimentm.csv')


