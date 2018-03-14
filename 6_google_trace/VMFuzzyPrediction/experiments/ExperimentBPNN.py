import math

from pandas import DataFrame
import numpy as np 
from __init__fuzzy import *


def experiment(sliding_number=3, hidden_node=15):
    dat_nn = np.asarray(scaler.fit_transform(dat))
    X_train_size = int(len(dat_nn)*0.7)
    sliding = np.array(list(SlidingWindow(dat_nn, sliding_number)))
    X_train_nn = sliding[:X_train_size]
    y_train_nn = dat_nn[sliding_number:X_train_size+sliding_number].reshape(-1,1)
    X_test_nn = sliding[X_train_size:]
    y_test_nn = dat_nn[X_train_size+sliding_number-1:].reshape(-1,1)
    y_actual_test = dat[X_train_size+sliding_number-1:].tolist()

    estimator = KerasRegressor(learning_rate=0.01, hidden_nodes=[hidden_node], steps=5000, optimize='Adam')
    estimator.fit(X_train_nn, y_train_nn)
    y_pred = scaler.inverse_transform(estimator.predict(X_test_nn))
    score_mape = mean_absolute_error(y_pred, y_actual_test)
    score_rmse = math.sqrt(mean_squared_error(y_pred,y_actual_test))
    np.savez('model_saved/BPNN_%s_%s' % (sliding_number, score_mape), y_pred=y_pred, y_true=y_actual_test)
    return sliding_number, score_rmse, score_mape

result = [[experiment(sliding_number=i) for i in np.arange(2,6)] for j in np.arange(10)]
#np.savez("BPNN_epochs",result=result)
results = DataFrame(np.array(result).reshape(-1,3), columns=["sliding_number","rmse","mae"])
results.to_csv('experiment_logs/bpnn_experiment.csv')
