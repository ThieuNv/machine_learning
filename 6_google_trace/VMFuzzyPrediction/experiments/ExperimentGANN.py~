from sklearn.grid_search import GridSearchCV

from estimators.GAEstimator import GAEstimator
from io_utils.NumLoad import *

param_dicts = {
    "cross_rate":np.arange(0.1,1.0,step=0.05),
    "pop_size":[45,55,60],
    "mutation_rate":np.arange(0.01,0.07,step=0.01),
    'gen_size': [100,150,200]
}
n_windows = 3
n_hidden = 10
#dataFeeder = GFeeder(skip_lists=3)
#X_train,y_train,X_test,y_test = dataFeeder.split_train_and_test(n_sliding_window=n_windows)
X_train,y_train,X_test,y_test = load_training_from_npz("../bpnn_direct.npz")
neural_shape = [len(X_train[0]),n_hidden,1]
estimator = GAEstimator()
fit_param = {'neural_shape':neural_shape}
# estimator.fit(X,y,**fit_param)
gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=-1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X_train,y_train)
print gs.best_estimator_
print gs.best_estimator_.score(X_test,y_test)