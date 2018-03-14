from sklearn.grid_search import GridSearchCV

from estimators.ACOEstimator import ACOEstimator
from io_utils.GFeeder import GFeeder
from utils.initializer import *
from io_utils.NumLoad import *
param_dicts = {
    "Q":np.arange(0.01,0.05,step=0.01),
    "epsilon":np.arange(0.1,0.6,step=0.05),
    "number_of_solutions":np.arange(30,200)
}
n_windows = 3
n_hidden = 15

dataFeeder = GFeeder()
X_train,y_train, X_test,y_test = load_training_from_npz("fuzzy_train_direct.npz")
neural_shape = [len(X_train[0]),n_hidden,1]
estimator = ACOEstimator()
archive_solution = construct_solution(estimator.number_of_solutions,neural_shape)
fit_param = {'neural_shape':neural_shape,"archive":archive_solution}
# estimator.fit(X,y,**fit_param)
gs = GridSearchCV(estimator,param_grid=param_dicts,n_jobs=-1,fit_params=fit_param,scoring='mean_squared_error')
gs.fit(X_train,y_train)
print gs.best_estimator_
print gs.best_estimator_.score(X_test,y_test)
