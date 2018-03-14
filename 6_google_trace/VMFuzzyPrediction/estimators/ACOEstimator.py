from sklearn.base import BaseEstimator
#from monitors.BaseMonitor import Monitor
from FeedFlow import FeedFlow
from utils.initializer import *
import sys

# In[150]:
class ACOEstimator(BaseEstimator):
    def __init__(self,neural_shape=None,number_of_weights=None,number_of_solutions=100,max_epochs = 100,error_criteria = 0.9,
                 epsilon = 0.75,const_sd = 0.1,Q=0.08,hidden_nodes=None,**kwargs):
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.error_criteria = error_criteria
        self.number_of_solutions = self.k = number_of_solutions
        self.number_of_weights = number_of_weights
        self.const_sd = const_sd
        self.neural_shape = neural_shape
        self.Q = Q #Attractive param, q is small, the best-ranked solution is strongly preferred
        self.best_loss = np.Inf
        self.best_archive = []
        self._estimator_type="regressor"
        self.top_k = 1
        self.hidden_nodes = hidden_nodes
        #self.monitor = Monitor(type_train="ACO Optimization")
    def get_params(self, deep=True):
        return {
            "epsilon":self.epsilon,
            "max_epochs":self.max_epochs,
            "number_of_solutions":self.number_of_solutions,
            "Q":self.Q,
            "error_criteria":self.error_criteria,
            "number_of_weights":self.number_of_weights,
            "neural_shape":self.neural_shape,
            "const_sd":self.const_sd
        }
    def set_params(self, **params):
        for param,value in params.items():
            self.__setattr__(param,value)
        return self
    def optimize(self,X=None,y=None):
        self.X = X
        self.y = y
        if self.archive == None:
            self.archive = construct_solution(self.number_of_solutions,self.neural_shape)
        self.sorted_archive = self.calculate_fitness(self.score_fn,self.archive)
        weights = self.calculate_weights(self.archive.shape)
        self.archive = self.sampling_more(self.sorted_archive,weights,self.epsilon)
        self.sorted_archive = self.calculate_fitness(self.score_fn,self.archive)
        for i in np.arange(self.max_epochs):
            # try:
                # print summary_writter[-1]
            if(self.sorted_archive==None):
                return self.archive[0]
            self.archive = self.sampling_more(self.sorted_archive,weights,self.epsilon)
            self.sorted_archive = self.calculate_fitness(self.score_fn,self.archive)
            # print "ACO - Epoch %s. Training loss: %s"%(i,self.best_loss)
            #self.monitor.update(step=i,loss=self.best_loss)
            # except Exception as e:
            #     print e
            #     break
        sys.stdout.write("Found best loss %f"%self.monitor.min_loss)
        return self.best_archive

    def fit(self, X, y,**param):

        self.X = X
        self.y = y
        if (param.has_key('neural_shape')):
            self.neural_shape = param.get("neural_shape")
            self.n_output = self.neural_shape[-1]
            self.n_hidden = self.neural_shape[1:-1]
            self.number_of_layers = len(self.neural_shape)
        else:
            self.n_input = len(X[0])
            self.n_output = len(y[0])
            self.neural_shape = self.hidden_nodes.tolist()
            self.neural_shape.insert(0, self.n_input)
            self.neural_shape.append(self.n_output)
            self.n_hidden = self.hidden_nodes
        self.archive = param.get("archive")
        if param.has_key("top_k"):
            self.top_k = param.get("top_k")
        self.score_fn = FeedFlow(self.neural_shape)
        self.score_fn.set_weights(self.optimize(X,y))
        return self

    def predict(self,X):
        return self.score_fn.flow(X)
    def score(self,X,y):
        return self.score_fn.score(X,y)
    def calculate_fitness(self,score_fn,archive):
        fitness_solution = np.zeros(archive.shape[0])
        for (index,candidate) in enumerate(archive):
            score_fn.set_weights(candidate)
            result = score_fn.score(self.X,self.y)
            # print result
            fitness_solution[index] = result
        min_score = fitness_solution.min()
        # print min_score
        if(min_score >= self.error_criteria):
            return None
        sorted_idx = np.argsort(fitness_solution)
        sorted_archive = np.zeros(archive.shape)
        for (index,item) in enumerate(sorted_idx):
            sorted_archive[item] = archive[index]
        if min_score < self.best_loss :
            self.best_loss = min_score
            self.best_archive = sorted_archive[0]
        return sorted_archive
    # In[102]:

    def calculate_weights(self,shape):
        weights = np.zeros([shape[0],1])
        # qk = 0.1
        co_efficient = 1.0 / (0.1 * np.sqrt(2*np.pi))
        for index in np.arange(shape[0]):
            exponent = np.square(index-1) / (2*np.square(self.Q*shape[0]))
            weights[index] = np.multiply(co_efficient,np.exp(-exponent))
        return weights

    def compute_standard_deviation(self,i,l,archive,epsilon):
        __doc__ = " compute standard deviation with i, l, archive and epsilon"
        #Constant sd
        sd = 0.01
        sum_dev = np.abs(np.sum(archive[l]-archive[l][i])/(archive.shape[0]-1))
        # if(sum_dev <= 0.00001):
        #    return sd
        return np.multiply(sum_dev,epsilon)
    # In[104]:

    def choose_pdf(self, archive_shape, weights):
        sum_weights = np.sum(weights)
        temp = 0
        l = 0
        pro_r = np.random.uniform(0.0, 1.0)
        for (index,weight) in enumerate(weights):
            temp = temp + weight/sum_weights
            if(temp > pro_r):
                l = index
        return l

    def sampling_more(self, archive, weights, epsilon):
        # pdf = 0
        len_of_weights = range(archive.shape[1])
        next_archive = np.zeros(archive.shape)
        for index in np.arange(archive.shape[0]):
            i_pdf = self.choose_pdf(archive.shape,weights)
    #         for item in np.arange(archive.shape[1]):
    #             sigma = self.compute_standard_deviation(item,i_pdf,archive,epsilon)
    #             mu = archive[pdf][item]
    # #             print sigma,mu
    #             next_archive[index][item] = np.random.normal(mu,sigma)
            next_archive[index] = [np.random.normal(archive[index,item],self.compute_standard_deviation(item,i_pdf,archive,epsilon)) for item in len_of_weights]
        return next_archive
