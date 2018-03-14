# from lasagne.nonlinearities import sigmoid
from pyevolve import Crossovers
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Initializators, Mutators
from pyevolve import Selectors
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from FeedFlow import FeedFlow
import numpy as np
class GAEstimator(BaseEstimator):
    def __init__(self,gen_size=400,pop_size = 225,cross_rate=0.9,mutation_rate = 0.01,freq_stats=10):
        # self.n_input = n_input
        # self.fan_in = n_input
        # self.fan_out = 15
        # self.theta_shape = (self.n_input,1)
        self.gen_size = self.number_of_solutions = gen_size
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.freq_stats = freq_stats
    def get_params(self, deep=True):
        return {
            "gen_size": self.gen_size,
            "pop_size": self.pop_size,
            "cross_rate": self.cross_rate,
            "mutation_rate": self.mutation_rate,
            "freq_stats": self.freq_stats
        }
    def set_params(self, **params):
        for param,value in params.items():
            self.__setattr__(param,value)
        return self
    # def activation(self,x):
    #         return sigmoid(x)
    def eval_score(self,chronosome):
        # theta = np.zeros(self.theta_shape)
        # for i in np.arange(self.theta_shape[0]):
        #     theta[i] = chronosome[i]
        # return self.costFunction(self.X_data,self.y_data,theta)
        self.score_fn.set_weights(np.array(chronosome.genomeList))
        return self.score_fn.score(self.X,self.y)
    def fit(self,X,y,**param):
        self.neural_shape = param.get("neural_shape")
        self.n_input = self.neural_shape[0]
        self.n_output = self.neural_shape[-1]
        self.n_hidden = self.neural_shape[1]
        self.number_of_weights = self.n_hidden*(self.n_input+1)+self.n_output*(self.n_hidden+1)
        self.score_fn = FeedFlow(self.neural_shape)
        self.X = X
        self.y = y
        #setting params
        self.weights = G1DList.G1DList(self.number_of_weights)
        lim = np.sqrt(6)/np.sqrt((self.n_input+self.n_output))
        #Khoi tao trong so
        self.weights.setParams(rangemin=-lim,rangemax=lim)
        # cai dat ham khoi tao
        self.weights.initializator.set(Initializators.G1DListInitializatorReal)
        #cai dat ham dot bien
        self.weights.mutator.set(Mutators.G1DListMutatorRealGaussian)
        #cai dat ham do thich nghi
        self.weights.evaluator.set(self.eval_score)
        # cai dat ham lai ghep
        self.weights.crossover.set(Crossovers.G1DListCrossoverUniform)
        #code genetic
        # thiet lap he so lai ghep
        self.ga = GSimpleGA.GSimpleGA(self.weights)
        self.ga.selector.set(Selectors.GRouletteWheel)
        self.ga.setMutationRate(self.mutation_rate)
        self.ga.setCrossoverRate(self.cross_rate)
        self.ga.setPopulationSize(self.pop_size)
        self.ga.setGenerations(self.pop_size)
        self.ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
        self.ga.evolve(freq_stats=self.freq_stats)
        self.best_archive = self.getParam()
        return self
    def getParam(self):
        return np.array(self.ga.bestIndividual().genomeList)
    def predict(self,X):
        return self.score_fn.predict(X)
    def score(self,X,y):
        return np.sqrt(mean_squared_error(y, self.predict(X)))