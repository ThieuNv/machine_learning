import numpy as np
from sklearn.metrics import mean_squared_error
import theano
class FeedFlow:
    def __init__(self,neural_shape=None,uniform_init=True,activation=None):
        print "Initilization"
        if(activation==None):
            self.activation = self.relu
        else:
            self.activation = self.relu
        self.neural_shape = neural_shape
        self.number_of_layers = len(neural_shape)
        self.weight_layers = [ (self.neural_shape[t-1],self.neural_shape[t]) for t in range(1,len(self.neural_shape))]
        self.bias_layers = [self.neural_shape[t] for t in range(1,len(self.neural_shape))]
        total_nodes_per_layer = zip(self.weight_layers,self.bias_layers)
        self.total_nodes = 0
        for layer in total_nodes_per_layer:
            self.total_nodes += (layer[0][0]+1)*layer[0][1]
        self.W, self.b = self.initialize_params(self.weight_layers,self.bias_layers)
    def initialize_params(self,weights_layer,bias_layer):
        W = []
        b = []
        for weight in weights_layer:
            W.append(self.xavier_initiliazer(weight))
        for bias in bias_layer:
            b.append(np.zeros(bias))
        return W,b
    def xavier_initiliazer(self,shape=None,uniform=True):
        seed = np.random.randint(1,self.number_of_layers*1E2)
        random_engine = np.random.RandomState(seed=seed)
        if(uniform==True):
            range_init = np.sqrt(6.0/(shape[0]+shape[1]))
            return random_engine.uniform(-range_init,range_init,size=(shape[0],shape[1]))
        else:
            stdv = np.sqrt(3.0/(shape[0]+shape[1]))
            return random_engine.normal(0.0,scale=stdv,size=(shape[0],shape[1]))
    def set_weights(self,weights_matrix):
        if(len(weights_matrix)!=self.total_nodes):
            print "Check again weights shape, must be equal with total nodes"
            return
        self.total_nodes_per_layer = zip(self.weight_layers,self.bias_layers)
        current_pos = 0
        W = []
        b = []
        for layer in self.total_nodes_per_layer:
            total_nodes = (layer[0][0]+1)*layer[0][1]
            weights = weights_matrix[current_pos:total_nodes+current_pos]
            current_pos = total_nodes
            b.append(weights[-layer[1]:])
            W.append(np.array(weights[:-layer[1]].reshape(layer[0])))
        self.W = W
        self.b = b
        return W,b
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def relu(self,x):
        return theano.tensor.nnet.relu(np.array(x))
    def activator(self,X,weights,bias,activator_fn=None):
        output = np.add(np.dot(X,weights),bias)
        if(activator_fn==None):
            return output
        return activator_fn(output)
    def flow(self,X):
        param_network = zip(self.W,self.b)
        pipe_activator = [self.activation for t in range(self.number_of_layers-1)]
        pipe_activator[-1] = None
        input_flow = X
        output = []
        for (index,param) in enumerate(param_network):
            output = self.activator(input_flow,param[0],param[1],pipe_activator[index])
            input_flow = output
        return output
    def predict(self,X):
        return self.flow(X)
    def score(self,X=None,y=None):
        # if(y==None):
        #     y = self.y
        # if(X==None):
        #     X = self.X
        return np.sqrt(mean_squared_error(y,self.flow(X)))
    def construct_candidate(self):
        W,b = self.initialize_params(self.weight_layers,self.bias_layers)
        param_network = zip(W,b)
        total_weights = []
        for param in param_network:
            temp_weight = np.concatenate([param[0].flatten(),param[1].flatten()])
            total_weights.append(temp_weight)
        return np.concatenate([total_weights[0],total_weights[1]]).flatten()
    def construct_solution(self,number_of_solutions):
        return np.array([self.construct_candidate() for t in range(number_of_solutions)])
    def optimize(self,algorithm,X,y,summary_writter=None):
        self.X = X
        self.y = y
        weights_matrix = algorithm.optimize(self,summary_writter,X=X,y=y)
        return self.set_weights(weights_matrix)
