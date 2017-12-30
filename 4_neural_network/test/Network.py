import numpy as np
import random
import help_function

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)    # So luong neuron trong cac layer tuong ung
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]    #
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        print self.num_layers
        print self.biases
        print self.weights

        # np.random.randn: phan phoi Gauss voi mean = 0, do lech chuan = 1
        # net = Network( [2, 3, 1] )    => 2 neurons trong first layer, 3 neurons trong second layer, 1 neuron in last


    def feedforward(self, a):
        """ Return the ouput of the network if 'a-activation' is input """
        for b, w in zip(self.biases, self.weights):
            a = help_function.sigmoid(np.dot(w, a) + b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.
        "training_data": list of tuples "(x, y)" (training inputs, desired outputs).
        epochs: So luong epoch dung train
        mini_batch_size: Kich thuoc mini-batch
         eta: learning rate
        Neu co' test_data --> NN se danh gia sau moi~ epoch. Dieu nay huu ich cho viec track progress nhung lam cham NN.
        """

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [ training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)     # update sau moi chay xong 1 mini-batch

            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)


    def update_mini_batch(self, mini_batch, eta):
        """ Moi~ epoch, tron random training data va chia chung thanh cac mini-batch.
        Cap nhat w, b dung GD vs BP doi voi single mini-batch.
        mini_batch: list of tuples "(x, y)",
        eta : is the learning rate.

        Tinh toan gradient cho moi training example trong mini_batch va update w, b tuong ung
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)      # Tinh gradient cua ham cost
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


nn = Network( [2, 3, 1] )

