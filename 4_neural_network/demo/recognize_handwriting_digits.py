"""
recognize_handwriting_digits.py
~~~~~~~~~~
Thuc hien giai thuat stochastic gradient descent cho mang feedforward neural network.
Gradients duoc tinh bang backpropagation.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """ list ``sizes`` : gom so luong neuron trong tung cac tang cua mang.
        Vd: Mang co 3 tang vs so luong lan luot la: 2, 3, 1  ==> sizes = [2, 3, 1]
        biases, weights: khoi tao ngau nhien dung phan phoi Gauss vs mean = 0, variance = 1
        Note: first_layer = input layer -> ko co biases.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a: activation`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """ Train network su dung: mini-batch SGD.
        `training data`: list tuples (x-training input, y-desired output)
        Neu truyen `test_data` thi mang se danh gia lai test data sau moi epoch, cac phan cai tien se~ duoc in ra.
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)

            mini_batches = [ training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches: self.update_mini_batch(mini_batch, eta)

            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """ Cap nhat w, b dung GD su dung giai thuat BP voi tung mini-batch
        `mini_batch`: list tuple(x-input, y-desired).
        `eta`: learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """ Tra ve tuple `(nabla_b, nabla_w)` : gradient cua ham cost C_x
        `nabla_b` and `nabla_w` : layer-by-layer lists cua numpy arrays, giong nhu `self.biases` va `self.weights`.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]   # list luu toan bo cac activations, layer-by-layer
        zs = []             # list luu toan bo cac z vectors, layer-by-layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Chi so l nghia la: l = 1 -> last layer of neurons. l = 2 -> second-last layer of neurons. ..
        # Danh chi so nhu vay vi python co the su dung chi so am trong lists
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """ Tra ve so test input ma output cua mang correct(dung) .
        ouput cua mang duoc gia su la chi so cua bat ky neuron nao trong lop cuoi cung
        co activation cao nhat.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Tra ve vector dao ham rieng cua C_x / dao ham rieng cua a cho output activations.
        """
        return (output_activations-y)

#### Helper functions
def sigmoid(z):
    """sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Dao ham cua sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))