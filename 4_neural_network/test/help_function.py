import numpy as np

def sigmoid(z):
    """ z: vector / Numpy array --> Ham se ap dung tren toan cac phan tu --> Vectorized form """
    return 1.0 / (1.0 + np.exp(-z))

