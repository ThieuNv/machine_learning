import numpy as np
__ALL__ = [
    'X_train',
    'y_train',
    'X_test',
    'y_test'
]
def load_training_from_npz(name):
    files = np.load(name)
    data_load = [files[key] for key in __ALL__]
    return data_load