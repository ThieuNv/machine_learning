from __future__ import division
import numpy as np
from sklearn.metrics.base import *
def mean_absolute_percentage_error(y_pred=None,y_true=None):
    # sum_iter = 0
    sum_iter = [abs((a_pred-a_true)/a_true) for a_pred , a_true in zip(y_pred,y_true)]
    return sum(sum_iter)/(len(sum_iter))

