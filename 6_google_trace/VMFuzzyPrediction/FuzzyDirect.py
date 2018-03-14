import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from estimators.NeuralFlow import *
from utils.SlidingWindowUtil import SlidingWindow
import skflow as sf
from sklearn import datasets, metrics
from sklearn.metrics import mean_absolute_error
from utils.GraphUtil import *
from sklearn.preprocessing import OneHotEncoder

scaler = MinMaxScaler()
dat = scaler.fit_transform(pd.read_csv('sample_610_10min_unnormalize.csv',parse_dates=True,index_col=0,names=['cpu_rate'])['cpu_rate'][:3000])
dat = pd.Series(dat.round(4))

partition_size = 0.001
umin = math.floor(min(dat));
umax = math.ceil(max(dat));
# 2: Partition of universe
# Method: Dividing in the half-thousands
def get_midpoint(ptuple):
    return 0.5*(ptuple[0]+ptuple[1])
def get_midpoint_vector(tuple_vector):
    return [get_midpoint(x) for x in tuple_vector];
def get_fuzzy_class(point, partition_size):
    return int(math.floor(point / partition_size))
def get_fuzzy_dataset(data):
    u_class = []
    for item in data:
        u_class.append(get_fuzzy_class(item,partition_size))
    return u_class
def mapping_class(u_class):
    unique_class = np.unique(u_class)
    index = np.arange(unique_class.shape[0])
    inverted = {}
    mapping = {}
    for idx,val in enumerate(unique_class):
        mapping[val] = idx
        inverted[idx] = val
    return mapping, inverted
def defuzzy(index, inverted,midpoints):
    f_class = inverted[index]
    return midpoints[f_class]

sliding_number = 2
# result = []
for sliding_number in np.arange(2,11):

	nIter = int((umax-umin)/partition_size)
	u_vectorized = []

	for i in range(nIter) :
	    u_vectorized.append((umin + i*partition_size,umin + (i+1)*partition_size));

	u_midpoints = get_midpoint_vector(u_vectorized)
	u_class = np.array(get_fuzzy_dataset(dat),dtype=np.int32)

	u_unique_inverted, u_unique_mapping = mapping_class(u_class)
	u_class_transform = [u_unique_inverted[item] for item in u_class]

	X_train_size = int(len(u_class_transform)*0.7)
	sliding = np.array(list(SlidingWindow(u_class_transform, sliding_number)))
	sliding = np.array(sliding, dtype=np.int32)
	X_train = sliding[:X_train_size]
	y_train = u_class_transform[sliding_number:X_train_size+sliding_number]
	X_test = sliding[X_train_size:]
	y_test = u_class_transform[X_train_size+sliding_number-1:]
	y_actual_test = dat[X_train_size+sliding_number-1:].tolist()
	# # Define classifier
	n_hidden = len(u_unique_inverted) + sliding_number
	# 

	classifier = NeuralFlowRegressor(hidden_nodes=[n_hidden], n_classes=len(u_unique_inverted),optimize='Adam'
		                         ,steps=10000,learning_rate=1E-02, activation='relu')
	a = classifier.fit(X_train, y_train)
	ypred = classifier.predict(X_test)
	ypred_defuzzy = [defuzzy(item,u_unique_mapping,u_midpoints) for item in ypred]
	score = mean_absolute_error(ypred_defuzzy,y_actual_test)

	np.savez('fuzzy_neuro_%s_%s'%(sliding_number,score),y_pred=ypred_defuzzy,y_true=y_actual_test)
